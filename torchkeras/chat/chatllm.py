import torch
import sys 
from copy import deepcopy
from .conversations import conv_templates, get_conv_template 

#chat tool for chatglm2-6b,baichuan-13b,internlm-chat-7b,qwen-7b-chat and more...
class ChatLLM:
    def __init__(self,model,tokenizer,
                 model_type=None,
                 max_chat_rounds=20,
                 max_new_tokens=512,
                 stream=True,
                 history=None
                ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type if model_type else self.get_model_type() 
        conv = get_conv_template(self.model_type)
        self.stop_words_ids = [[w] for w in conv.stop_token_ids] if conv.stop_token_ids else []
        self.model.generation_config.stop_words_ids = self.stop_words_ids
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.model.eval()
        self.history = [] if history is None else history
        self.max_chat_rounds = max_chat_rounds
        self.stream = stream
        
        try:
            self.register_magic() 
            response = self('你好')
            if not self.stream:
                print(response)
            print('register magic %%chat sucessed ...',file=sys.stderr)
            self.history = self.history[:-1]
        except Exception as err:
            print('register magic %%chat failed ...',file=sys.stderr)
            raise err 
        
    def get_model_type(self):
        model_cls = str(self.model.__class__).split('.')[-1].lower()[:-2] 
        keys = list(conv_templates.keys()) 
        max_same,most_type = 0,None
        for k in keys:
            same = 0
            for a,b in zip(k,model_cls):
                if a==b:
                    same+=1
                else:
                    break
            if same>max_same:
                max_same = same
                most_type = k 
        if max_same>=3:
            return most_type
        else:
            raise Exception('Error: get_model_type failed @ model_cls='+model_cls)
            return None
        
    @classmethod
    def build_messages(cls,query=None,history=None,system=None):
        messages = []
        history = history if history else [] 
        if system is not None:
            messages.append({'role':'system','content':system})
        for prompt,response in history:
            pair = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
            messages.extend(pair)
        if query is not None:
            messages.append({"role": "user", "content": query})
        return messages
    
    def build_prompt(self,messages):
        model = self.model
        if not hasattr(model,'conv_template'):
            model_type = self.get_model_type()
            model.conv_template =get_conv_template(model_type)
        conv = deepcopy(model.conv_template)
        msgs_sys = [d for d in messages if d['role']=='system']
        if msgs_sys:
            conv.set_system_message(msgs_sys[0]['content'])

        for d in messages:
            if d['role']=='user':
                conv.append_message(conv.roles[0], d['content'])
            elif d['role']=='assistant':
                conv.append_message(conv.roles[1], d['content'])
            else:
                raise Exception('role must be one of (system,user,assistant)')

        if d['role']!='assistant':
            conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
        
    def chat(self, messages, stream=False, generation_config=None):
        model,tokenizer = self.model,self.tokenizer
        prompt = self.build_prompt(messages)
        
        inputs = tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(model.device) for k, v in inputs.items()}
        if generation_config is not None:
            generation_config = deepcopy(model.generation_config.update(**generation_config))
        else:
            generation_config =  deepcopy(model.generation_config)

        stop_words_ids = model.generation_config.stop_words_ids
        stop_set = set([x for a in stop_words_ids for x in a])

        if not stream:
            from transformers import PreTrainedModel
            model.__class__.generate = PreTrainedModel.generate  # disable stream
            output_ids = model.generate(
                 **inputs,
                 stop_words_ids = stop_words_ids,
                 generation_config = generation_config,
                 return_dict_in_generate = False,
            )
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
            end_token_idx = 0
            for end_token_idx in range(len(output_ids)):
                if output_ids[end_token_idx].item() in stop_set:
                    break
            outputs = tokenizer.decode(
                output_ids[:end_token_idx], skip_special_tokens=True
            )
            return outputs
        else:
            from .stream_generate  import NewGenerationMixin, StreamGenerationConfig
            model.__class__.generate = NewGenerationMixin.generate
            model.__class__.sample_stream = NewGenerationMixin.sample_stream
            stream_config = StreamGenerationConfig(**generation_config.to_dict(),do_stream=True)
            
            def stream_generator():
                outputs = []
                for token in model.generate(**inputs,
                                            generation_config=stream_config,
                                            do_sample=True,
                                            stop_words_ids=stop_words_ids,
                                            return_dict_in_generate = False,
                                           ):
                    token_idx = token.item()
                    outputs.append(token_idx)
                    if token_idx in stop_set:
                        break 
                    yield tokenizer.decode(outputs, skip_special_tokens=True)
            return stream_generator()
        
    def __call__(self,query):
        from IPython.display import display,clear_output 
        len_his = len(self.history)
        if len_his>=self.max_chat_rounds+1:
            self.history = self.history[len_his-self.max_chat_rounds:]
        messages = self.build_messages(query=query,history=self.history)
        if not self.stream:
            response = self.chat(messages,stream=False)
            self.history.append((query,response))
            return response 
        
        result = self.chat(messages,stream=True)
        for response in result:
            print(response)
            clear_output(wait=True)
        self.history.append((query,response))
        return response
    
    
    def register_magic(self):
        import IPython
        from IPython.core.magic import (Magics, magics_class, line_magic,
                                        cell_magic, line_cell_magic)
        @magics_class
        class ChatMagics(Magics):
            def __init__(self,shell, pipe):
                super().__init__(shell)
                self.pipe = pipe

            @line_cell_magic
            def chat(self, line, cell=None):
                "Magic that works both as %chat and as %%chat"
                if cell is None:
                    return self.pipe(line)
                else:
                    print(self.pipe(cell))       
        ipython = IPython.get_ipython()
        magic = ChatMagics(ipython,self)
        ipython.register_magics(magic)
        