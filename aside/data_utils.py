from functools import partial
import sys
import os
sys.path.append(os.path.abspath("..")) 
from torch.utils.data import Dataset
import torch
from model import *
from model_api import *
from utils.utils import aside_encode,get_aside_segments,aside_encode_start,multiturn_aside_encode,tool_prompt_format
import torch.distributed as dist


class Alpaca_Embedding_Dataset(Dataset): # only for embeddings - ASIDE/ISE - only SFT
    def __init__(self, data: List[Dict],tokenizer, input_token,assistant_token,max_length=512,completion_only=False):
        self.data = data
        self.tokenizer = tokenizer
        self.input_token = input_token
        self.assistant_token = assistant_token
        self.max_length = max_length
        self.completion_only = completion_only
        self.tokenizer.padding_side = "right" # right padding for SFT
        self.printed_input = False
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        conv = [{'role':'user','content':data_point['instruction']},
                {'role':'input','content':data_point['input']},] # ensure there is a input role
        conv.append({'role':'assistant','content':data_point['output']})
        encoded = tool_prompt_format(conv,None,self.tokenizer,encode=True) 
        input_ids,attention_mask = encoded['input_ids'],encoded['attention_mask']
        encoded = aside_encode_start(encoded,self.tokenizer,self.input_token,key = 'segment_ids')
        segment_ids = encoded['segment_ids']
        if not self.printed_input: 
            print (f'FULL PROMPT: {conv}')
            segments = get_aside_segments(encoded,self.tokenizer)
            for i,s in enumerate(segments):
                print (f'Rotated segment {i}: {s}') 
            self.printed_input = True
        # Add eos
        encoded['input_ids'] = torch.hstack((input_ids, torch.Tensor([[self.tokenizer.eos_token_id]]))).long()
        encoded['attention_mask'] = torch.hstack((attention_mask, torch.Tensor([[True]]))).bool()
        encoded['segment_ids'] = torch.hstack((segment_ids, torch.Tensor([[1]]))).long()
        
        if self.completion_only:
            encoded = aside_encode_start(encoded,self.tokenizer,self.assistant_token,key = 'labels')
            # this is only a mask
            labels = encoded['input_ids'].clone()
            labels[encoded['labels'] == 0] = -100 # only compute loss on completion
            encoded['labels'] = labels
        
        # encoded['segment_ids']*= 0 # TEST
        return {k:v.squeeze(0) for k,v in encoded.items()}
        
        
class ToolDataset(Dataset):
    """
    Custom Dataset class for Tool use
    """

    def __init__(self, data: List[Dict], template_info: Dict, handler: CustomModelHandler, assistant_token, tool_token, user_token,max_length=4096,completion_only=True):
        self.data = data
        self.template_info = template_info
        self.handler = handler
        self.max_length = max_length
        self.printed_input = False
        self.assistant_token = assistant_token
        self.tool_token = tool_token
        self.user_token = user_token
        self.completion_only = completion_only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Prepares a single training example with instruction/data separation.
        
        Returns:
            Dict: Contains 'input_ids', 'attention_mask', and optionally 'segment_ids'
                  for ASIDE models. segment_ids indicate whether each token is part
                  of instructions (0) or data (1).
        """
        data_point = self.data[idx]
        conv = []
        data_key = 'from' if 'from' in data_point['conversations'][0] else 'role' # some datasets use 'role' instead of 'from'
        value_key = 'value' if 'value' in data_point['conversations'][0] else 'content'
        for x in data_point['conversations']:
            if x[data_key] == 'function': # convert to tool
                x[data_key] = 'tool'
            conv.append({'role':x[data_key],'content':x[value_key].lstrip('\n')}) # remove leading \n (ToolLLM have these)

        formatted = self.handler.tokenizer.apply_chat_template(conv,tokenize=False,add_generation_prompt=False,enable_thinking=False,tools=data_point['tools'])
        encoded = self.handler.tokenizer(formatted, add_special_tokens=False,return_tensors = 'pt',truncation=False)
        input_ids,attention_mask = encoded['input_ids'][0],encoded['attention_mask'][0]
        
        segment_ids = None
        # split into assistant and tool (tool is for ASIDE)
        to_iter = 2 if self.handler.embedding_type in ['ise','forward_rot'] else 1
        for token_type in range(to_iter):
            if token_type == 0:
                start,end = [self.assistant_token[0]], [self.assistant_token[1]]
                assistant_mask = multiturn_aside_encode(encoded,self.handler.tokenizer,start,end)[0]
                labels = input_ids.clone()
                labels[~assistant_mask.bool()] = -100  # Only compute loss on assistant parts
            else: # both
                start,end = [self.assistant_token[0],self.tool_token[0]], [self.assistant_token[1],self.tool_token[1]]
                segment_ids = multiturn_aside_encode(encoded,self.handler.tokenizer,start,end)[0]

        if not self.printed_input: 
            print (f'Prompt : {self.handler.tokenizer.decode(input_ids.tolist())}')
            if segment_ids is not None:
                chunks = get_aside_segments({'input_ids':[input_ids],'segment_ids':[segment_ids]},self.handler.tokenizer)
                for j,chunk in enumerate(chunks):
                    print (f'Rotated chunk {j}: {chunk}')
                
            ## just to get the segments
            if self.completion_only:
                assistant_chunks = []
                start = False
                for l in labels.tolist():
                    if l != -100 and not start:
                        start = True
                        assistant_chunks.append([l])
                    elif l != -100 and start:
                        assistant_chunks[-1].append(l)
                    else:
                        start = False
                for j,chunk in enumerate(assistant_chunks):
                    print (f'Assistant loss chunk {j}: {self.handler.tokenizer.decode(chunk)}')
            self.printed_input = True
        
        batch = {
            "input_ids": input_ids.tolist(), 
            "attention_mask": attention_mask.tolist(),
            # "labels": labels.tolist() # adding labels seem to incurr a high lost? # try to not add.
        }
        if self.completion_only:
            batch['labels'] = labels.tolist()
        if segment_ids is not None:
            batch["segment_ids"] = segment_ids.tolist()
        
        batch = {k:v[:self.max_length] for k,v in batch.items()} # trim to max length
        return batch
    

def check_input_format(sample): # needs to be a list of dict with conversations key and each conversation is in a chat format
    if 'conversations' not in sample:
        return False
    if not isinstance(sample['conversations'],list):
        return False
    if not isinstance(sample['conversations'][0],dict):
        return False
    return True

def input_formatting_fn(dataset):
    alpaca_conversation_dataset = []
    for sample in dataset:
        alpaca_conversation_dataset.append({'conversations':
            [{'role':'user','content':sample['instruction']},
             {'role':'input','content':sample.get('input','')}, # some datasets do not have input
             {'role':'assistant','content':sample['output']},],
            'tools': None})
    return alpaca_conversation_dataset
    
    
class Tool_Input_Dataset(Dataset): # allows both input and tool roles
    def __init__(self, data: List[Dict],tokenizer,embedding_type,token_args,completion_only=False):
        self.data = data
        self.tokenizer = tokenizer
        self.input_token = token_args['input_token']
        self.tool_token = token_args['tool_token']
        self.assistant_token = token_args['assistant_token']
        self.user_token = token_args['user_token']
        self.completion_only = completion_only
        self.tokenizer.padding_side = "right" # right padding for SFT
        self.printed_input = False
        self.embedding_type = embedding_type
        self.label_span = ([self.assistant_token[0]],[self.assistant_token[1]]) # for completion only
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        conv = data_point['conversations']
        roles = [x['role'] for x in data_point['conversations']]
        if 'input' in roles:
            rotate_start,rotate_end = [self.assistant_token[0],self.input_token[0]], [self.assistant_token[1],self.input_token[1]]
        else:
            rotate_start,rotate_end = [self.assistant_token[0],self.tool_token[0]], [self.assistant_token[1],self.tool_token[1]]
        
        encoded = tool_prompt_format(conv,data_point.get('tools',None),self.tokenizer,encode=True) 
        input_ids,attention_mask = encoded['input_ids'],encoded['attention_mask']

        segment_ids = None
        # split into assistant and tool (tool is for ASIDE)
        to_iter = 2 if self.embedding_type in ['ise','forward_rot'] else 1
        for token_type in range(to_iter):
            if token_type == 0:
                assistant_mask = multiturn_aside_encode(encoded,self.tokenizer,self.label_span[0],self.label_span[1])[0]
                labels = input_ids.clone()
                labels[~assistant_mask.bool()] = -100  # Only compute loss on assistant parts
            else: # both
                segment_ids = multiturn_aside_encode(encoded,self.tokenizer,rotate_start,rotate_end)[0]
        
        ## Demo 1 example
        if not self.printed_input:
            if dist.get_rank() == 0: 
                print (f'Prompt : {self.tokenizer.decode(input_ids.tolist())}')
                if segment_ids is not None:
                    chunks = get_aside_segments({'input_ids':[input_ids],'segment_ids':[segment_ids]},self.tokenizer)
                    for j,chunk in enumerate(chunks):
                        print (f'Rotated chunk {j}: {chunk}')
            self.printed_input = True
        
        
        batch = {
            "input_ids": input_ids.tolist(), 
            "attention_mask": attention_mask.tolist(),
        }
        if self.completion_only:
            batch['labels'] = labels.tolist()
        if segment_ids is not None:
            batch["segment_ids"] = segment_ids.tolist()

        return batch