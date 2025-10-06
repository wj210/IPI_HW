from rich.console import Console
from rich.panel import Panel
import torch
import numpy as np
import gc
from tqdm import tqdm
import concurrent.futures
from copy import deepcopy

qwen_tool_start_token = "<tool_response>"
qwen_tool_end_token = "</tool_response>"

alpaca_format = "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\nInstruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"

def seed_all():
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

def pprint(text):
    """Pretty print the model's generated text using rich."""
    console = Console(width=100)
    console.print(Panel(text, border_style="blue"))

def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()
    
def encode_fn(texts,tokenizer):
    return tokenizer(texts,return_tensors = 'pt',padding='longest',truncation=False)

def google_encode_fn(messages,tokenizer): # only for agentdojo!
    if messages[0]['role'] == 'system':
        system_prompt = messages[0]['content'] + ' The function results will be delimited between "<function_result>" and "</function_result>".'
        messages = messages[1:]
    if messages[0]['role'] == 'user':
        messages[0]['content'] = system_prompt + '\n\n' + messages[0]['content']
    for m in messages:
        if m['role'] not in ['user','assistant']: # tool
            m['content'] = f"<tool_call>\n{m['content']}\n</tool_call>"
            m['role'] = 'user' # treat it like a user message
    
    formatted = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    return formatted
    
def tool_prompt_format(messages,tools,tokenizer,encode=False,enable_thinking=False): # for tool-use agents like Qwen
    if messages[-1]['role'] == 'assistant':
        add_generation_prompt = False
    else:
        add_generation_prompt = True
    if 'qwen' in tokenizer.name_or_path.lower():
        formatted = tokenizer.apply_chat_template(messages,add_generation_prompt=add_generation_prompt,tokenize=False,enable_thinking=enable_thinking,tools=tools)
    else:
        formatted = tokenizer.apply_chat_template(messages,add_generation_prompt=add_generation_prompt,tokenize=False,tools=tools)
    return encode_fn(formatted,tokenizer) if encode else formatted

def aside_encode(encoded,tokenizer,start_token=None,end_token=None):
    if 'qwen' in tokenizer.name_or_path.lower():
        start_token = qwen_tool_start_token if start_token is None else start_token
        end_token = qwen_tool_end_token if end_token is None else end_token
    assert start_token is not None and end_token is not None, "start_token and end_token must be provided"
    device = encoded['input_ids'].device
    if isinstance(start_token, str):
        start_ids = torch.tensor(tokenizer.encode(start_token,add_special_tokens=False)).to(device)
        end_ids = torch.tensor(tokenizer.encode(end_token,add_special_tokens=False)).to(device)
    else: # is a list of same length as start_token
        start_ids = [torch.tensor(tokenizer.encode(t,add_special_tokens=False)).to(device) for t in start_token]
        end_ids = [torch.tensor(tokenizer.encode(t,add_special_tokens=False)).to(device) for t in end_token]

    is_error = False
    segment_ids = torch.zeros_like(encoded['input_ids'])
    for jj, input_ids in enumerate(encoded['input_ids']):
        i = 0
        spans = []
        if isinstance(start_ids, list):
            curr_start_ids = start_ids[jj]
            curr_end_ids = end_ids[jj]
        else:
            curr_start_ids = start_ids
            curr_end_ids = end_ids
        while i <= len(input_ids) - len(curr_start_ids):
            if torch.equal(input_ids[i:i+len(curr_start_ids)], curr_start_ids):
                # locate matching end after this start
                j = i + len(curr_start_ids)
                found_end = False
                while j <= len(input_ids) - len(curr_end_ids):
                    if torch.equal(input_ids[j:j+len(curr_end_ids)], curr_end_ids):
                        spans.append((i, j+len(curr_end_ids))) # span is between start and end
                        found_end = True
                        i = j + len(curr_end_ids)
                        break
                    j += 1
                if not found_end:
                    # No closing tag; don't label this open span to avoid corruption
                    i += len(curr_start_ids)
            else:
                i += 1
        if len(spans):
            for span in spans:
                segment_ids[jj,span[0]:span[1]] = 1
        else:
            is_error = True
    # if is_error:
    #     print("Error in encoding aside segments!")
    encoded['segment_ids'] = segment_ids
    return encoded

qwen_start_tokens = ["\n<|im_start|>assistant","\n<|im_start|>user\n<tool_response>"]
qwen_end_tokens = ["<|im_end|>\n","</tool_response><|im_end|>\n"]

def multiturn_aside_encode(encoded,tokenizer,start_tokens=[],end_tokens=[],until_last_token=None):
    """
    we have a list of start and end_tokens, all of these should be rotated. This is to account for the uncontrollable turn order in agentic tool use, such as tool -> assistant or tool -> user
    """
    if 'qwen' in tokenizer.name_or_path.lower(): # default for qwen
        if len(start_tokens) == 0:
            start_tokens = qwen_start_tokens
            # until_last_token = qwen_start_tokens[0] if until_last_token is None else until_last_token
        if len(end_tokens) == 0:
            end_tokens = qwen_end_tokens
    
    assert any([tool_t in start_tokens[-1] for tool_t in ['tool_response','reference_data']]),f'Last start and end token should point to the input/tool.' # ensure last is tool or input
    device = encoded['input_ids'].device
    start_ids = [torch.tensor(tokenizer.encode(start_token,add_special_tokens=False)).to(device) for start_token in start_tokens]
    end_ids = [torch.tensor(tokenizer.encode(end_token,add_special_tokens=False)).to(device) for end_token in end_tokens]
    until_last_ids = torch.tensor(tokenizer.encode(until_last_token,add_special_tokens=False)).to(device) if until_last_token is not None else None
    
    # to ensure we first look for tool and note down which pos ids does it start from.
    start_ids = start_ids[::-1] # reverse the order, so that we first look for the last start token, which should be the tool/input
    end_ids = end_ids[::-1]

    assert len(start_tokens) == len(end_tokens) != 0, "start_tokens and end_tokens must be provided and of same length"
    # assert until_last_token is not None, "until_last_token must be provided, this is the start token that is allowed to be unclosed - should be the assistant token."

    mask = torch.zeros_like(encoded['input_ids'])
    for jj, input_ids in enumerate(encoded['input_ids']):
        curr_input_str = tokenizer.decode(input_ids)
        if start_tokens[-1] not in curr_input_str:
            continue # no tool/input, skip the rotation
        tool_ids = -1 # note down where is the tool/input start
        for curr_start_ids,curr_end_ids in zip(start_ids,end_ids):
            i = 0
            while i <= len(input_ids) - len(curr_start_ids):
                if torch.equal(input_ids[i:i+len(curr_start_ids)], curr_start_ids):
                    j = i + len(curr_start_ids)
                    found = False
                    while j <= len(input_ids) - len(curr_end_ids):
                        if torch.equal(input_ids[j:j+len(curr_end_ids)], curr_end_ids):
                            if torch.equal(curr_start_ids, start_ids[0]): # mark when a complete tool span is found
                                tool_ids = deepcopy(j+len(curr_end_ids))
                                mask[jj,i:j+len(curr_end_ids)] = 1
                            elif torch.equal(curr_start_ids, start_ids[-1]) and i > tool_ids: # if we are only looking for assistant span, then look after the tool span
                                mask[jj,i:j+len(curr_end_ids)] = 1
                            i = j + len(curr_end_ids)
                            found = True
                            break
                        j += 1
                    if not found:
                        if until_last_ids is not None and torch.equal(input_ids[i:i+len(until_last_ids)], until_last_ids):
                            # allow to be unclosed, mark until end of input
                            mask[jj,i:len(input_ids)] = 1
                            break
                        else:
                            i += len(curr_start_ids)
                else:
                    i += 1
    return mask

def aside_encode_start(encoded,tokenizer,start_token,key = 'segment_ids'): # given a start token, mark everything from that onwards as 1.
    device = encoded['input_ids'].device
    start_ids = torch.tensor(tokenizer.encode(start_token,add_special_tokens=False)).to(device)
    mask = torch.zeros_like(encoded['input_ids'])
    for jj, input_ids in enumerate(encoded['input_ids']): # assume only one span
        i = 0
        while i <= len(input_ids) - len(start_ids):
            if torch.equal(input_ids[i:i+len(start_ids)], start_ids):
                mask[jj,i:] = 1
                break
            else:
                i += 1
    encoded[key] = mask
    return encoded

def get_aside_segments(encoded,tokenizer,track_val = 1):
    input_ids = encoded['input_ids'][0]
    segment_ids = encoded['segment_ids'][0]
    start = False
    spans = []
    for i,s in zip(input_ids.tolist(),segment_ids.tolist()):
        if s == track_val and not start:
            start = True
            spans.append([i])
        elif s == track_val and start:
            spans[-1].append(i)
        else:
            start = False
    decoded = [tokenizer.decode(s) for s in spans]
    return decoded

def print_aside(encoded,tokenizer):
    to_print = []
    start = False
    for i,l in zip(encoded['input_ids'][0],encoded['segment_ids'][0]):
        if l == 1 and not start:
            to_print.append([i.item()])
            start = True
        elif l == 1 and start:
            to_print[-1].append(i.item())
        elif l == 0:
            start = False
    pprint (f'FULL PROMPT: {tokenizer.decode(encoded["input_ids"][0])}')
    if len(to_print) > 0:
        for i,seg in enumerate(to_print):
            pprint (f"SEGMENT {i}: {tokenizer.decode(seg)}")


def generate_fn(model, inps,gen_kwargs,use_tqdm=False) -> str:
    with torch.no_grad():
        output_ids = model.generate(
            **inps,
            **gen_kwargs,
        )
    # Return only the newly generated tokens when possible
    gen_ids = output_ids[:, inps['input_ids'].shape[-1]:]
    text = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return text

def vllm_generate(model,texts,gen_kwargs,use_tqdm=False):
    if not isinstance(texts,list):
        texts = [texts]
    out = model.generate(texts, sampling_params = gen_kwargs,use_tqdm = use_tqdm)
    return [o.outputs[0].text for o in out]

def async_process(fn,inps,workers=10,msg=''):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        if len(msg):
            out = list(tqdm(executor.map(fn,inps),total = len(inps),desc = msg))
        else:
            out = list(executor.map(fn,inps))
    return out