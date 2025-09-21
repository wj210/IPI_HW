from rich.console import Console
from rich.panel import Panel
import torch
import numpy as np
import gc
from tqdm import tqdm
import concurrent.futures

qwen_tool_start_token = "<tool_response>"
qwen_tool_end_token = "</tool_response>"

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
    if 'qwen' in tokenizer.name_or_path.lower():
        formatted = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=enable_thinking,tools=tools)
    else:
        formatted = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,tools=tools)
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
                        spans.append((i+len(curr_start_ids), j+len(curr_end_ids))) # span is between start and end
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
    if is_error:
        print("Error in encoding aside segments!")
    encoded['segment_ids'] = segment_ids
    return encoded


def generate_fn(model, tokenizer, inps, max_new_tokens: int = 512,do_sample: str = False) -> str:
    with torch.no_grad():
        output_ids = model.generate(
            **inps,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            temperature = 0.
        )
    # Return only the newly generated tokens when possible
    gen_ids = output_ids[:, inps['input_ids'].shape[-1]:]
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
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