from ast import arg
import os,sys
sys.path.append(os.path.abspath(".")) 
sys.path.append(os.path.abspath("aside"))
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.utils import *
from utils.model_utils import load_model
torch.set_grad_enabled(False)
from vllm import SamplingParams
from functools import partial
import argparse
from datasets import load_dataset
from constants import *

def format_instruction_data(instr,data): # requires the additional input key.
    return [
        {'role':'user','content':instr},
        {'role':'input','content':data}
    ]
    
def avg_results(results):
    return {k:np.mean(v) for k,v in results.items()}

def format_mcq(sample):
    choices = '\n'.join([f"({chr(i+65)}) {c}" for i,c in enumerate(sample['choices'])])
    instruction = f'Instruction: {sample["instruction"]}\n\nChoices:\n{choices}'
    sample['instruction'] = instruction
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default=None,help='model path')
    parser.add_argument('--tasks',type=str,default=['reclor','mmlu_pro','gpqa'],help='which tasks to eval, split by ,')
    parser.add_argument('--use_vllm',action = 'store_true',help='whether to use vllm')
    parser.add_argument('--batch_size',type=int,default=16,help='batch size if not using vllm')
    args = parser.parse_args()
    seed_all()
    
    model_path = args.model_path
    if 'Qwen/' not in model_path:
        model_path = os.path.join(MODEL_DIR,model_path)
    torch_dtype = torch.bfloat16
    model,tokenizer,is_aside = load_model(model_path,use_vllm=args.use_vllm,dtype=torch_dtype,vllm_kwargs = {'gpu_memory_utilization':0.8,'enable_chunked_prefill':True})
    print (f'is_aside: {is_aside}')
    if is_aside:
        args.use_vllm = False # not yet supported
    
    # setup generate fn for either vllm or HF
    gen_fn = vllm_generate 
    gen_kwargs = SamplingParams(temperature=0.,max_tokens=1,stop=[tokenizer.eos_token]) 
    
    eval_ds = {}
    if 'reclor' in args.tasks:
        ## ReClor dataset
        reclor_ds = load_dataset("metaeval/reclor",split = 'validation').to_list()
        for d in reclor_ds:
            d['instruction'] = d.pop('question')
            d['input'] = d.pop('context')
            d['choices'] = d.pop('answers')
            d['answer'] = chr(d['label'] + 65)
            d = format_mcq(d) # format the instruction
        
        eval_ds['reclor'] = reclor_ds
    
    if 'mmlu_pro' in args.tasks:
        mmlu_ds = load_dataset("TIGER-Lab/MMLU-Pro")
        for d in mmlu_ds:
            d['instruction'] = d.pop('question')
            d['choices'] = d.pop('options')
            d['answer'] = d.pop('answer')
            d = format_mcq(d)

        eval_ds['mmlu_pro'] = mmlu_ds

    if 'gpqa' in args.tasks:
        gpqa_ds_raw = load_dataset("Idavidrein/gpqa", "gpqa_main",split = 'train').to_list()
        gpqa_ds = []
        for d in gpqa_ds_raw:
            choices = [d['Correct Answer'],d['Incorrect Answer 1'],d['Incorrect Answer 2'],d['Incorrect Answer 3']]
            ans_str = d['Correct Answer']
            random_ids = np.random.permutation(len(choices))
            choices = [choices[i] for i in random_ids]
            ans = choices.index(ans_str)
            gpqa_ds.append(
                {
                    'instruction': d['Question'],
                    'choices': choices,
                    'answer': chr(65 + ans),  # Convert to A, B, C, D
                }
            )
        eval_ds['gpqa'] = gpqa_ds
        
    for task,dataset in eval_ds.items():
        print (f'Eval {task} with {len(dataset)} samples')
    
    # Eval
    
    def eval_mcq(dataset,batch_size=-1):
        acc = []
        batch_size = len(dataset) if batch_size == -1 or args.use_vllm else batch_size # if use vllm, use full batch
        for i in tqdm(range(0,len(dataset),batch_size),total = len(dataset)//batch_size):
            batch = dataset[i:i+batch_size]
            answer = [d['answer'] for d in batch]
            instrs = [d['instruction'] for d in batch]
            inputs = [d['input'] for d in batch]
            prompts = [tool_prompt_format(format_instruction_data(inst,inp),tools=None,tokenizer=tokenizer,encode=False) for inst,inp in zip(instrs,inputs)]
            
            prompts = [prompt + 'The answer is (' for prompt in prompts] # add this suffix
            
            if not args.use_vllm:
                prompts = encode_fn(prompts,tokenizer)
            
            pred = gen_fn(model,prompts,gen_kwargs,use_tqdm=True)
            acc.extend([p.lower() == a.lower() for p,a in zip(pred,answer)])
        return acc
    
    result_store = {}
    for task,dataset in eval_ds.items():
        acc = np.mean(eval_mcq(dataset,batch_size=args.batch_size))
        print (f'{task} Acc: {acc:.2f}%')
        result_store[task] = np.round(acc,3)
        
    result_dir = 'results/mcq_utility'
    os.makedirs(result_dir,exist_ok=True)
    result_path = os.path.join(result_dir,f'{os.path.basename(model_path)}.json')
    with open(result_path,'w') as f:
        json.dump(result_store,f)

if __name__ == "__main__":
    main()
    
    
    
    
    
    

