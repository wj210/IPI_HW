import os,sys
sys.path.append(os.path.abspath(".")) 
sys.path.append(os.path.abspath("InjecAgent")) 
sys.path.append(os.path.abspath("aside"))
from aside.model import *
from aside.model_api import *
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict,Counter
from utils.utils import *
from copy import deepcopy
from utils.plot_utils import *
from utils.model_utils import load_model
torch.set_grad_enabled(False)
import pickle
from functools import partial
home_dir = "/export/home2/weijie210/StruQ"
cache_dir = f'{home_dir}/cache'
os.makedirs(cache_dir,exist_ok=True)
seed_all()
from vllm import LLM, SamplingParams

from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import *
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
import argparse


def run_task(agent_pipeline,suite,user_task,injection_task=None,task_injection = {},return_messages=False):
    environment = suite.load_and_inject_default_environment(task_injection)
    if isinstance(user_task, BaseUserTask):
        task_environment = user_task.init_environment(environment)
    else:
        task_environment = environment
    pre_environment = task_environment.model_copy(deep=True)
    if isinstance(user_task, BaseUserTask):
        prompt = user_task.PROMPT
    else:
        prompt = user_task.GOAL

    messages = []
    runtime = FunctionsRuntime(suite.tools)

    _, _, task_environment, messages, _ = agent_pipeline.query(prompt, runtime, task_environment)
    model_output = model_output_from_messages(messages)
    functions_stack_trace = functions_stack_trace_from_messages(messages)
    utility = suite._check_task_result(
            user_task,
            model_output or [],
            pre_environment,
            task_environment,  # type: ignore
            functions_stack_trace,
        )
    security = None
    if injection_task is not None:
        security = suite._check_task_result(
            injection_task,
            model_output or [],
            pre_environment,
            task_environment,  # type: ignore
            functions_stack_trace,
        )
    if not return_messages:
        return utility,security
    else:
        return utility,security, messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default="Qwen3-8B-ToolLLM",help='model path')
    parser.add_argument('--base_model_path',type=str,default="Qwen/Qwen3-8B",help='base model path')
    parser.add_argument('--thinking',action='store_true',help='enable chain-of-thought')
    parser.add_argument('--use_vllm',action='store_true',help='Uses VLLM')
    parser.add_argument('--suites', type =str,nargs = '+', default = ['slack','banking','travel','workspace'], help = 'suites to run')
    
    args = parser.parse_args()

    is_api = any([s in args.model_path.lower() for s in ['gpt','deepseek','claude']])
    if not is_api:
        data_dir = '/dataset/common/huggingface/model'
        torch_dtype = torch.bfloat16
        model_path = args.model_path
        if 'tool' in model_path.lower():
            model_path = os.path.join(data_dir, model_path)
        model,tokenizer,is_aside = load_model(model_path,use_vllm=args.use_vllm,dtype=torch_dtype,vllm_kwargs = {'gpu_memory_utilization':0.8,'enable_chunked_prefill':False})
        
        ## no prompt-based defense
        defense = None
        additional_encode_fn = partial(aside_encode,tokenizer=tokenizer) if is_aside else lambda x:x

        agent_pipeline = AgentPipeline.from_config(
            PipelineConfig(
                llm='local-tool',
                defense=defense,
                system_message_name=None,
                system_message=None,
                tool_output_format=None,
                extra_args={
                    'model':model,
                    'tokenizer':tokenizer,
                    'encode_fn': partial(tool_prompt_format,tokenizer=tokenizer,enable_thinking = args.thinking),
                    'additional_encode_fn': additional_encode_fn,
                    'gen_kwargs': {'max_new_tokens':512,'do_sample':False,'eos_token_id':[tokenizer.eos_token_id]} if not args.use_vllm else SamplingParams(max_tokens=512,temperature=0,stop_token_ids = [tokenizer.eos_token_id]),
                    'is_vllm': args.use_vllm,
                },
            )
        )
    else:
        defense = None
        with open(f'/export/home2/weijie210/ipi_huawei/ds_key.txt','r') as f:
            ds_key = f.read().strip()
        os.environ['OPENAI_API_KEY'] = ds_key
        agent_pipeline = AgentPipeline.from_config(
                PipelineConfig(
                    llm=model_path,
                    defense=defense,
                    system_message_name=None,
                    system_message=None,
                    tool_output_format=None,
                    extra_args ={'max_tokens':1024}
                    
                )
            )
    
    benchmark_version = 'v1.2.1'
    attack_name = 'important_instructions'

    if 'qwen' in model_path.lower():
        base_model_name = 'qwen3'
    elif 'llama3' in model_path.lower():
        base_model_name = 'llama3'
    else:
        base_model_name = model_path


    model_n = model_path.split('/')[-1]
    result_dict = 'results/agentdojo'
    os.makedirs(result_dict,exist_ok=True)

    result_file = f'{result_dict}/{base_model_name}_{model_n}.json'
    msg_file = f'{result_dict}/{base_model_name}_{model_n}_messages.pkl'
    
    if args.thinking:
        result_file = result_file.replace('.json','_cot.json')
        msg_file = msg_file.replace('.pkl','_cot.pkl')
    
    debug_messages = defaultdict(list)
    if os.path.exists(result_file):
        all_suite_results = json.load(open(result_file,'r'))
        existing_suites = list(all_suite_results.keys())
        args.suites = [s for s in args.suites if s not in existing_suites]  
        ## turn   all_suite_results into defaultdict(dict)
        all_suite_results = defaultdict(dict, all_suite_results)
    else:
        all_suite_results = defaultdict(dict)
        
    
    if os.path.exists(msg_file):
        all_suite_asr_messages = pickle.load(open(msg_file,'rb'))
    else:
        all_suite_asr_messages = defaultdict(dict)

    print (f'REMAINING SUITES: {args.suites}')
    if len(args.suites):
        for suite_name in args.suites:
            suite = get_suite(benchmark_version, suite_name)
            print (suite.injection_tasks)
            print(f"**Running benchmark for suite: '{suite.name}'**")
            user_tasks_to_run = suite.user_tasks.values()
            print (f"Num User tasks to run: {len(user_tasks_to_run)}")
            attacker_ = load_attack(attack_name, suite, target_pipeline = agent_pipeline)
            print ('is dos_attack:', attacker_.is_dos_attack)
            injection_tasks_to_run = suite.injection_tasks
            print (f"Num Injection tasks to run: {len(injection_tasks_to_run)}")

            #Injection utility
            injection_task_utility = {}
            for injection_task_id, injection_task in tqdm(injection_tasks_to_run.items(),total = len(injection_tasks_to_run),desc = f'Running Injection utility for {suite_name}'):
                utility,_,message = run_task(agent_pipeline,suite,injection_task,return_messages=True)
                injection_task_utility[injection_task_id] = utility
                debug_messages['injection'].append(message)
                

            print (f'{suite_name} Injection utility: {np.mean(list(injection_task_utility.values())):.2f}')
            all_suite_results[suite_name]['injection_utility'] = np.mean(list(injection_task_utility.values()))

        #     # ## Non-injection utility
            task_utility = {}
            for user_task in tqdm(user_tasks_to_run,total = len(user_tasks_to_run),desc = f'Running Task utility for {suite_name}'):
                utility,_,message = run_task(agent_pipeline,suite,user_task,return_messages=True)
                task_utility[user_task.ID] = utility
                debug_messages['utility'].append(message)
                
                
            print (f'{suite_name} Task utility: {np.mean(list(task_utility.values())):.2f}')
            all_suite_results[suite_name]['task_utility'] = np.mean(list(task_utility.values()))
            
            ## With attack
            utility_with_attack = {}
            all_messages = {}
            suite_security_results =  {}
            for user_task in tqdm(user_tasks_to_run,total = len(user_tasks_to_run),desc = f'Running Attack for {suite_name}'):
                for injection_task_id in injection_tasks_to_run:
                    injection_task = suite.get_injection_task_by_id(injection_task_id)
                    task_injections = attacker_.attack(user_task, injection_task)
                    utility,security,messages = run_task(agent_pipeline,suite,user_task,injection_task,task_injections,return_messages=True)
                    utility_with_attack[(user_task.ID, injection_task_id)] = utility
                    suite_security_results[(user_task.ID, injection_task_id)] = security
                    all_messages[(user_task.ID, injection_task_id)] = messages
            all_suite_asr_messages[suite_name] = all_messages
            all_suite_results[suite_name]['asr'] = np.mean(list(suite_security_results.values()))
            all_suite_results[suite_name]['attack_utility'] = np.mean(list(utility_with_attack.values()))

            print (f'{suite_name} Task with injection Utility: {np.mean(list(utility_with_attack.values())):.2f}')
            print (f'{suite_name} ASR: {np.mean(list(suite_security_results.values())):.2f}')
            
        with open(result_file, 'w') as f:
            json.dump(all_suite_results, f, indent=4)
            
    for suite_name in all_suite_results:
        suite_results = all_suite_results[suite_name]
        print (f"Results for {suite_name}:")
        print (f"Injection utility: {suite_results['injection_utility']:.2f}")
        print (f"Task utility: {suite_results['task_utility']:.2f}")
        print (f"Task with injection utility: {suite_results['attack_utility']:.2f}")
        print (f"ASR: {suite_results['asr']:.2f}")
    

if __name__ == "__main__":
    main()
    