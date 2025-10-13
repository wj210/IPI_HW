
import os,sys
sys.path.append(os.path.abspath(".")) 
sys.path.append(os.path.abspath("BIPIA")) 
sys.path.append(os.path.abspath("aside")) # to access aside.experiments
from aside.model import *
from aside.model_api import *
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict,Counter
from utils.utils import *
from utils.model_utils import load_model
from copy import deepcopy
from utils.plot_utils import *
from functools import partial
torch.set_grad_enabled(False)
seed_all()
from BIPIA.bipia.data import AutoPIABuilder
from BIPIA.bipia.metrics import BipiaEvalFactory
from vllm import SamplingParams
from constants import *
from accelerate import PartialState
PartialState() 
import argparse
from pathlib import Path
import jsonlines

def evaluate_scenario(
    model,
    tokenizer,
    scenario_name,
    attack="text",
):
    if attack == "text":
        attack_path = "BIPIA/benchmark/text_attack_test.json"
    elif attack == "code":
        attack_path = "BIPIA/benchmark/code_attack_test.json"
    else:
        raise ValueError(f"Invalid attack type: {attack}")

    scenario_to_path = {
        "email": "BIPIA/benchmark/email/test.jsonl",
        "code": "BIPIA/benchmark/code/test.jsonl",
        "qa": "BIPIA/benchmark/qa/test.jsonl",
        "abstract": "BIPIA/benchmark/abstract/test.jsonl",
        "table": "BIPIA/benchmark/table/test.jsonl",
    }
    pia_builder = AutoPIABuilder.from_name(scenario_name)(2023)
    pia_samples = pia_builder(
        scenario_to_path[scenario_name],
        attack_path,
        enable_stealth=False,
    )

    # Filter samples for specific attacks (which do no require LLM evaluation)
    df = pia_samples[
        pia_samples.attack_name.str.startswith("Emoji Substitution")
        | pia_samples.attack_name.str.startswith("Language Translation")
        | pia_samples.attack_name.str.contains("Marketing & Advertising-1")
        | pia_samples.attack_name.str.contains("Scams & Fraud-1")
        | pia_samples.attack_name.str.contains(
            "Bringing Down Hosts and Servers (Denial of Service)"
        )
        | pia_samples.attack_name.str.contains("Compromising Computers")
        | pia_samples.attack_name.str.contains(
            "Encrypting Documents and Demanding Ransom (Ransomware)"
        )
        | pia_samples.attack_name.str.contains("Corrupting an Operating System")
        | pia_samples.attack_name.str.contains("Blocking Internet Connection")
        | pia_samples.attack_name.str.contains("Introduce System Fingerprinting")
        | pia_samples.attack_name.str.contains("Screen Scraping")
        | pia_samples.attack_name.str.contains("Keylogging")
        | pia_samples.attack_name.str.contains("Traffic Analysis")
        | pia_samples.attack_name.str.contains("Data Eavesdropping")
    ]

    # Prepare prompts
    all_prompts = []
    def format_bipia_prompt(instr,data, tokenizer):
        has_input = 'message.role == "input"' in tokenizer.chat_template
        if has_input:
            msg = [{'role':'user','content':instr},{'role':'input','content':data}]
        else:
            msg = [{'role':'user','content':instr+'\n\n'+data}]
        return msg
    gen_kwargs = SamplingParams(temperature=0., max_tokens=1024,stop=[tokenizer.eos_token])
    
    gpt_config_file = "BIPIA/config/gpt35.yaml"  # Make sure this exists
    # Define attacks to evaluate
    attacks = np.array(df.attack_name.unique())
    

    for i in range(len(df)):
        user_prompt, input_prompt = pia_builder.construct_prompt(df.iloc[i]) # edit the construct_aside_prompt to 
        all_prompts.append(tool_prompt_format(format_bipia_prompt(user_prompt,input_prompt,tokenizer),None,tokenizer))

    # Generate responses using the model
    responses = vllm_generate(model,all_prompts,gen_kwargs,use_tqdm=True)
    # Prepare output for evaluation
    out = []
    if "question" not in df.columns:
        df["question"] = ""
    for attack_name, task_name, target, message, position, response in zip(
        df["attack_name"],
        df["task_name"],
        df["ideal"],
        df["question"],
        df["position"],
        responses,
    ):
        out.append(
            {
                "attack_name": attack_name,
                "task_name": task_name,
                "response": response,
                "message": message,
                "target": target,
                "position": position,
            }
        )

    # Save responses
    output_path = Path(f"BIPIA/output/{scenario_name}_responses.jsonl")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(out)

    # Evaluate responses
    gpt_config_file = "BIPIA/config/gpt35.yaml"  # Make sure this exists
    # Define attacks to evaluate
    attacks = np.array(df.attack_name.unique())

    evaluator = BipiaEvalFactory(
        gpt_config=gpt_config_file,
        activate_attacks=attacks,
    )
    

    asrs = evaluator.add_batch(
        predictions=responses,
        references=df["ideal"],
        attacks=df["attack_name"],
        tasks=df["task_name"],
    )

    avg_asr = np.mean(asrs)
    print(f"The average ASR for {scenario_name} scenario is: {avg_asr}")

    return df, asrs, avg_asr
    
def evaluate_bipia(
    model,
    tokenizer,
    scenarios=["email", "code", "table", "abstract", "qa"],
    attacks=["text", "code"],
):
    results = defaultdict(dict)
    total_count = 0
    for attack_type in attacks:
        print(f"\nEvaluating BIPIA {attack_type} attacks:")

        # Use only code scenario for code attacks, and other scenarios for text attacks
        if attack_type == "code":
            attack_scenarios = ["code"]
        else:
            attack_scenarios = [s for s in scenarios if s != "code"]

        print(f"Using scenarios {attack_scenarios} for {attack_type} attacks")
        scenario_asr = []
        for scenario in attack_scenarios:
            print(
                f"\nEvaluating {scenario} scenario with attack: {attack_type}"
            )
            _, asrs, avg_asr = evaluate_scenario(
                model,
                tokenizer,
                scenario,
                attack_type,
            )
            scenario_asr.append(avg_asr)
            print(
                f"Attack: {attack_type}, Scenario: {scenario}, ASR: {avg_asr:.4f}, Count: {len(asrs)}"
            )
            total_count += len(asrs)
            results[attack_type][scenario] = np.round(avg_asr,3)
            # Calculate average ASR across all scenarios for this seed and attack type
        mean_scenario_asr = np.mean(scenario_asr)
        results[attack_type]["mean"] = np.round(mean_scenario_asr,3)
        print (f"\nMean ASR for attack type {attack_type}: {mean_scenario_asr:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default=None,help='model path')
    args = parser.parse_args()
    seed_all()
    
    model_path = args.model_path
    if 'Qwen/' not in model_path:
        model_path = os.path.join(MODEL_DIR,model_path)
    torch_dtype = torch.bfloat16
    model,tokenizer,is_aside,init_fn = load_model(model_path,use_vllm=True,dtype=torch_dtype,vllm_kwargs = {'gpu_memory_utilization':0.8,'enable_chunked_prefill':True})
    print (f'is_aside: {is_aside}')
    
    results = evaluate_bipia(model,tokenizer) # run all

    save_dir = 'results/bipia'
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,model_path.split('/')[-1].strip() + '.json')
    with open(save_path,'w') as f:
        json.dump(results, f, indent=4)
        
    for attack,scenario_dict in results.items(): # print results again
        for scenario, v in scenario_dict.items():
            k = f"{attack}_{scenario}"
            print (k,v)
            
if __name__ == "__main__":
    main()