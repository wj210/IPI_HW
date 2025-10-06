import os
import json
import torch
import sys
sys.path.append(os.path.abspath("..")) 
from utils.utils import aside_encode,get_aside_segments,aside_encode_start,multiturn_aside_encode

from typing import Any, Dict, List

from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling,DataCollatorWithPadding, Trainer
from transformers import TrainerCallback, TrainerState, TrainerControl

import math
from transformers.trainer_utils import is_main_process


import argparse
import logging

from trl import SFTTrainer,DPOTrainer,DPOConfig
from model import *
from typing import List, Dict
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from model_api import *
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from aside_dpo import DPOTrainerWithSegments,apply_chat_tool_template,apply_standard_chat_template
from data_utils import *

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from embeddings_init import generate_isoclinic_rotation_matrix
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def merge_zero_shards_and_save(checkpoint_dir):
    """
    Merges DeepSpeed ZeRO-sharded checkpoints into a single FP32 state dict.
    
    This function converts distributed DeepSpeed checkpoints (which are sharded across
    multiple GPUs) into a single consolidated checkpoint file that can be loaded on
    any device. Essential for saving final model checkpoints after distributed training.
    
    Args:
        checkpoint_dir (str): Directory containing the ZeRO-sharded checkpoint files.
                             Typically contains files like zero_pp_rank_*_mp_rank_00_model_states.pt
        
    """
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=checkpoint_dir,
        output_dir=checkpoint_dir,
    )

    print(f"Saved merged FP32 checkpoint to {checkpoint_dir}")
    
class CustomDataCollator:
    """
    Pads segment_ids too
    """
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, features):
        padded_segment_ids, padded_labels = None, None
        if "segment_ids" in features[0] and features[0]["segment_ids"] is not None:
            # Collect segment_ids from each feature and convert them to tensors.
            segment_ids = [torch.tensor(feature.pop("segment_ids")) for feature in features]
            padded_segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        
        if 'labels' in features[0] and features[0]['labels'] is not None:
            labels = [torch.tensor(feature.pop("labels")) for feature in features]
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        batch = self.base_collator(features)
        if padded_segment_ids is not None:
            batch["segment_ids"] = padded_segment_ids
        if padded_labels is not None:
            batch["labels"] = padded_labels
        
        return batch 
    
def main(config_path: str, emb_type: str,hparams: Dict[str, Any]):
    config = load_config(config_path)
    instruct_model_path = data_model_path = tokenizer_path = config["model_path"]
    checkpoint_to_load_from = None

    train_dataset_path = config["train_dataset_path"]
    assert "eval_dataset_path" in config.keys(), "Update Config to new format"
    eval_dataset_path = config["eval_dataset_path"]
    output_dir = args.output_dir
    
    if 'emb_type' == 'forward_rot':
        architecture_name = 'ASIDE'
    else:
        architecture_name = 'Vanilla'
    
    if len(hparams['extra_names']):
        hparams['extra_names'] = '_' + hparams['extra_names']
    ## Output path name
    output_dir = os.path.join(output_dir,f"Qwen3-8B_{hparams['mode']}_{architecture_name}{hparams['extra_names']}")
    
    ## Logging
    # Include hyperparams in run name
    run_name = f"{instruct_model_path.split('/')[0].strip()}_train_alpha={hparams['rotation_alpha']}"
    train_logs_path = os.path.join(config["train_logs_path"], run_name)
    os.makedirs(train_logs_path, exist_ok=True)
    log_file = os.path.join(train_logs_path, "losses.log")  # Specify file path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Save to file
            logging.StreamHandler()  # Print to console
        ]
    )

    # Load dataset
    all_train_data,all_eval_data = [],[]
    assert len(train_dataset_path) == len(eval_dataset_path), "Train and Eval dataset paths must have the same length"
    assert len(train_dataset_path[0]) == 2 and len(eval_dataset_path[0]) == 2, "should include both the path and whether it is input or tool data" # input is alpaca and tool is regular tool agentic use
    for train_info,eval_info in zip(train_dataset_path,eval_dataset_path): # train type is either 'input' or 'tool'
        train_path, train_type = train_info
        eval_path, eval_type = eval_info
        train_data = json.load(open(train_path))
        eval_data = json.load(open(eval_path))
        if hparams['num_data'] > 0:
            if isinstance(train_data, HFDataset):
                train_data = train_data.select(range(hparams['num_data']))
            else:
                random_ids = torch.randperm(len(train_data))[:hparams['num_data']]
                train_data = [train_data[i] for i in random_ids]
            print (f'Using only {len(train_data)} data samples for training')
        
        # only need to check for input
        if train_type == 'input' and not check_input_format(train_data[0]): # convert to suitable format
            train_data = input_formatting_fn(train_data)
        if eval_type == 'input' and not check_input_format(eval_data[0]):
            eval_data = input_formatting_fn(eval_data)
            
        all_train_data.extend(train_data)
        all_eval_data.extend(eval_data)
    train_data = all_train_data
    eval_data = all_eval_data
    print (f'Number of training samples: {len(train_data)}')
    print (f'Number of evaluation samples: {len(eval_data)}')
        

    init_handler_count = 1 if hparams['mode'] == 'sft' else 2
    handlers = []
    for i in range(init_handler_count):
        handlers.append(CustomModelHandler(
            checkpoint_to_load_from, 
            instruct_model_path, 
            data_model_path, 
            tokenizer_path, 
            None, 
            embedding_type=emb_type,
            embeddings_init=hparams["embedding_init"],
            rotation_alpha=hparams["rotation_alpha"],
            add_linear_shift=hparams["add_linear_shift"],
            rotation_direction=hparams["rotation_direction"], 
            learned_rotation=hparams["learned_rotation"],
            gradual_rotation=hparams["gradual_rotation"],
            model_dtype=torch.bfloat16,
            load_from_checkpoint=checkpoint_to_load_from is not None,
            post_init_rotation=hparams["post_init_rotation"],
        ))
        if hparams['mode'] == 'sft':
            handlers[i].tokenizer.padding_side = "right"
        if handlers[i].tokenizer.pad_token is None:
            print("WARNING: pad_token is None")
    handler = handlers[0]
    if len(handlers) > 1: # this is just for DPO
        ref_model = handlers[1].model

    # add additional tokens to delimiter input data field
    handler.tokenizer.chat_template = open('./data/chat_template.jinja','r').read()
    ## add the special tokens
    handler.tokenizer.add_tokens(['<reference_data>','</reference_data>'])
    handler.model.resize_token_embeddings(len(handler.tokenizer))
    
    ## Set up train dataset
    token_args = {
        'input_token': config.get('input_token',None),
        'assistant_token': config.get('assistant_token',None),
        'user_token': config.get('user_token',None),
        'tool_token': config.get('tool_token',None),
    }
    

    #########
    ## sft ##
    #########
    if hparams['mode'] == 'sft':
        train_dataset = Tool_Input_Dataset(train_data, handler.tokenizer, handler.embedding_type,token_args,completion_only=hparams.get('completion_only',True))
        eval_dataset = Tool_Input_Dataset(eval_data, handler.tokenizer, handler.embedding_type,token_args,completion_only=hparams.get('completion_only',True))
    #########
    ## DPO ##
    #########
    else: 
        def convert_str_to_list(examples): # if chosen and rejected are str, while prompt is list of dict alr
            for e in examples:
                e['chosen'] = [{'role':'assistant','content':e['chosen']}]
                e['rejected'] = [{'role':'assistant','content':e['rejected']}]
            return examples
        
        def convert_metasecalign(examples):
            new_examples = []
            for e in examples:
                new_examples.append(
                    {
                        'prompt': [{'role':'user','content':e['instruction']},
                                   {'role':'input','content':e['input']}], # need to change the chat template 
                        'chosen': [{'role':'assistant','content':e['chosen']}],
                        'rejected': [{'role':'assistant','content':e['rejected']}],
                    }
                )
            return new_examples
        
        if 'metasecalign' not in train_dataset_path.lower():
            train_data = convert_str_to_list(train_data)
            train_data = [apply_chat_tool_template(x,tokenizer=handler.tokenizer) for x in train_data]
            eval_data = convert_str_to_list(eval_data)
            eval_data = [apply_chat_tool_template(x,tokenizer=handler.tokenizer) for x in eval_data]
        else: # use the default apply chat template
            ref_model.resize_token_embeddings(len(handler.tokenizer)) # set the ref model as well
            train_data = convert_metasecalign(train_data)
            train_data = [apply_standard_chat_template(x,tokenizer=handler.tokenizer) for x in train_data]
            eval_data = convert_metasecalign(eval_data)
            eval_data = [apply_standard_chat_template(x,tokenizer=handler.tokenizer) for x in eval_data]
            print (f'Example prompt: {train_data[0]["prompt"]}')
            print (f'Example chosen: {train_data[0]["chosen"]}')
            print (f'Example rejected: {train_data[0]["rejected"]}')
            
        train_dataset = HFDataset.from_list(train_data)
        eval_dataset = HFDataset.from_list(eval_data)

    if hparams['mode'] == 'dpo':
        data_collator = None
    else:
        data_collator = CustomDataCollator(
            tokenizer=handler.tokenizer,
            mlm=False,
        )

    total_steps = math.ceil((len(train_dataset) * hparams["num_train_epochs"]) / (hparams["per_device_train_batch_size"] * hparams["gradient_accumulation_steps"]  * (dist.get_world_size() if dist.is_initialized() else 1)))
    print (f"Total training steps: {total_steps}")
    if hparams['eval_percent']> 0:
        hparams['evaluation_strategy'] = 'steps'
        hparams['eval_steps'] = int(total_steps * hparams['eval_percent'])
        hparams['save_strategy'] = 'steps'
        hparams['save_steps'] = hparams['eval_steps']
        

    print('Datasets created')
    training_args = {
        'output_dir': output_dir,
        'num_train_epochs': hparams["num_train_epochs"],
        'per_device_train_batch_size': hparams["per_device_train_batch_size"],
        'per_device_eval_batch_size': hparams["per_device_eval_batch_size"],
        'gradient_accumulation_steps': hparams["gradient_accumulation_steps"],
        'learning_rate': hparams["learning_rate"],
        'weight_decay': hparams["weight_decay"],
        'lr_scheduler_type': hparams["lr_scheduler_type"],
        'warmup_ratio': hparams["warmup_ratio"],
        'logging_dir': train_logs_path,
        'logging_steps': hparams["logging_steps"],
        'log_level': "info",
        'eval_strategy': hparams["evaluation_strategy"],
        'save_strategy': hparams["save_strategy"],
        'eval_steps': hparams["eval_steps"],
        'save_steps': hparams["save_steps"],
        'save_total_limit': hparams["save_total_limit"],
        'load_best_model_at_end': hparams["load_best_model_at_end"],
        'prediction_loss_only': hparams["prediction_loss_only"] if hparams['mode'] == 'sft' else False,
        'bf16': hparams["bf16"],
        'remove_unused_columns': hparams["remove_unused_columns"],
        'deepspeed': "deepspeed_config.json" if not hparams['no_deepspeed'] else None,
        'report_to': hparams["report_to"],
        'metric_for_best_model': None,
        'gradient_checkpointing': True,
    }
    if hparams['mode'] == 'dpo':
        training_args['max_length'] = hparams['max_length']
        training_args['beta'] = hparams['beta']
        training_args['dataset_num_proc'] = 32
        training_args = DPOConfig(**training_args)
        handler.model.config.use_cache = False # important when gradient checkpointing is used
        ref_model.config.use_cache = False
    else:
        training_args = TrainingArguments(**training_args)
 
    handler.model.config.use_cache = False
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    total_steps_per_epoch = math.ceil(
    len(train_dataset) / (world_size * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
)
    print(f"Total steps per epoch: {total_steps_per_epoch}")

    if hparams['mode'] == 'sft':
        trainer = SFTTrainer(
            model=handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=handler.tokenizer,
            data_collator=data_collator,
        )
    else:
        dpo_args = {
            'model': handler.model,
            'ref_model': ref_model,
            'args': training_args,
            'train_dataset': train_dataset,
            'eval_dataset': eval_dataset,
            'processing_class': handler.tokenizer,
            'data_collator': data_collator,
        }
        if handler.embedding_type in ['ise','forward_rot']: #TODO need to edit to enable tool or input tokens
            dpo_trainer_class = DPOTrainerWithSegments # or can just change to from a token onwards
            assistant_token,tool_token = config['assistant_token'],config['tool_token']
            dpo_args['start_tokens'] = [assistant_token[0],tool_token[0]]
            dpo_args['end_tokens'] = [assistant_token[1],tool_token[1]]
            if 'metasecalign' in train_dataset_path.lower(): # simialr to alpaca, only take from data token onwards (include assistant)
                dpo_args['start_from'] = tool_token[0]
        else:
            dpo_trainer_class = DPOTrainer # use the one with segment ids
        trainer = dpo_trainer_class(**dpo_args)
    print('Trainer created')
    # Start training
    trainer.train()
    # Save the trained model and tokenizer
    print("Custom impl., saving last checkpoint")
    trainer.save_model(output_dir)
    # if args.no_deepspeed:
    #     trainer.tokenizer.save_pretrained(output_dir)
    # else:
    merge_zero_shards_and_save( 
        checkpoint_dir=output_dir
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with optional hyperparameters.")
    parser.add_argument("--emb_type", type=str, choices=["double_emb", "single_emb",  "ise", "forward_rot"], help="Embedding Type")
    parser.add_argument("--config_path", type=str, default="", help="Path to the config file.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints and logs.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="device eval batch size")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--max_length", type=int, default=768, help="Max length of dataset")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Steps between each logging.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["steps", "epoch"],
                        help="Evaluation strategy.")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"],
                        help="Save strategy.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluations.")
    parser.add_argument("--eval_percent", type=float, default=0.1, help="Percentage of data to use for evaluation.")
    parser.add_argument("--save_steps", type=int, default=300, help="Number of steps between model checkpoints.")
    parser.add_argument("--num_data", type=int, default=-1, help="Number of data points to use.")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of checkpoints to save.")
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=False,
                        help="Whether to load the best model at the end of training.")
    parser.add_argument("--prediction_loss_only", type=str2bool, default=False,
                        help="If True, only the prediction loss is used.")
    parser.add_argument("--bf16", type=bool, default=True, help="Use bf16 precision if available.")
    parser.add_argument("--activation_checkpointing", type=bool, default=False,
                        help="Whether to use gradient checkpointing.")
    parser.add_argument("--remove_unused_columns", type=bool, default=False, help="Remove unused columns in dataset.")
    parser.add_argument("--report_to", type=str, default='none',
                        help="Reporting framework (e.g., wandb, tensorboard).")
    parser.add_argument("--embedding_init", type=str, default=None, choices=[None, "copy", "rot_ind", "rot_isoclinic"],
                        help="Embedding initialization.")
    parser.add_argument("--rotation_alpha", type=float, default=None,
                        help="Embedding rotation constant.")
    parser.add_argument("--add_linear_shift", type=str2bool, default=False,
                    help="Add linear shift before rotation.")
    parser.add_argument("--learned_rotation", type=str2bool, default=None,
                    help="If rotation is parameter.")
    parser.add_argument("--gradual_rotation", type=str2bool, default=None,
                    help="If rotation is gradual.")
    parser.add_argument("--rotation_direction", type=str, default="right",
                    help="Embedding rotation direction.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training on GPUs.")

    parser.add_argument("--post_init_rotation", type=str2bool, default=False, help="Rotate embedding after initialization (normally used when loading from checkpoint).")

    parser.add_argument("--gradual_rot", type=str2bool, default=False, help="Use gradual rotation every step of training during 1st epoch")
    parser.add_argument("--no_deepspeed", default = False,type = bool)
    parser.add_argument("--mode", default = 'sft',type = str, help="sft or dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta value")
    parser.add_argument("--completion_only", action = 'store_true', help="If true, only compute loss on the assistant part of the output")
    
    # Parse the arguments
    args = parser.parse_args()

    # Prepare user_hparams
    user_hparams = vars(args)  # Convert parsed arguments into dictionary
    emb_type = user_hparams.pop("emb_type")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    print (f'Mode: {args.mode}')
    config_path = user_hparams['config_path']
    assert config_path != "", "Please provide a config path using --config_path"
    print (f'Using config path: {config_path}')
    main(config_path, emb_type,user_hparams)
