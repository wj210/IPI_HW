"""
ASIDE Fine-tuning Script

This is the main training script for reproducing the ASIDE (Architecturally Separated 
Instruction-Data Embeddings) experiments. It implements the supervised fine-tuning (SFT) 
procedure described in Section 4.1 of the paper using the Alpaca dataset.

Key Features:
- Trains models with ASIDE architectural modifications (rotation-based instruction/data separation)
- Supports multiple embedding types: single_emb , double_emb, ISE, forward_rot (=ASIDE)
- Implements gradual rotation during training (experimental feature)
- Distributed training support with DeepSpeed integration

Usage:
     srun --export=ALL deepspeed --num_gpus=8 --master_port=29509 \
        fine-tune.py --model_family llama_2_13b --train_version SFTv110 \
        --emb_type forward_rot --model_ix 1 --run_number 0 \
        --train_type full --num_train_epochs 3 --per_device_train_batch_size 2 \  
        --gradient_accumulation_steps 4 --learning_rate 1e-6 --lr_scheduler_type cosine \ 
        --warmup_ratio 0 --logging_steps 10 --evaluation_strategy epoch \ 
        --save_strategy epoch --eval_steps 1 --save_steps 1 --save_total_limit 1 \ 
        --load_best_model_at_end True --prediction_loss_only True --bf16 True \ 
        --embedding_init rot_isoclinic --rotation_alpha 1.57079633 \ 
        --learned_rotation False --add_linear_shift False --rotation_direction right \
        --gradual_rotation False


References:
    ASIDE: Architectural Separation of Instructions and Data in Language Models
    Section 4.1: Training procedure
"""

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
    Custom data collator for ASIDE models with segment ID support.
    
    This collator extends the standard language modeling collator to handle
    segment IDs that distinguish instruction from data tokens. Essential for
    ASIDE's conditional embedding mechanism.
    
    Args:
        tokenizer: HuggingFace tokenizer
        mlm (bool): Whether to use masked language modeling (typically False for ASIDE)
        
    Note:
        Segment IDs are crucial for ASIDE as they determine which tokens receive
        rotated embeddings during the forward pass.
    """
    def __init__(self, tokenizer, mlm=False,pad=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        # if pad:
        #     self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # else:
        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, features):
        """
        Collates batch with segment ID handling.
        
        Returns:
            Dict: Batch dictionary with 'input_ids', 'attention_mask', 'labels',
                  and 'segment_ids' (for ASIDE and ISE models)
        """
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


class EmbeddingRotationCallback(TrainerCallback):
    """
    Callback for gradual rotation during ASIDE training (experimental feature).
    
    This callback implements gradual rotation where the rotation angle increases
    linearly from 0 to the target angle during the first epoch. This experimental
    approach allows the model to gradually adapt to the architectural changes.
    
    Args:
        total_steps_per_epoch (int): Total training steps in one epoch
        
    Note:
        This is an experimental feature mentioned in the paper as future work.
        The standard ASIDE method applies full rotation (π/2) from the beginning.
        
    Mathematical Implementation:
        rotation_alpha(step) = target_alpha * (step / total_steps_per_epoch)
        where step ∈ [0, total_steps_per_epoch]
    """
    def __init__(self, total_steps_per_epoch: int):
        self.total_steps_per_epoch = total_steps_per_epoch

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        # Determine the current step within the epoch.
        if state.global_step <= self.total_steps_per_epoch:
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            dim = model.config.hidden_size
            # Calculate the rotation alpha based on the current step.
            rotation_alpha = model.global_rotation_alpha * (state.global_step / self.total_steps_per_epoch)
            
            with torch.no_grad():
                if dist.get_rank() == 0:
                    # create the new matrix
                    new_matrix = generate_isoclinic_rotation_matrix(dim, rotation_alpha, device, model_dtype)
                else:
                    # create a placeholder for it
                    new_matrix = torch.empty_like(model.rotation_matrix)

                dist.broadcast(new_matrix, src=0)
                model.rotation_matrix.data.copy_(new_matrix)
            
            # Broadcast the computed value (from rank 0) to all processes.

            
            if dist.get_rank() == 0:
                print(f"On step {state.global_step} set embedding rotation angle to {rotation_alpha}")
            
        return control



def main(model_family: str, emb_type: str, train_version: str, model_ix: int, run_number: int, hparams: Dict[str, Any]):
    """
    Main training function for ASIDE experiments.
    
    1. Loads configuration and datasets
    2. Initializes ASIDE model with appropriate embedding type
    3. Sets up custom trainer with logging
    4. Executes supervised fine-tuning on Alpaca data
    5. Saves final checkpoint with metadata
    
    Args:
        model_family (str): Model family identifier (e.g., "llama_3.1_8b")
        emb_type (str): Embedding type - determines ASIDE variant:
                       - "single_emb": Vanilla model
                       - "double_emb": Legacy double embedding approach
                       - "ise": ISE baseline method
                       - "forward_rot": Main ASIDE method with isoclinic rotation
        train_version (str): Training version identifier (e.g., "SFTv110")
        model_ix (int): Index of model in configuration's pure_models list
        run_number (int): Run number for experiment tracking
        hparams (Dict[str, Any]): Training hyperparameters
        
    """
    config_path = f"./configs/config_{model_family}_{train_version}.json"
    config = load_config(config_path)

    if dist.get_rank() == 0:
        print(config)
        print("\n", config["models"][emb_type]["pure_models"])
    

    pure_model_info = config["models"][emb_type]["pure_models"][model_ix]
    checkpoint_to_load_from = pure_model_info["checkpoint_to_load_from"]
    tokenizer_path = config["tokenizer_path"]
    instruct_model_path = pure_model_info.get("instruct_model_path", None)
    data_model_path = pure_model_info.get("data_model_path", None)
    chat_template_path = pure_model_info["chat_template_path"]

    train_dataset_path = config["train_dataset_path"]
    assert "eval_dataset_path" in config.keys(), "Update Config to new format"
    eval_dataset_path = config["eval_dataset_path"]
    train_version = config["training_version"]
    output_dir = os.path.join(args.output_dir,f'{model_family}_{run_number}_{train_version}')
    
    # Include hyperparams in run name
    run_name = f"{train_version}/{pure_model_info['name']}_train_{hparams['train_type']}_{train_version}_run={run_number}_alpha={hparams['rotation_alpha']}"

    train_logs_path = os.path.join(config["train_logs_path"], model_family, run_name)
    os.makedirs(train_logs_path, exist_ok=True)

    log_file = os.path.join(train_logs_path, "losses.log")  # Specify file path
    log_file_json = os.path.join(train_logs_path, "losses_metrics.json")  # Specify file path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Save to file
            logging.StreamHandler()  # Print to console
        ]
    )

    # Load dataset
    train_data = json.load(open(train_dataset_path))
    eval_data = json.load(open(eval_dataset_path))
    if hparams['num_data'] > 0:
        if isinstance(train_data, HFDataset):
            train_data = train_data.select(range(hparams['num_data']))
        else:
            random_ids = torch.randperm(len(train_data))[:hparams['num_data']]
            train_data = [train_data[i] for i in random_ids]
        print (f'Using only {len(train_data)} data samples for training')
    with open(config["prompt_templates_path"], "r") as f:
        templates = json.load(f)
    template_info = {
        "template_prompt": templates[0]
    }
    init_handler_count = 1 if hparams['mode'] == 'sft' else 2
    handlers = []
    for i in range(init_handler_count):
        handlers.append(CustomModelHandler(
            checkpoint_to_load_from, 
            instruct_model_path, 
            data_model_path, 
            tokenizer_path, 
            chat_template_path, 
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
            is_struq = 'StruQ' in train_version
        ))
        if hparams['mode'] == 'sft':
            handlers[i].tokenizer.padding_side = "right"
        if handlers[i].tokenizer.pad_token is None:
            print("WARNING: pad_token is None")
    handler = handlers[0]
    if len(handlers) > 1: # this is just for DPO
        ref_model = handlers[1].model
            
    max_length = hparams["max_length"]
    
    # set the prompt template to include input token first
    if 'metasecalign' in train_dataset_path.lower():
        handler.tokenizer.chat_template = open('./data/chat_template.jinja','r').read()
        ## add the special tokens
        handler.tokenizer.add_tokens(['<reference_data>','</reference_data>'])
        handler.model.resize_token_embeddings(len(handler.tokenizer))
    
    #########
    ## SFT ##
    #########
    
    ## NORMAL TOOL SFT
    if hparams['mode'] == 'sft':
        if 'tool' in train_dataset_path.lower(): # for tooluse, train on ToolACE
            train_dataset = ToolDataset(train_data, template_info, handler, assistant_token=config['assistant_token'], tool_token=config['tool_token'], user_token=config['user_token'], max_length=max_length, completion_only=hparams.get('completion_only',True))
            eval_dataset = ToolDataset(eval_data, template_info, handler, assistant_token=config['assistant_token'], tool_token=config['tool_token'], user_token=config['user_token'], max_length=max_length, completion_only=hparams.get('completion_only',True))
        
        ## METASECALIGN FOR ASIDE/ISE
        elif 'metasecalign' in train_dataset_path.lower() and handler.embedding_type in ['ise','forward_rot']:
            handler.tokenizer.chat_template = open('./data/chat_template.jinja','r').read() 
            train_dataset = Alpaca_Embedding_Dataset(train_data, handler.tokenizer, input_token=config['tool_token'][0], assistant_token=config['assistant_token'][0], max_length=max_length, completion_only=hparams.get('completion_only',True)) 
            eval_dataset = Alpaca_Embedding_Dataset(eval_data, handler.tokenizer, input_token=config['tool_token'][0], assistant_token=config['assistant_token'][0], max_length=max_length, completion_only=hparams.get('completion_only',True))
        
        ## THE ORIGINAL ASIDE CODE FOR ALPACA
        else:
            train_dataset = AlpacaDataset(train_data, template_info, handler, max_length=max_length,combine_system_user=config.get('combine_system_user',False),start_token=config.get('start_token',None),end_token=config.get('end_token',None),completion_only=hparams['completion_only'])
            eval_dataset = AlpacaDataset(eval_data, template_info, handler, max_length=max_length,combine_system_user=config.get('combine_system_user',False),start_token=config.get('start_token',None),end_token=config.get('end_token',None),completion_only=hparams['completion_only'])
            
        print (f'Dataset used: {train_dataset.__class__.__name__}')
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

    print (f'Type of dataset used: {train_dataset.__class__.__name__}')
    if hparams['mode'] == 'dpo':
        data_collator = None
    else:
        data_collator = CustomDataCollator(
            tokenizer=handler.tokenizer,
            mlm=False,
            pad = True if 'tool' in train_dataset_path.lower() else False
        )
    
    if handler.tokenizer.pad_token is None:
        print("WARNING: pad_token is None")
        
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
        handler.model.config.use_cache = False 
        ref_model.config.use_cache = False
    else:
        training_args = TrainingArguments(**training_args)
 
    handler.model.config.use_cache = False
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    total_steps_per_epoch = math.ceil(
    len(train_dataset) / (world_size * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
)
    print(f"Total steps per epoch: {total_steps_per_epoch}")
    callbacks = [EmbeddingRotationCallback(total_steps_per_epoch)] if hparams["gradual_rotation"] else []
    
    if hparams['mode'] == 'sft':
        trainer = SFTTrainer(
            model=handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=handler.tokenizer,
            data_collator=data_collator,
            # loss_log_file=log_file_json,
            callbacks=callbacks,
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
            'callbacks': callbacks,
        }
        if handler.embedding_type in ['ise','forward_rot']:
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
    
    # do a single eval before training
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
    # Update config with new checkpoint info
    new_checkpoint_info = {
        "desc": f"{pure_model_info['desc']} trained with {train_version}",
        "name": f"{pure_model_info['name']}_{train_version}",
        "checkpoint_path": output_dir,
        "instruct_model_path": instruct_model_path,
        "data_model_path": instruct_model_path,
        "chat_template_path": chat_template_path,
        "parent_pure_model_name": pure_model_info['name'],
        "run_number": run_number,
        "hyperparams": hparams
    }

    run_info_file = os.path.join(train_logs_path, "run_info.json")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_main_process(local_rank):
        with open(run_info_file, "w+") as f:
            json.dump(new_checkpoint_info, f)


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
    parser.add_argument("--model_family", type=str, help="E.g., tiny_old, llama3_8b, etc.")
    parser.add_argument("--emb_type", type=str, choices=["double_emb", "single_emb",  "ise", "forward_rot"], help="Embedding Type")
    parser.add_argument("--model_ix", type=int, help="Index of the model in the pure_models list.")
    parser.add_argument("--run_number", type=str, default=0, help="Number of the run.")
    parser.add_argument("--train_version", type=str, help="e.g. SFTv11")

    parser.add_argument("--train_type", type=str, default="full", help="full or lora")
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
    model_family = user_hparams.pop("model_family")
    emb_type = user_hparams.pop("emb_type")
    model_ix = user_hparams.pop("model_ix")
    run_number = user_hparams.pop("run_number")
    train_version = user_hparams.pop("train_version")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    print (f'Mode: {args.mode}')
    main(model_family, emb_type, train_version, model_ix, run_number, user_hparams)
