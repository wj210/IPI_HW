#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2,3,4

output_dir="/mnt/disk1/wjyeo/models/MetaSecAlign" # change to place to store large models
export WANDB_API_KEY="4d0cfb6b964e4092b544eaa50ffa07ae36cc5249"
export WANDB_PROJECT="MetaSecAlign_DPO_Vanilla"


extra_name='ToolAlpaca'
config_path='./configs/qwen/tool_and_alpaca_vanilla_dpo.json' ## for MetaSecAlign with both alpaca and tool data

beta=0.01 # try smaller beta for more stable training
lr=1e-6

deepspeed --master_port=29509 train.py \
  --config_path $config_path \
  --emb_type single_emb \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate $lr \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --evaluation_strategy epoch \
  --save_strategy steps \
  --eval_steps 1 \
  --save_steps 1 \
  --save_total_limit 1 \
  --load_best_model_at_end True \
  --prediction_loss_only True \
  --bf16 True \
  --embedding_init rot_isoclinic \
  --rotation_alpha 1.57079633 \
  --learned_rotation False \
  --add_linear_shift False \
  --rotation_direction right \
  --gradual_rotation False \
  --output_dir $output_dir \
  --report_to wandb \
  --eval_percent 0.2 \
  --extra_names "${extra_name}_lr${lr}_b${beta}" \
  --num_data 20000 \
  --mode dpo \
  --beta $beta 
