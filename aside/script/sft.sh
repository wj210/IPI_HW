#!/bin/bash
echo "Running job with CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
cuda_path="cuda_visible_devices.txt"
if [ -f $cuda_path ]; then
  export CUDA_VISIBLE_DEVICES=$(cat $cuda_path)
  num_gpu=$(cat "$cuda_path" | tr ', ' '\n' | grep -c '[0-9]')
  echo "num gpus = $num_gpu"
else
  echo "cuda_visible_devices.txt file not found."
fi


output_dir="/dataset/common/huggingface/model" # change to place to store large models
export WANDB_API_KEY="4d0cfb6b964e4092b544eaa50ffa07ae36cc5249"
export WANDB_PROJECT="MetaSecAlign_DPO_ASIDE_ISE"

config_path='./configs/qwen/tool_and_alpaca_sft.json' # includes both alpaca and tool data
extra_names="ToolAndAlpaca" # extra names to add to output dir

deepspeed --master_port=29509 train.py \
--config_path $config_path \
--emb_type forward_rot \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--learning_rate 2e-5 \
--lr_scheduler_type cosine \
--warmup_ratio 0.04 \
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
--extra_names $extra_names \
--completion_only
