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
export WANDB_PROJECT="Tool_DPO"


# deepspeed --master_port=29509 fine-tune.py \
# --model_family qwen3_8b \
# --train_version SFTv70_Tool_dpo \
# --emb_type single_emb \
# --model_ix 0 \
# --run_number Vanilla \
# --train_type full \
# --num_train_epochs 1 \
# --per_device_train_batch_size 4 \
# --gradient_accumulation_steps 4 \
# --learning_rate 5e-7 \
# --lr_scheduler_type cosine \
# --warmup_ratio 0.1 \
# --logging_steps 10 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --eval_steps 1 \
# --save_steps 1 \
# --save_total_limit 1 \
# --load_best_model_at_end True \
# --prediction_loss_only True \
# --bf16 True \
# --embedding_init rot_isoclinic \
# --rotation_alpha 1.57079633 \
# --learned_rotation False \
# --add_linear_shift False \
# --rotation_direction right \
# --gradual_rotation False \
# --output_dir $output_dir \
# --report_to wandb \
# --max_length 4096 \
# --mode dpo \
# --beta 0.1 \
# --num_data 40000 


deepspeed --master_port=29509 fine-tune.py \
--model_family qwen3_8b \
--train_version SFTv70_Tool_dpo \
--emb_type forward_rot \
--model_ix 0 \
--run_number ASIDE \
--train_type full \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-7 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--logging_steps 10 \
--evaluation_strategy epoch \
--save_strategy epoch \
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
--max_length 4096 \
--mode dpo \
--beta 0.1 \
--num_data 40000 