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


output_dir="/dataset/common/huggingface/model/IPI" # change to place to store large models
export WANDB_API_KEY="4d0cfb6b964e4092b544eaa50ffa07ae36cc5249"
export WANDB_PROJECT="ToolLLM_SFT_Vanilla_TEST"

num_data=100000

deepspeed --master_port=29509 fine-tune.py \
--model_family qwen3_8b \
--train_version Tool_SFT \
--emb_type single_emb \
--model_ix 0 \
--run_number Vanilla_$num_data \
--train_type full \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 2e-6 \
--lr_scheduler_type cosine \
--warmup_ratio 0.04 \
--logging_steps 10 \
--evaluation_strategy epoch \
--save_strategy steps \
--eval_steps 1 \
--save_steps 1 \
--save_total_limit 6 \
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
--eval_percent 0.2 \
--completion_only \
--num_data $num_data \
