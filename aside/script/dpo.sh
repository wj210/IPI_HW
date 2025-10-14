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

output_dir="/dataset/common/huggingface/model" # SET YOUR SAVED PATH HERE
export WANDB_API_KEY="4d0cfb6b964e4092b544eaa50ffa07ae36cc5249" # SET YOUR WANDB API KEY HERE
export WANDB_PROJECT="ToolAlpaca_ASIDE_ISE_DPO" # SET YOUR WANDB PROJECT NAME HERE


extra_name='ToolAlpaca_ASIDE'
config_path='./configs/qwen/tool_and_alpaca_aside_dpo.json' ## for MetaSecAlign with both alpaca and tool data

beta=0.01 # try smaller beta for more stable training
lr=1e-6

for beta in 0.1 0.05 0.01
do
  deepspeed --master_port=29509 train.py \
    --config_path $config_path \
    --emb_type forward_rot \
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
    --extra_names "${extra_name}_b${beta}" \
    --num_data 20000 \
    --mode dpo \
    --beta $beta 
done