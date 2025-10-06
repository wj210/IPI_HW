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

model_path="Qwen/Qwen3-8B"
for model_path in Qwen3-8B_5e-7_MetaSecAlign_DPO Qwen3-8B_1e-5_MetaSecAlign_DPO
do
  python src/eval_agentdojo.py \
  --model_path $model_path \
  --use_vllm 
done    