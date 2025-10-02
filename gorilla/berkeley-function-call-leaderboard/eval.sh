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


# export BFCL_PROJECT_ROOT=/export/home2/weijie210/ipi_huawei/gorilla/berkeley-function-call-leaderboard
model_name='Qwen/Qwen3-8B-FC'
test_category='all'
local_model_path='/dataset/common/huggingface/model/Qwen3-8B-Tool_ASIDE_SFT' # use for aside and use lower  gpu memory
gpu_memory=0.8
bfcl generate \
  --model $model_name \
  --test-category $test_category \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization $gpu_memory \
  --local-model-path $local_model_path \

# bfcl evaluate --model $model_name --test-category $test_category