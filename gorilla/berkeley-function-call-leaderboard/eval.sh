#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# lora version (prompt)
model_name='Qwen/Qwen3-8B-ASIDE-ToolAlpaca-SFT' # ASIDE/Vanilla swapped with FC
test_category='all'
local_model_path='/mnt/disk1/wjyeo/models/Qwen3-8B_sft_ASIDE_ToolAndAlpaca' # use for aside and use lower  gpu memory
gpu_memory=0.9
bfcl generate \
  --model $model_name \
  --test-category $test_category \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization $gpu_memory \
  --local-model-path $local_model_path 

bfcl evaluate --model $model_name --test-category $test_category
