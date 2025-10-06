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


# model_name='Qwen/Qwen3-8B-MetaSecAlign-DPO-1e6' # ASIDE/Vanilla swapped with FC
# test_category='all'
# local_model_path='/dataset/common/huggingface/model/Qwen3-8B_1e-6_MetaSecAlign_DPO' # use for aside and use lower  gpu memory
# gpu_memory=0.9
# bfcl generate \
#   --model $model_name \
#   --test-category $test_category \
#   --backend vllm \
#   --num-gpus 1 \
#   --gpu-memory-utilization $gpu_memory \
#   --local-model-path $local_model_path \
#   --tool-role input

# bfcl evaluate --model $model_name --test-category $test_category 

model_name='Qwen/Qwen3-8B-MetaSecAlign-DPO-1e5' # ASIDE/Vanilla swapped with FC
test_category='all'
local_model_path='/dataset/common/huggingface/model/Qwen3-8B_1e-5_MetaSecAlign_DPO' # use for aside and use lower  gpu memory
gpu_memory=0.9
bfcl generate \
  --model $model_name \
  --test-category $test_category \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization $gpu_memory \
  --local-model-path $local_model_path \
  --tool-role input

bfcl evaluate --model $model_name --test-category $test_category 

model_name='Qwen/Qwen3-8B_ASIDE_MetaSecAlign_SFT' # ASIDE/Vanilla swapped with FC
test_category='all'
local_model_path='/dataset/common/huggingface/model/Qwen3-8B_ASIDE_MetaSecAlign_SFT' # use for aside and use lower  gpu memory
gpu_memory=0.9
bfcl generate \
  --model $model_name \
  --test-category $test_category \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization $gpu_memory \
  --local-model-path $local_model_path \
  --tool-role input

bfcl evaluate --model $model_name --test-category $test_category 

