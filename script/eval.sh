#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


for model_path in MetaSecAlign/Qwen3-8B_dpo_Vanilla_ToolAlpaca_lr1e-6_b0.1 MetaSecAlign/Qwen3-8B_dpo_Vanilla_ToolAlpaca_lr1e-6_b0.01
do
  python src/eval_sep.py \
  --model_path $model_path 

  python src/eval_mcq_utility.py \
  --model_path $model_path
  
done    