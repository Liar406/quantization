#!/bin/bash 
configure_id=$1
model=$2
# output_dir=$3
task=$3
# task=hellaswag,rte,mmlu
echo "${configure_id} ${task}"
lm_eval --model hf \
    --model_args pretrained=${model} \
    --tasks $task \
    --device cuda:0 \
    --batch_size 32 \
    --output_path results/ours/${configure_id}.json
    
