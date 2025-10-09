#!/bin/bash 
configure_id=$1
model=$2
output_dir=$3
# task=piqa,winogrande,arc_easy,arc_challenge,hellaswag,boolq
task=hellaswag,rte,mmlu
echo "${configure_id} ${task}"
lm_eval --model hf \
    --model_args pretrained=${model} \
    --tasks $task \
    --device cuda:0 \
    --batch_size 32 \
    --output_path results/${output_dir}/Llama-2-7b-hf-${configure_id}.json
    
# tasks=(piqa winogrande hellaswag arc_easy arc_challenge wikitext)
# models=('Llama-2-7b-hf')

# for task in "${tasks[@]}"; do
#     mkdir -p results/${task}
#     for model in "${models[@]}"; do
#         echo "${model} ${task}"
#         lm_eval --model hf \
#             --model_args pretrained=../models/${model} \
#             --tasks $task \
#             --device cuda:0 \
#             --batch_size 32 \
#             --output_path results/${model}/${task}.json
#     done
# done
