#!/bin/bash 
# tasks=triviaqa
tasks=(hellaswag rte mmlu)
models=('Llama-2-7b-hf' 'Llama-2-7b-qint4' 'Llama-2-7b-qint8' 'Llama-2-7b-qint-top_m_alpha_values')

# tasks=winogrande
for task in "${tasks[@]}"; do
    mkdir -p results/${task}
    for model in "${models[@]}"; do
        echo $model
        lm_eval --model hf \
            --model_args pretrained=../models/${model} \
            --tasks $task \
            --device cuda:0 \
            --batch_size 8 \
            --output_path results/${task}/${model}.json

    done
done