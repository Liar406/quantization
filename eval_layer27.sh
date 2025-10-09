cd quantization_metric/
model=../models/patch/Llama-2-7b-hf-quantization-27
output_dir=layer27_extend
for i in {21..28}; do
    echo $i
    start=$(date +%s.%N)
    
    python main_low.py --bit_layers bit_layers/${output_dir}/configure_${i}-4to2.json --save_dir ${model}
    cd ../lm-evaluation-harness
    bash run_scripts/eval_low.sh configure_${i}-4to2 ${model} ${output_dir}
    rm -rf /mnt/bn/life-mllm/users/cxr/quantization/models/patch/Llama-2-7b-hf-quantization-27
    cd ../quantization_metric/
    
    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "Execution time: $runtime seconds"
done

# python main_low.py --bit_layers bit_layers/sides2middle/configure_.json --save_dir ../models/patch/Llama-2-7b-hf-quantization
# cd ../lm-evaluation-harness
# bash run_scripts/eval.sh ${i}
# cd ../quantization_metric/