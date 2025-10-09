cd quantization_metric/
model=../models/patch/Llama-2-7b-hf-quantization-preliminary
output_dir=preliminary_extend
for i in {5..21}; do
    if [ -f "bit_layers/${output_dir}/preliminary_${i}.json" ]; then
        echo $i
        start=$(date +%s.%N)
        
        python main_low.py --bit_layers bit_layers/${output_dir}/preliminary_${i}.json --save_dir ${model}
        end=$(date +%s.%N)
        runtime=$(awk "BEGIN {print $end - $start}")
        cd ../lm-evaluation-harness
        bash run_scripts/eval_low.sh preliminary_${i}_84 ${model} ${output_dir}
        rm -rf ${model}
        cd ../quantization_metric/
        
        end=$(date +%s.%N)
        runtime=$(awk "BEGIN {print $end - $start}")
        echo "Execution time: $runtime seconds"
    else
        echo "bit_layers/${output_dir}/preliminary_${i}.json 不存在"
    fi

    
done

# python main_low.py --bit_layers bit_layers/sides2middle/configure_.json --save_dir ../models/patch/Llama-2-7b-hf-quantization
# cd ../lm-evaluation-harness
# bash run_scripts/eval.sh ${i}
# cd ../quantization_metric/