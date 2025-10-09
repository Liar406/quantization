export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"

cd quantization_metric/
model=../models/patch/Llama-2-7b-hf-quantization-alltasks
output_dir=alltasks
tasks=piqa,winogrande,arc_easy,arc_challenge,hellaswag,boolq
bit_layers_dir=/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/bit_layers
result_dir=/mnt/bn/life-mllm/users/cxr/quantization/lm-evaluation-harness/results

for bits in {2..5}; do
    echo $bits
    start=$(date +%s.%N)
    layer_index=$(python obtain_bpc.py --result_dir "${result_dir}/${output_dir}" --bits ${bits} --configure_dir "${bit_layers_dir}/${output_dir}")
    echo "layer_index: ${layer_index}"
    python produce_maxbits.py --save_dir "${bit_layers_dir}/${output_dir}" --layer_index "$layer_index" --bits ${bits} #造配置文件

    mkdir -p ${result_dir}/${output_dir}/bits_${bits}
    for file in ${bit_layers_dir}/${output_dir}/bits_${bits}/*; do
        echo "$file"
        configure_id=$(basename "$file" .json)
        echo "${configure_id}"
        python -u main_low.py --bit_layers $file --save_dir ${model}
        cd ../lm-evaluation-harness
        bash run_scripts/eval.sh ${configure_id} ${model} "${output_dir}/bits_${bits}" ${tasks}
        rm -rf $model
        cd ../quantization_metric/
        end=$(date +%s.%N)
        runtime=$(awk "BEGIN {print $end - $start}")
        echo "Execution time: $runtime seconds"
    done    
    
done
