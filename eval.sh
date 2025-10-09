export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"

cd quantization_metric/
model=../models/patch/Llama-2-7b-hf-quantization
output_dir=KurtosisAlpha_values
tasks=piqa,winogrande,arc_easy,arc_challenge,hellaswag,boolq
# bit_layers_dir=/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/bit_layers
# result_dir=/mnt/bn/life-mllm/users/cxr/quantization/lm-evaluation-harness/results


start=$(date +%s.%N)
# rm -rf $model


for file in /mnt/bn/life-mllm/users/cxr/quantization/ours2/*; do
    echo "$file"
    cd ../quantization_metric
    configure_id=$(basename $file .json)
    python -u main_low.py --bit_layers $file  --save_dir ${model}
    cd ../lm-evaluation-harness
    bash run_scripts/eval.sh ${configure_id} ${model} ${tasks}
    rm -rf ${model}
    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "Execution time: $runtime seconds"
done


