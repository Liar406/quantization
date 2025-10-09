# model_id=("Qwen3-4B" "Qwen3-8B" "Llama-3.2-3B-Instruct" "Llama-2-13b-hf")
model_id=("Llama-3.1-70B-Instruct")

for m in "${model_id[@]}"; do 
    echo $m
    MODELS="../models/$m"
    # MODELS="meta-llama/Llama-2-7b-hf"
    mkdir -p tmp/kurtosis-dump/"$(basename $MODELS)"
    python src/dump.py kurtosis \
        --model $MODELS \
        --output-dir tmp/kurtosis-dump/"$(basename $MODELS)"
    
    python test.py --model $m
    
done
