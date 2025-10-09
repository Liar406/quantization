result_dir=/mnt/bn/life-mllm/users/cxr/quantization/lm-evaluation-harness/results
output_dir=singletask_arc_easy
bits=6
bit_layers_dir=/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/bit_layers
python final_bpc.py --result_dir "${result_dir}/${output_dir}" --bits ${bits} --configure_dir "${bit_layers_dir}/${output_dir}"
