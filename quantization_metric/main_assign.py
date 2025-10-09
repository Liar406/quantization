from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from alphalora.expert_number import calculate_expert
import json
import torch
from lsaq_quant_assign import quantize_llama_like
import os 

# model_names = ['Qwen3-8B', 'Qwen3-4B', 'Mistral-7B-Instruct-v0.3','Llama-3.2-3B-Instruct']
model_names = ['Llama-2-7b-hf']
for model_name in model_names:
    model_id = f"../models/{model_name}"
    # Llama-3.2-3B-Instruct
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,        # 随机采样（非贪婪）
            top_k=50,              # 限制采样候选
            top_p=0.95,            # nucleus sampling
            temperature=0.7        # 控制生成多样性
        )

    # 解码为字符串
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)
    print(model)
    # 计算 alpha_values
    all_layer_alpha = calculate_expert(model)
    with open(f"ZD/ZD_{os.path.basename(model_id)}.json", "w") as f:
        json.dump(all_layer_alpha, f, indent=4)

# for bit in [8]:
#     print(bit)
#     layer_to_quant = list(range(32))
#     print(layer_to_quant)
#     mlp_quant = [f'layers.{item}.mlp' for item in layer_to_quant]
#     self_attn_quant = [f'layers.{item}.self_attn' for item in layer_to_quant]

#     print(f'quanting ... ')
#     model_lsaq = quantize_llama_like(model, mlp_quant, self_attn_quant, 4)
#     print(f'quanted')
#     print(model_lsaq)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             do_sample=True,        # 随机采样（非贪婪）
#             top_k=50,              # 限制采样候选
#             top_p=0.95,            # nucleus sampling
#             temperature=0.7        # 控制生成多样性
#         )

#     # 解码为字符串
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(decoded)

    

#     # save_dir = f"../models/Llama-2-7b-hf-qint{bit}"
#     # model_lsaq.save_pretrained(save_dir)
#     # tokenizer.save_pretrained(save_dir)
