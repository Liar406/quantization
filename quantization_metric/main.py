from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from alphalora.expert_number import calculate_expert
import json
import torch
from lsaq_quant import quantize_llama_like
import json
import numpy as np

kurtosis_path = '../lm-quant-toolkit/kurtosis_means/kurtosis_means-Llama-2-7b-hf.json'
model_id = "../models/Llama-2-7b-hf"
alpha_path='../chenxr/alpha_values/alpha_values_llama-2-7b.json'
with open(kurtosis_path, 'r', encoding='utf-8') as f:
    kurtosis_means = json.load(f)
with open(alpha_path, 'r', encoding='utf-8') as f:
    alpha_values = json.load(f)
m=5
top_m_kurtosis = np.argsort(kurtosis_means)[-m:][::-1].tolist()
top_m_alpha_values = np.argsort(alpha_values)[-m:][::-1].tolist()
print('top_m_indices', top_m_kurtosis)
print('top_m_alpha_values', top_m_alpha_values)
intersection = list(set(top_m_kurtosis) & set(top_m_alpha_values))
print('intersection', intersection)

quant_config = QuantoConfig(weights="int8")  # 仅对 Linear 层权重量化

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:0",
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

bit = 2
bit_layers = [4]*32

for indix in top_m_kurtosis:
    bit_layers[indix] = 8
for indix in intersection:
    bit_layers[indix] = 4
# for indix in top_m_alpha_values:
#     bit_layers[indix] = 2
layer_to_quant = list(range(32))
print(layer_to_quant)
mlp_quant = [f'layers.{item}.mlp' for item in layer_to_quant]
self_attn_quant = [f'layers.{item}.self_attn' for item in layer_to_quant]

print(f'quanting ... ')
model_lsaq = quantize_llama_like(model, mlp_quant, self_attn_quant, bit, bit_layers)
print(f'quanted')
print(model_lsaq)

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

# 计算 alpha_values
# all_layer_alpha = calculate_expert(model)
# with open(f"alpha_values_llama-2-7b-qint{bit}.json", "w") as f:
#     json.dump(all_layer_alpha, f, indent=4)

save_dir = f"../models/Llama-2-7b-qint-k3i2"
model_lsaq.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
