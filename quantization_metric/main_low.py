from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from alphalora.expert_number import calculate_expert
import json
import torch
from lsaq_quant import quantize_llama_like
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="parser")
parser.add_argument("--bit_layers", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
args = parser.parse_args()

model_id = "../models/Llama-2-7b-hf"

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
# bit_layers = [4]*32
with open(args.bit_layers, 'r', encoding='utf-8') as f:
    bit_layers = json.load(f)
# bit_layers = json.loads(args.bit_layers)

assert len(bit_layers) == 32
# set some layers to 2 bit
# for index in [0, 1, 11]:
#     bit_layers[index] = 8

# [28, 26, 20, 16, 24]
# for index in list(range(24,29)):
#     bit_layers[index] = 2

print('bit_layers: ', bit_layers)

layer_to_quant = list(range(32))
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

# save_dir = f"../models/Llama-2-7b-hf-Kurtosis_only"
model_lsaq.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
