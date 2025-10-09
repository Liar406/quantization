import json
import numpy as np


alpha_path = '/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/alpha_values/alpha_values_llama-2-7b.json'
kurtosis_path = '/mnt/bn/life-mllm/users/cxr/quantization/lm-quant-toolkit/kurtosis_means/kurtosis_means-Llama-2-7b-hf.json'

# 打开并读取文件
with open(alpha_path, 'r', encoding='utf-8') as f:
    alpha_datas = json.load(f)

with open(kurtosis_path, 'r', encoding='utf-8') as f:
    kurtosis_datas = json.load(f)

sigma = 1e-6

def Boxplot_clip(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Clip 阈值
    k = 1.5
    clip_min = Q1 - k * IQR
    clip_max = Q3 + k * IQR

    # Clip
    data_clipped = np.clip(data, clip_min, clip_max)
    return data_clipped

# data = []
# for kurtosis_data, alpha_data in zip(kurtosis_datas, alpha_datas):
#     data.append(
#         kurtosis_data/(alpha_data + sigma)
#     )
# print(data)
# scores_array = np.array(data)
# # 从大到小排序的索引
# indices = np.argsort(-scores_array)
# print(indices)

kurtosis_array = np.array(kurtosis_datas)
kurtosis_array = Boxplot_clip(kurtosis_array)
kurtosis_normalized = kurtosis_array / (np.sum(kurtosis_array) + 1e-8)

alpha_array = np.array(alpha_datas)
alpha_array = Boxplot_clip(alpha_array)
alpha_normalized = alpha_array / (np.sum(alpha_array) + 1e-8)
print(kurtosis_normalized)
print(alpha_normalized)

import cvxpy as cp
import numpy as np

layers = len(alpha_normalized)
bits = np.array([4,8])
num_bits = len(bits)

a = alpha_normalized # importance
b = kurtosis_normalized # sensitivity
# diffs = [bb - aa for aa, bb in zip(a, b)]
diffs = [-aa  for aa, bb in zip(a, b)]

print(diffs)
sorted_indices = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)
print('sorted_indices', sorted_indices)

alpha_var = cp.Variable((layers, num_bits), boolean=True)

beta = 0.5
total_score = 0
B_max = max(bits)
for l in range(layers):
    B_l = cp.sum(cp.multiply(bits, alpha_var[l, :]))
    reward = kurtosis_normalized[l] - alpha_normalized[l]
    total_score += reward * (1 + B_l / B_max)




constraints = []

# 每层只能选一个 bit
for l in range(layers):
    constraints.append(cp.sum(alpha_var[l, :]) == 1)

# 总 bit budget 约束（比如总 budget = 16）
print(alpha_var @ bits)
budget = 4*32 + 4*16
total_bits = cp.sum(alpha_var @ bits)
constraints.append(total_bits <= budget)

# 定义并求解
prob = cp.Problem(cp.Maximize(total_score), constraints)
prob.solve(solver=cp.ECOS_BB)  # 也可以尝试 GUROBI 等

# 输出结果
for l in range(layers):
    chosen_bit = bits[np.argmax(alpha_var.value[l, :])]
    print(f"Layer {l}: chosen bit = {chosen_bit}")

allocated_bits = [bits[np.argmax(alpha_var.value[l, :])] for l in range(layers)]
print('allocated_bits', allocated_bits)
print(sum(allocated_bits))

def Boxplot_clip(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Clip 阈值
    k = 1.5
    clip_min = Q1 - k * IQR
    clip_max = Q3 + k * IQR

    # Clip
    data_clipped = np.clip(data, clip_min, clip_max)
    return data_clipped
