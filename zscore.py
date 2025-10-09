import numpy as np
import json

def kurtosis_outlier_layers(kurtosis_values, threshold=3.0, method="subtract"):
    """
    根据 Kurtosis 值列表，计算差分并用 z-score 检测异常层

    参数:
        kurtosis_values (list or np.ndarray): Kurtosis 值序列 (s1, s2, ..., sn)
        threshold (float): z-score 阈值 (默认 3.0)
        method (str): 差分方式
            "subtract" -> si+1 - si
            "divide"   -> si+1 / si

    返回:
        z_scores (np.ndarray): 差分的 z-score
        outlier_indices (list): 被判定为异常的层索引 (对应原始 Kurtosis 序列中的层号)
    """
    values = np.array(kurtosis_values, dtype=float)

    # Step 1: 差分
    if method == "subtract":
        diffs = np.diff(values)  # s2-s1, s3-s2, ...
    elif method == "divide":
        diffs = values[1:] / values[:-1]
    else:
        raise ValueError("method must be 'subtract' or 'divide'")

    # Step 2: 计算 z-score
    mean = np.mean(diffs)
    std = np.std(diffs, ddof=1)  # 样本标准差
    z_scores = abs((diffs - mean) / std)
    results = [(i+1, z) for i, z in enumerate(z_scores)]
    results.sort(key=lambda x: x[1], reverse=False)
    print(results)
    results = [i[0] for i in results]
    
    return results

k_path = '/mnt/bn/life-mllm/users/cxr/quantization/lm-quant-toolkit/kurtosis_means/kurtosis_means-Llama-2-7b-hf.json'
with open(k_path, "r", encoding="utf-8") as f:
    kurtosis_values = json.load(f)

kurtosis_outlier_layers(kurtosis_values)