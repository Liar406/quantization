import json
import numpy as np
import os
from zscore import kurtosis_outlier_layers

def mean_and_var(arr):
    """
    输入: arr (np.ndarray 或能转成 np.array 的对象)
    输出: (mean, variance)
    """
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr)
    var = np.var(arr)   # 默认是总体方差，如果要无偏估计可用 ddof=1
    return mean, var

def select_layers_shaped(kurtosis, alpha, k=5, return_scores=False,
                         pos_gaussians=None, neg_gaussians=None):
    """
    强塑形联合指标:
      score_i = (alpha_i / kurtosis_i) * ( 1
                                           + sum_j b_j * exp(-((kurtosis_i - c_j)^2) / s_j^2)
                                           - sum_t d_t * exp(-((kurtosis_i - u_t)^2) / r_t^2) )

    参数
    ----
    kurtosis : list/ndarray   κ
    alpha    : list/ndarray   α
    k        : 取前 k 个下标
    pos_gaussians : [(c, s, b), ...]  正向高斯(中心c, 带宽s>0, 系数b>0)
    neg_gaussians : [(u, r, d), ...]  负向高斯(中心u, 带宽r>0, 系数d>0)

    返回
    ----
    idx_topk : list[int]
    (可选) scores : ndarray
    """
    krt = np.asarray(kurtosis, dtype=float)
    alp = np.asarray(alpha, dtype=float)
    if krt.shape != alp.shape:
        raise ValueError("kurtosis 与 alpha 形状不一致")

    base = alp - krt

    c, sigma = mean_and_var(krt)
    beta = 1.5
    weight = 1.0 + beta * np.exp(-((krt - c) ** 2) / (sigma ** 2))
    scores = base * weight
    return scores

def joint_score(kurtosis, alpha,
                alpha0=4.20, s_alpha=0.090, 
                s_k=1.254, w_alpha=9.48, w_k=1.02):
    """
    Kurtosis-Alpha 联合指标 (Taguchi 损失型)
    -------------------------
    参数
    kurtosis : list/ndarray
        每层的 kurtosis 值
    alpha : list/ndarray
        每层的 alpha 值
    alpha0 : float
        alpha 的目标值（名义最佳点）
    s_alpha : float
        alpha 的尺度因子
    s_k : float
        kurtosis 的尺度因子
    w_alpha, w_k : float
        alpha 和 kurtosis 的权重

    返回
    scores : ndarray
        每层的综合得分（越大越好）
    """
    k = np.array(kurtosis)
    a = np.array(alpha)

    loss_k = (k / s_k) ** 2
    loss_a = ((a - alpha0) / s_alpha) ** 2

    scores = -(w_k * loss_k + w_alpha * loss_a)
    return scores

def simple_joint_score(kurtosis, alpha, stable_rank):
    """
    极简联合指标:
    S_l = -k_l * (alpha_l - Q80(alpha))^2
    
    参数
    ----
    kurtosis : list/ndarray
        每层的 kurtosis 值
    alpha : list/ndarray
        每层的 alpha 值
    
    返回
    ----
    scores : ndarray
        每层的综合得分（越大越好）
    """
    k = np.array(kurtosis)
    a = np.array(alpha)
    s = np.array(stable_rank)
    print('alpha_hat_datas')
    # h = np.array(alpha_hat_datas)
    k = (k - k.min()) / (k.max() - k.min())
    # k_exp = np.exp(k)   # 防止溢出
    # k = k_exp / np.sum(k_exp)
    
    a = 1 / a
    a = (a - a.min()) / (a.max() - a.min())
    # a_exp = np.exp(a)   # 防止溢出
    # a = a_exp / np.sum(a_exp)

    s = 1 / s
    s = (s - s.min()) / (s.max() - s.min())
    # s_exp = np.exp(s)  
    # s = s_exp / np.sum(s_exp)

    # h = 1 / h
    # # h= (h - h.min()) / (h.max() - h.min())
    # h_exp = np.exp(h)   # 防止溢出
    # h = h_exp / np.sum(h_exp)
    
    # alpha_target = np.percentile(a, 80)   # α 的 80 分位数
    # # alpha_target = 0
    # scores = -k * (a - alpha_target) ** 2
    
    # k = k * 0.5
    # h = h * 0.5
    # print(k)
    # print(a)
    # print(h)
    # a = a + 0.3 * h


    # a = a +1
    # k = k +1 
    # h = h +1
    print('a', {i:aa for i, aa in enumerate(a)})
    print()
    print('k', {i:aa for i, aa in enumerate(k)})
    print()
    print('s', {i:aa for i, aa in enumerate(s)})
    # k = 100 * k
    # scores = (k / a) * (k - a)
    k = 0.2 * k
    s = 1.2 * s
    scores = k + a + s
    # scores = (k * h) * (k + h)
    # print()
    # print('scores', {i:aa for i, aa in enumerate(scores)})
    return scores

model_names = ['Llama-2-7b-hf',] 
            #    'Llama-2-13b-hf', 'Qwen3-8B', 'Qwen3-4B', 'Mistral-7B-Instruct-v0.3','Llama-3.2-3B-Instruct']

for model in model_names:
    # model = "Llama-2-7b-hf"
    print(model)
    alpha_path = f'/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/alpha_values/alpha_values_{model}.json'
    kurtosis_path = f'/mnt/bn/life-mllm/users/cxr/quantization/lm-quant-toolkit/kurtosis_means/kurtosis_means-{model}.json'
    alpha_hat_path = f'/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/alpha_hat/alpha_values_{model}.json'
    stable_rank_path = f'/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/stable_rank/stable_rank_{model}.json'
    zd_path = f'/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/ZD/ZD_{model}.json'
    bi_path = f'/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/BI/BI_{model}.json'

    with open(stable_rank_path, 'r', encoding='utf-8') as f:
        stable_rank_datas = json.load(f)

    with open(alpha_path, 'r', encoding='utf-8') as f:
        alpha_datas = json.load(f)

    with open(kurtosis_path, 'r', encoding='utf-8') as f:
        kurtosis_datas = json.load(f)

    with open(alpha_hat_path, 'r', encoding='utf-8') as f:
        alpha_hat_datas = json.load(f)

    with open(zd_path, 'r', encoding='utf-8') as f:
        zd_datas = json.load(f)

    with open(bi_path, 'r', encoding='utf-8') as f:
        bi_datas = json.load(f)

    print(len(kurtosis_datas))
    print('kurtosis_datas', np.argsort(kurtosis_datas))
    print('stable_rank_datas', np.argsort([-a for a in stable_rank_datas]))
    print('alpha_datas', np.argsort([-a for a in alpha_datas]))
    print('alpha_hat_datas', np.argsort([-a for a in alpha_hat_datas]))
    
    print('zd_datas', np.argsort([-a for a in zd_datas]))
    print('bi_datas', np.argsort([a for a in bi_datas]))

    scores = simple_joint_score(kurtosis_datas, alpha_datas, stable_rank_datas)
    idx_sorted = np.argsort(scores)
    # print('scores:', scores)
    print('idx_sorted:',idx_sorted)
    print()

    kurtosis_idx = np.argsort(kurtosis_datas).tolist()
    stable_rank_idx = np.argsort([-a for a in stable_rank_datas]).tolist()
    alpha_idx = np.argsort([-a for a in alpha_datas]).tolist()
    bi_idx = np.argsort([a for a in bi_datas]).tolist()
    z_idx = kurtosis_outlier_layers(kurtosis_datas)
    zd_idx = np.argsort([-a for a in zd_datas]).tolist()
    layrs = [5, 10]

    for layr in layrs:
    #     bits = [4 for _ in range(len(alpha_datas))]
    #     print(kurtosis_idx[:layr])
    #     for i in kurtosis_idx[:layr]:
    #         bits[i] = 2
    #     with open(os.path.join('/mnt/bn/life-mllm/users/cxr/quantization/baselines', f'{model}_kurtosis_idx_{layr}.json'), "w", encoding="utf-8") as f:
    #         json.dump(bits, f, ensure_ascii=False, indent=4)

    #     bits = [4 for _ in range(len(alpha_datas))]
    #     for i in alpha_idx[:layr]:
    #         bits[i] = 2
    #     with open(os.path.join('/mnt/bn/life-mllm/users/cxr/quantization/baselines', f'{model}_alpha_idx_{layr}.json'), "w", encoding="utf-8") as f:
    #         json.dump(bits, f, ensure_ascii=False, indent=4)

    #     bits = [4 for _ in range(len(alpha_datas))]
    #     for i in z_idx[:layr]:
    #         bits[i] = 2
    #     with open(os.path.join('/mnt/bn/life-mllm/users/cxr/quantization/baselines', f'{model}_z_idx_{layr}.json'), "w", encoding="utf-8") as f:
    #         json.dump(bits, f, ensure_ascii=False, indent=4)

        # bits = [4 for _ in range(len(alpha_datas))]
        # print(bi_idx[:layr])
        # for i in bi_idx[:layr]:
        #     bits[i] = 2
        # with open(os.path.join('/mnt/bn/life-mllm/users/cxr/quantization/baselines_1', f'{model}_bi_idx_{layr}.json'), "w", encoding="utf-8") as f:
        #     json.dump(bits, f, ensure_ascii=False, indent=4)

        kurtosis_datas, stable_rank_datas
        bits = [4 for _ in range(len(alpha_datas))]
        print(idx_sorted[:layr])
        for i in idx_sorted[:layr]:
            bits[i] = 2
        with open(os.path.join('/mnt/bn/life-mllm/users/cxr/quantization/ours2', f'{model}_aks_plus_idx_sorted_{layr}.json'), "w", encoding="utf-8") as f:
            json.dump(bits, f, ensure_ascii=False, indent=4)
        