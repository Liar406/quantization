import matplotlib.pyplot as plt
import numpy as np
import json
e_path = '/Users/chenxinrong02/Desktop/src/MoLA_con/arch/expert/arch_standardnormal-rank_difference-2.json'
plt.rcParams.update({'font.size': 13})
import json

# 假设你的 json 文件名是 data.json
with open(e_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

y1 = []
for i in range(32):
    y1.append(data[f"model.layers.{i}.mlp.gate_proj"])

x = np.arange(1, 33)



fig, ax = plt.subplots(figsize=(12, 3))

# 只绘制一组柱状图，居中显示
ax.bar(x, y1, width=0.6, label='Expert Number per Layer', color='skyblue', edgecolor='black', alpha=0.7)

ax.set_xlabel('Layer Index')
ax.set_ylabel('Expert Number')
ax.set_title(f'NormalE')
ax.set_ylim(0, 10)

xticks_to_show = [1, 5, 9, 13, 17, 21, 25, 29, 32]
plt.xticks(xticks_to_show)

ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(f'normal_e.pdf', format='pdf', bbox_inches='tight')
plt.close()
