import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 13})
# 横坐标：1 到 32
x = np.arange(1, 33)

# 两组纵坐标数据
model_name = 'LLaMA-2-7B'
y1 = [36.5, 37.75, 32.0, 44.5, 29.75, 37.0, 44.5, 30.5, 28.25, 29.75, 42.25, 35.5, 44.0, 35.25, 39.25, 44.5, 31.25, 41.0, 32.5, 36.5, 40.25, 42.75, 25.0, 52.5, 35.5, 45.25, 38.75, 31.0, 40.75, 40.25, 50.5, 35.75]
y2 = [49.33, 28.67, 28.0, 42.67, 39.0, 39.0, 54.67, 43.67, 42.67, 33.67, 35.33, 35.33, 36.0, 40.67, 63.33, 43.0, 51.33, 42.67, 37.0, 54.33, 57.0, 38.33, 66.33, 50.0, 51.33, 43.0, 51.0, 48.33, 49.33, 59.0, 54.0, 47.0]

model_name = 'Mistral-7B'
y1 = [36.25, 30.5, 30.0, 29.25, 40.0, 35.25, 36.0, 31.0, 33.0, 39.0, 17.75, 29.5, 22.75, 37.75, 37.75, 36.75, 38.0, 24.25, 40.75, 28.75, 41.75, 42.25, 27.0, 35.75, 36.25, 31.75, 42.5, 46.0, 47.75, 31.5, 33.25, 35.0]
y2 = [26.67, 31.67, 30.0, 33.67, 43.33, 44.0, 33.67, 51.0, 32.67, 27.33, 30.0, 43.33, 43.0, 45.67, 52.67, 48.67, 44.0, 41.33, 40.33, 50.67, 49.0, 39.67, 39.33, 45.33, 33.33, 46.67, 54.0, 47.33, 57.33, 50.0, 51.33, 42.0]

# model_name = 'LLaMA-3-8B'
# y1 = [36.25, 30.5, 34.5, 35.75, 40.0, 38.0, 36.0, 35.75, 33.0, 39.0, 17.75, 29.5, 22.75, 37.75, 37.75, 36.75, 38.0, 24.25, 40.75, 28.75, 41.75, 42.25, 27.0, 35.75, 36.25, 31.75, 42.5, 46.0, 47.75, 31.5, 33.25, 35.0]
# y2 = [26.67, 31.67, 30.0, 33.67, 43.33, 44.0, 32.67, 28.67, 32.67, 27.33, 30.0, 43.33, 43.0, 45.67, 52.67, 48.67, 44.0, 41.33, 40.33, 50.67, 49.0, 39.67, 39.33, 45.33, 33.33, 46.67, 54.0, 47.33, 57.33, 50.0, 51.33, 42.0]
y1 = [a*4 for a in y1]
y2 = [a*3 for a in y2]

y1 = [sum(y1[i:i+8]) for i in range(0, len(y1), 8)]
y2 = [sum(y2[i:i+8]) for i in range(0, len(y2), 8)]
print(y1)
print(y2)
x_labels = ['1~8', '9~16', '17~24', '25~32']
x = np.arange(len(x_labels))  # 0 到 7

# 设置柱宽和偏移
width = 0.35
offset = width / 2

# 创建图像
fig, ax = plt.subplots(figsize=(5, 3))

# width = 0.33
# offset = width / 2  # 用于左右错开

# # 创建画布和坐标轴
# fig, ax = plt.subplots(figsize=(12, 3))

# 绘制两组柱状图，错开显示
ax.bar(x - offset, y1, width=width, label='MHA', color='skyblue', edgecolor='black',alpha=0.7)
ax.bar(x + offset, y2, width=width, label='FFN', color='salmon', edgecolor='black',alpha=0.7)

# 设置标签与标题
ax.set_xlabel('Layer Index')
ax.set_ylabel('Submodule Ranks')
ax.set_title(f'{model_name}')
ax.set_ylim(600, 1300)

# 设置横坐标只显示特定值
# xticks_to_show = [1, 5, 9, 13, 17, 21, 25, 29, 32]
# plt.xticks(xticks_to_show)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
# 添加图例
ax.legend(loc='upper left', fontsize=10)

# 保存图像
plt.tight_layout()
plt.savefig(f'bar_chart-{model_name}-total.pdf', format='pdf', bbox_inches='tight')
# 不显示图像
plt.close()
