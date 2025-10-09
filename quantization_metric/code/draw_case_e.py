import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 13})
# 横坐标：1 到 32
x = np.arange(1, 33)

# 两组纵坐标数据
model_name = 'LLaMA-2-7B'
y1 = [6.0, 5.5, 3.75, 5.25, 3.75, 4.75, 5.25, 4.25, 3.75, 4.0, 4.75, 4.75, 5.25, 4.25, 5.25, 4.75, 3.5, 4.75, 3.75, 4.25, 4.25, 5.0, 3.0, 6.0, 4.5, 5.25, 4.75, 3.5, 4.75, 4.5, 6.0, 4.25]
y2 = [5.67, 3.33, 3.33, 5.0, 4.67, 4.33, 6.33, 5.0, 5.0, 4.0, 4.0, 4.0, 4.67, 5.0, 7.0, 5.33, 6.0, 5.0, 4.33, 7.0, 6.33, 4.67, 7.67, 6.0, 6.0, 5.67, 6.0, 5.0, 5.67, 6.67, 6.33, 6.33]

# model_name = 'Mistral-7B'
# y1 = [6.5, 5.75, 4.75, 5.75, 3.5, 5.0, 5.75, 6.0, 5.25, 3.5, 5.75, 5.75, 6.0, 5.25, 6.5, 5.0, 3.5, 4.75, 4.25, 5.5, 3.75, 5.0, 4.25, 5.75, 4.5, 5.25, 6.0, 3.75, 6.0, 5.75, 5.5, 6.0]
# y2 = [5.0, 5.33, 4.0, 3.67, 5.0, 6.33, 5.0, 5.67, 4.0, 6.33, 6.33, 6.67, 5.33, 6.67, 4.0, 5.0, 6.0, 6.67, 6.33, 7.0, 6.0, 6.67, 7.0, 5.33, 6.67, 6.33, 7.33, 6.0, 7.0, 6.33, 6.67, 6.0]

# model_name = 'LLaMA-3-8B'
# y1 = [5.5, 5.75, 6.25, 6.0, 6.25, 6.25, 6.25, 5.75, 6.0, 6.25, 3.5, 5.75, 4.5, 5.25, 6.25, 5.75, 5.75, 3.25, 5.75, 4.0, 5.5, 6.25, 3.75, 4.75, 4.75, 4.25, 5.75, 6.25, 6.25, 5.0, 4.5, 5.0]
# y2 = [3.33, 4.33, 4.0, 5.0, 6.33, 6.0, 4.67, 4.0, 4.67, 4.0, 4.67, 6.33, 6.0, 6.33, 7.0, 6.67, 6.33, 5.33, 5.67, 6.67, 6.67, 5.0, 5.33, 6.33, 4.67, 6.0, 7.0, 6.0, 7.33, 6.67, 7.0, 6.67]

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

# 绘制柱状图
ax.bar(x - offset, y1, width=width, label='MHA', color='skyblue', edgecolor='black',alpha=0.7)
ax.bar(x + offset, y2, width=width, label='FFN', color='salmon', edgecolor='black',alpha=0.7)

# width = 0.33
# offset = width / 2  # 用于左右错开

# # 创建画布和坐标轴
# fig, ax = plt.subplots(figsize=(12, 3))

# # 绘制两组柱状图，错开显示
# ax.bar(x - offset, y1, width=width, label='MHA', color='skyblue', edgecolor='black',alpha=0.7)
# ax.bar(x + offset, y2, width=width, label='FFN', color='salmon', edgecolor='black',alpha=0.7)

# 设置标签与标题
ax.set_xlabel('Layer Index')
ax.set_ylabel('Expert Number')
ax.set_title(f'{model_name}')
ax.set_ylim(50, 200)

# # 设置横坐标只显示特定值
# xticks_to_show = [1, 5, 9, 13, 17, 21, 25, 29, 32]
# plt.xticks(xticks_to_show)

ax.set_title(f'{model_name}')
# ax.set_ylim(20, 60)

# 设置自定义 x 轴标签
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
# 添加图例
ax.legend(loc='upper right', fontsize=10)

# 保存图像

plt.tight_layout()
# plt.savefig(f'bar_chart-{model_name}-grouped.png', dpi=300)
plt.savefig(f'bar_chart-{model_name}-expert.pdf', format='pdf', bbox_inches='tight')
plt.close()