import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})  # 统一把默认字体调到 12pt

# Example data: 3 groups, each with 8 bars
groups = ['8-th layer', '16-th layer', '24-th layer']
n_groups = len(groups)
n_series = 8
bar_width = 0.11  # width of each bar
x = np.arange(n_groups)

# Your actual data
data1 = np.array([84.32, 84.11, 83.99, 83.82,  84.21, 83.85, 83.96, 83.66])
data2 = np.array([84.53, 83.86, 84.32, 84.24, 83.79, 84.36, 83.98, 84.42])
data3 = np.array([84.22, 83.93, 84.02, 84.24, 83.97, 83.85, 84.13, 83.90])
for data in [data1, data2, data3]:
    for i, dt in enumerate(data):
        data[i] = np.round(int(1043 * dt /100) / 1043 * 100, 2)
        
print(data1)
# Construct data matrix (8 bars, 3 groups)
data = np.vstack([data1, data2, data3]).T  # shape (8,3)

# Soft pastel color palette
colors = [
    '#D6EAF8',  # very light blue
    '#FAD7A0',  # light pastel orange
    '#D5F5E3',  # light pastel green
    '#F2F3F4',  # very light gray
    '#E8DAEF',  # soft lavender
    '#FADBD8',  # light pink
    '#FCF3CF',  # light yellow
    '#F5B7B1'   # soft coral pink
]

plt.figure(figsize=(8, 5))

# [A\textsuperscript{2}MoLE, ex +- 2, ex +-4, rank一半+-2, rank一半+-4, rank总数一定-随机分配]
labels = ['GuiLoMo', 'IEN(2)', 'IEN(4)', 'DEN(2)', 'DEN(4)', 'MRA_half(2)', 'MRA_half(4)', 'MRA_random']
# Plot each bar with pastel color and black edge
for i in range(n_series):
    plt.bar(
        x + i * bar_width,
        data[i],
        width=bar_width,
        label=labels[i],
        color=colors[i],
        edgecolor='black',
        linewidth=0.8
    )

# Configure axes
plt.xticks(
    x + (n_series - 1) * bar_width / 2,
    groups
)
# plt.xlabel('')
# plt.ylabel('COLA')
# plt.title('Case Study Across 8 Perturbations')
plt.legend(
    ncol=4,
    fontsize='small',
    bbox_to_anchor=(1, 1),
    loc='upper right'
)

# —— 新增：固定纵轴范围及刻度间隔 —— 
plt.ylim(83, 85)
plt.yticks(np.arange(83, 85 + 1e-6, 0.3))

plt.tight_layout()

# Save without showing
output_path = 'grouped_bar_chart_pastel.pdf'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path, format='pdf', bbox_inches='tight')
