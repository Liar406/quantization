import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
# 示例数据
x = [
# 'NormalE-Dec',
'NormalE-Uni',
# 'NormalE-Inc',
'NormalE-NormalR',
# 'MoLA(2468)-Dec',
'MoLA(2468)-Uni',
# 'MoLA(2468)-Inc',
'MoLA(2468)-NormalR',
# 'MoLA(5555)-Dec',
# 'MoLA(5555)-Uni',
# 'MoLA(5555)-Inc',
# 'MoLA(5555)-NormalR',
# 'MoLA(8642)-Dec',
'MoLA(8642)-Uni',
# 'MoLA(8642)-Inc',
'MoLA(8642)-NormalR',
]

y1 = [
# 84.927,
84.521,
# 83.072,
83.826,
# 84.347,
83.014,
# 83.246,
84.231,
# 83.71,
# 84.115,
# 83.42,
# 82.782,
# 83.826,
83.768,
# 82.724,
84.289
]
y2 = [
# 92.086,
91.321,
# 91.232,
91.816,
# 91.951,
90.737,
# 91.052,
91.906,
# 91.906,
# 91.996,
# 91.456,
# 91.501,
# 92.176,
91.996,
# 91.456,
91.636,
]


max_y1 = max(y1)
idx_max_y1 = y1.index(max_y1)

# min_y1 = min(y1)
# idx_min_y1 = y1.index(min_y1)

max_y2 = max(y2)
idx_max_y2 = y2.index(max_y2)

# min_y2 = min(y2)
# idx_min_y2 = y2.index(min_y2)

plt.figure(figsize=(14, 9))
plt.plot(x, y1, label='MRPC', marker='o')
plt.plot(x, y2, label='ScienceQA', marker='s')
plt.ylim(80, 95)
# 最高分水平虚线
plt.axhline(max_y1, ls='--', lw=1, color='tab:orange')
plt.axhline(max_y2, ls='--', lw=1, color='tab:blue')

# 最低分水平虚线
# plt.axhline(min_y1, ls=':', lw=1, color='tab:orange')
# plt.axhline(min_y2, ls=':', lw=1, color='tab:blue')

# 最高分竖直虚线
plt.axvline(idx_max_y1, ls='--', lw=1, color='tab:orange')
plt.axvline(idx_max_y2, ls='--', lw=1, color='tab:blue')

# 最低分竖直虚线
# plt.axvline(idx_min_y1, ls=':', lw=1, color='tab:orange')
# plt.axvline(idx_min_y2, ls=':', lw=1, color='tab:blue')

# 最高分标注
plt.text(idx_max_y1, max_y1 + 0.2, 'The highest score on MRPC', ha='center', va='bottom', color='tab:orange')
plt.text(idx_max_y2, max_y2 + 0.2, 'The highest score on ScienceQA', ha='center', va='bottom', color='tab:blue')

# 最低分标注
# plt.text(idx_min_y1, min_y1 - 0.6, 'The lowest score on MRPC', ha='center', va='top', color='tab:orange')
# plt.text(idx_min_y2, min_y2 - 0.6, 'The lowest score on ScienceQA', ha='center', va='top', color='tab:blue')

# 交点画叉，突出显示
plt.scatter(idx_max_y1, max_y1, marker='x', color='tab:orange', s=100, lw=2)
# plt.scatter(idx_min_y1, min_y1, marker='x', color='tab:orange', s=100, lw=2)
plt.scatter(idx_max_y2, max_y2, marker='x', color='tab:blue', s=100, lw=2)
# plt.scatter(idx_min_y2, min_y2, marker='x', color='tab:blue', s=100, lw=2)

plt.title('Performance Comparison Across 16 Expert Number and Rank Configurations')
plt.xlabel('Configurations')
plt.ylabel('Scores')
plt.xticks(range(len(x)), x, rotation=90, ha='center')
plt.legend()
plt.tight_layout()
plt.savefig('Downstream-line.pdf', format='pdf', bbox_inches='tight')
plt.close()
