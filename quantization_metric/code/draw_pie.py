# import matplotlib.pyplot as plt

# plt.rcParams['axes.unicode_minus'] = False

# labels = ['75%~100%', '50%~75%', '25%~50%', '0%~25%']
# percentages = [45.1, 26.6, 14.6, 13.7]
# colors = ['#AFCBFF', '#FFD6A5', '#D6BCFA', '#C4F0C5']

# fig, ax = plt.subplots(figsize=(6, 6))
# wedges, texts, autotexts = ax.pie(
#     percentages, labels=labels, colors=colors,
#     autopct='%.1f%%', startangle=90, pctdistance=0.8
# )

# for text in texts:
#     text.set_fontsize(12)
# for autotext in autotexts:
#     autotext.set_fontsize(11)
#     autotext.set_color('black')

# ax.axis('equal')

# ax.legend(
#     wedges, labels,
#     title="Proportion Ranges",
#     loc="upper left",
#     bbox_to_anchor=(-0.3, 1),
#     fontsize=12,
#     title_fontsize=13
# )

# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt

values = [38.1, 32.3, 20.9, 8.7]
labels = ['0.75~1.00', '0.50~0.75', '0.25~0.50', '0.00~0.25']

# 四种淡色
my_colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9']
from matplotlib import cm
cmap = cm.get_cmap('YlOrRd')          # 红-黄热度渐变
my_colors = [cmap(v/100) for v in values]
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=False)


wedges, texts, autotexts = ax.pie(
    values,
    colors=my_colors,
    autopct='%1.1f%%',
    pctdistance=0.8,             
    labeldistance=1.15,
    startangle=90,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
centre_circle = plt.Circle((0, 0), 0.55, fc='white')   # 0.55 半径可微调
ax.add_artist(centre_circle)

ax.legend(
    wedges,
    labels,
    title='ED',
    loc='lower right',
    bbox_to_anchor=(1.3, 0.15),   # 更紧凑
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    markerscale=0.6,
    labelspacing=0.3
)

ax.axis('equal')
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.85)
plt.savefig('pie_chart.pdf', format='pdf', bbox_inches='tight')

