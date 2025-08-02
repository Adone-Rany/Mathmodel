# 图4-4
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体：新罗马 + 宋体
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 年份：2011–2020
years = list(range(2011, 2021))

# 碳排放量（单位可自行补充）
emissions = [65193.342, 67502.613, 66749.376, 64853.276,
             66074.810, 68526.125, 70451.557, 71502.003, 74096.331, 72633.324]

# 前一年数据（2010年 + 当前 emissions 的前9个）
previous_emissions = [56360.052] + emissions[:-1]

# 年增长率 %
growth_rate = [(emissions[i] - previous_emissions[i]) / previous_emissions[i] * 100 for i in range(len(emissions))]

# 创建双轴图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 折线图（左轴）— 年增长率
ax1.plot(years, growth_rate, color='tab:red', marker='o', label='年增长率')
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('年增长率 (%)', fontsize=12, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(-5, 20)

# 横坐标强制显示所有年份
plt.xticks(years)

# 柱状图（右轴）— 碳排放总量
ax2 = ax1.twinx()
ax2.bar(years, emissions, alpha=0.6, color='tab:blue', label='碳排放总量')
ax2.set_ylabel('碳排放总量 (单位)', fontsize=12, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_ylim(60000, 75000)

# 图例合并
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# 图标题
ax1.set_title('2011-2020年碳排放总量及年增长率', fontsize=14)

plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
