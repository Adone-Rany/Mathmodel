import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimSun']
rcParams['font.family'] = 'sans-serif'
rcParams['mathtext.fontset'] = 'stix'

# 数据
years = np.arange(2010, 2021)
gdp = np.array([41383.87, 45952.65, 50660.20, 55580.11, 60359.43, 65552.00,
                70665.71, 75752.20, 80827.71, 85556.13, 88683.21])  # 亿元


def calculate_growth_rates():
    # 转换为DataFrame便于计算
    df = pd.DataFrame({'Year': years, 'GDP': gdp})

    # 计算环比增长率（当前年GDP / 前一年GDP - 1）* 100%
    df['环比增长率(%)'] = (df['GDP'].pct_change() * 100).round(2)

    # 初始化同比增长率为NaN
    df['同比增长率(%)'] = np.nan

    # 计算五年规划的同比（当前年 vs 5年前）
    for i in range(6, len(df)):
        df.loc[i, '同比增长率(%)'] = ((df.loc[i, 'GDP'] / df.loc[i - 5, 'GDP'] - 1) * 100).round(2)

    return df

df_growth = calculate_growth_rates()
print("GDP增长数据：\n", df_growth)


def plot_chain_growth_and_gdp():
    plt.figure(figsize=(10, 6))

    # 左纵坐标：环比折线图
    ax1 = plt.gca()
    ax1.plot(df_growth['Year'][1:], df_growth['环比增长率(%)'][1:],
             'go-', linewidth=2, markersize=8, label='环比增长率(%)')
    ax1.set_xlabel('年份', fontsize=12)
    ax1.set_ylabel('环比增长率(%)', color='g', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylim(0, 15)  # 调整范围使曲线更清晰

    # 右纵坐标：GDP柱状图
    ax2 = ax1.twinx()
    ax2.bar(df_growth['Year'][1:], df_growth['GDP'][1:],
            color='skyblue', alpha=0.6, label='GDP（万亿元）')
    ax2.set_ylabel('GDP（万亿元）', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(40000, 90000)  # 调整范围

    # 标题和图例
    plt.title('2011-2020年GDP环比增长率与总值', fontsize=14, pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 图4-19
plot_chain_growth_and_gdp()






