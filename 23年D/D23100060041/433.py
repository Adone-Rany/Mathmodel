import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimSun']
rcParams['font.family'] = 'sans-serif'

# 数据准备
data = {
    '年份': range(2010, 2021),
    '人口(万人)': [7869.34, 8022.99, 8119.81, 8192.44, 8281.09, 8315.11, 8381.47, 8423.50, 8446.19, 8469.09, 8477.26],
    '碳排放量(万吨)': [56360.052, 65193.342, 67502.613, 66749.376, 64853.276, 66074.810, 68526.125, 70451.557,
                       71502.003, 74096.331, 72633.324],
    'GDP(亿元)': [41383.87, 45952.65, 50660.20, 55580.11, 60359.43, 65552.00, 70665.71, 75752.20, 80827.71, 85556.13,
                  88683.21],
    '能源消费量(万吨标准煤)': [23539.31, 26860.03, 27999.22, 28203.10, 28170.51, 29033.61, 29947.98, 30669.89, 31373.13,
                               32227.51, 31438.00]
}

df = pd.DataFrame(data)


def calculate_spearman_correlation():
    # 按指定顺序选择列
    column_order = ['碳排放量(万吨)', 'GDP(亿元)', '人口(万人)', '能源消费量(万吨标准煤)']
    df_analysis = df[column_order]

    # 计算Spearman相关系数
    corr_matrix = df_analysis.corr(method='spearman')

    # 输出相关系数表（保留4位小数）
    print("Spearman相关系数表（按指定顺序展示）:")
    print(corr_matrix.round(3))

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt=".3f",  # 显示3位小数
                cmap='coolwarm',
                center=0,
                linewidths=0.5,
                annot_kws={"size": 12},
                vmin=-1, vmax=1)  # 固定颜色范围
    plt.title('各因素与碳排放量的Spearman相关性分析', fontsize=14, pad=20)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()


# 执行函数
calculate_spearman_correlation()