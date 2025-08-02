import pandas as pd
import numpy as np
from scipy.stats import shapiro, skew, kurtosis,bartlett,f_oneway


pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.width', None)         # 不限制总宽度
pd.set_option('display.max_colwidth', None)  # 每列最大内容显示
def generate_shapiro_result_table(data, variable_name="碳排放量（万 tCO2）"):
    """
    对数据进行 Shapiro-Wilk 正态性检验并返回结果表格 DataFrame。
    """
    data = np.array(data)
    n = len(data)
    median = np.median(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 样本标准差
    skewness = skew(data)
    kurt = kurtosis(data)  # 默认 Fisher（与 Excel 一致），正态分布理论值为0
    W, p = shapiro(data)

    df = pd.DataFrame({
        "变量名": [variable_name],
        "样本量": [n],
        "中位数": [round(median, 3)],
        "平均值": [round(mean, 3)],
        "标准差": [round(std, 2)],
        "偏度": [round(skewness, 3)],
        "峰度": [round(kurt, 3)],
        "S-W 检验": [f"{round(W, 3)} ({round(p, 3)})"]
    })

    return df

def generate_bartlett_result_table(group1, group2):
    """
    对两个数据组进行 Bartlett 方差齐性检验，并输出结果表格 DataFrame。
    """
    # 计算各组统计
    g1_mean = np.mean(group1)
    g1_std = np.std(group1, ddof=1)
    g2_mean = np.mean(group2)
    g2_std = np.std(group2, ddof=1)

    # 总体合并
    pooled = group1 + group2
    pooled_mean = np.mean(pooled)
    pooled_std = np.std(pooled, ddof=1)

    # Bartlett 检验
    stat, p = bartlett(group1, group2)

    # 构建表格
    df = pd.DataFrame([
        ["1", len(group1), round(g1_mean, 1), round(g1_std, 2), round(stat, 5), 1, round(p, 5)],
        ["2", len(group2), round(g2_mean, 1), round(g2_std, 2), "", "", ""],
        ["池化", len(pooled), round(pooled_mean, 1), round(pooled_std, 2), "", "", ""]
    ], columns=["组", "计数", "均值", "标准差", "Bartlett 统计量", "自由度", "p 值"])

    return df

def anova_analysis_table(group1, group2):
    """
    对两组数据进行单因素方差分析，返回标准化结果表格 DataFrame。
    """

    # 样本数量
    n1, n2 = len(group1), len(group2)
    total_n = n1 + n2

    # 组内总平方和（SSE）
    sse = sum((x - np.mean(group1)) ** 2 for x in group1) + sum((x - np.mean(group2)) ** 2 for x in group2)

    # 总平均值
    grand_mean = np.mean(group1 + group2)

    # 组间平方和（SSA）
    ssa = n1 * (np.mean(group1) - grand_mean) ** 2 + n2 * (np.mean(group2) - grand_mean) ** 2

    # 总平方和（SST）
    sst = ssa + sse

    # 自由度
    df_between = 1
    df_within = total_n - 2
    df_total = total_n - 1

    # 均方
    ms_between = ssa / df_between
    ms_within = sse / df_within

    # F检验值
    F = ms_between / ms_within

    # 使用 scipy 验证 p 值（更稳妥）
    _, p_value = f_oneway(group1, group2)

    # 缩放为“*10^7”表示（如图中）
    def scale(x): return round(x / 1e7, 5)

    # 构建表格
    table = pd.DataFrame([
        ["组间", scale(ssa), df_between, scale(ms_between), round(F, 2), round(p_value, 3)],
        ["组内", scale(sse), df_within, scale(ms_within), "", ""],
        ["合计", scale(sst), df_total, "", "", ""]
    ], columns=["差异来源", "误差平方和（*10⁷）", "自由度", "均方差（*10⁷）", "F 值", "p 值 (Prob>F)"])

    return table


data = [
    65193.342, 67502.613, 66749.376, 64853.276,
    66074.810, 68526.125, 70451.557, 71502.003,
    74096.331, 72633.324
]

# Shapiro-Wilk 检验法进行正态检验
df_result = generate_shapiro_result_table(data)
print(df_result)

# 方差齐次性检验分组：2011–2015 为第一组，2016–2020 为第二组
df_bartlett = generate_bartlett_result_table(data[:5], data[5:])
print(df_bartlett)

# 方差分析
df_anova = anova_analysis_table(data[:5], data[5:])
print(df_anova)





