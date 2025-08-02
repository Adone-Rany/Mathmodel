import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# 设置中文字体为宋体，英文字体为Times New Roman
rcParams['font.sans-serif'] = ['SimSun']  # 宋体
rcParams['font.family'] = 'sans-serif'
rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为STIX（类似于Times New Roman）

# 数据
population = np.array(
    [7869.34, 8022.99, 8119.81, 8192.44, 8281.09, 8315.11, 8381.47, 8423.50, 8446.19, 8469.09, 8477.26])  # 万人
carbon_emission = np.array(
    [56360.052, 65193.342, 67502.613, 66749.376, 64853.276, 66074.810, 68526.125, 70451.557, 71502.003, 74096.331,
     72633.324])  # 万吨


def plot_fitting():
    # 线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(population[1:], carbon_emission[1:])
    line = slope * population[1:] + intercept

    # 绘制散点图和拟合直线
    plt.figure(figsize=(8, 6))
    plt.scatter(population[1:], carbon_emission[1:], color='b', label='数据点')
    plt.plot(population[1:], line, 'r-', label=f'拟合直线: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value ** 2:.2f}')

    # 设置标题和坐标轴标签
    plt.title('人口数量与碳排放量的线性拟合', fontsize=14, fontweight='bold')
    plt.xlabel('人口数量（万人）', fontsize=12)
    plt.ylabel('碳排放量（万吨）', fontsize=12)
    plt.xlim(7900, 8600)
    plt.ylim(64000, 75000)
    # 添加图例
    plt.legend(loc='best', fontsize=10)

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_fitting2():
    # 使用所有数据点（去掉[1:]的限制）
    x = population
    y = carbon_emission

    # 线性拟合
    slope, intercept, r_value_lin, p_value, std_err = stats.linregress(x, y)
    line_lin = slope * x + intercept

    # 二次多项式拟合
    coeff_quad = np.polyfit(x, y, 2)
    poly_quad = np.poly1d(coeff_quad)
    line_quad = poly_quad(x)
    r_value_quad = np.corrcoef(y, line_quad)[0, 1]

    # 绘制图形
    plt.figure(figsize=(10, 7))

    # 绘制数据点
    plt.scatter(x, y, color='b', s=80, label='数据点', zorder=5)

    # 添加数据标签
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, f'({xi:.0f}, {yi:.0f})',
                 ha='center', va='bottom', fontsize=9)

    # 绘制拟合线
    plt.plot(x, line_lin, 'r--', lw=2,
             label=f'线性拟合: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value_lin ** 2:.2f}')
    plt.plot(x, line_quad, 'g-', lw=2,
             label=f'二次拟合: y = {coeff_quad[0]:.2f}x² + {coeff_quad[1]:.2f}x + {coeff_quad[2]:.2f}\n$R^2$ = {r_value_quad ** 2:.2f}')

    # 设置图形属性
    plt.title('人口数量与碳排放量的拟合分析', fontsize=16, pad=20)
    plt.xlabel('人口数量（万人）', fontsize=12)
    plt.ylabel('碳排放量（万吨）', fontsize=12)

    # 设置坐标轴范围
    plt.xlim(7800, 8500)
    plt.ylim(60000, 76000)

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 调整布局
    plt.tight_layout()
    plt.show()




def plot_per_capita_emission():
    # 计算人均碳排放量（吨/人）
    per_capita_emission = carbon_emission * 10000 / (population * 10000)  # 万吨转换为吨，万人转换为人

    # 绘制折线图
    plt.figure(figsize=(8, 6))
    plt.plot(population, per_capita_emission, 'go-', linewidth=2, markersize=8, label='人均碳排放量')

    # 设置标题和坐标轴标签
    plt.title('人口数量与人均碳排放量', fontsize=14, fontweight='bold')
    plt.xlabel('人口数量（万人）', fontsize=12)
    plt.ylabel('人均碳排放量（吨/人）', fontsize=12)

    # 添加数据标签
    for i, (pop, val) in enumerate(zip(population, per_capita_emission)):
        plt.text(pop, val, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 添加图例
    plt.legend(loc='best', fontsize=10)

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# 图4-15，4-16
plot_fitting()
plot_fitting2()
plot_per_capita_emission()