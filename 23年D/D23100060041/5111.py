import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimSun']
rcParams['font.family'] = 'sans-serif'

# 数据准备
years = np.arange(2010, 2021)
population = np.array([7869.34, 8022.99, 8119.81, 8192.44, 8281.09, 8315.11,
                       8381.47, 8423.50, 8446.19, 8469.09, 8477.26])  # 万人
energy = np.array([23539.31, 26860.03, 27999.22, 28203.10, 28170.51, 29033.61,
                   29947.98, 30669.89, 31373.13, 32227.51, 31438.00])  # 万吨标准煤


def plot_linear_regression():
    # 数据预处理（转换为二维数组）
    X = population.reshape(-1, 1)  # 自变量（人口）
    y = energy  # 因变量（能源消费量）

    # 最小二乘法线性回归
    model = LinearRegression()
    model.fit(X, y)

    # 获取回归参数
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    # 生成预测值
    y_pred = model.predict(X)

    # 绘制真实值与预测值对比图
    plt.figure(figsize=(10, 6))

    # 真实值折线（蓝色实线）
    plt.plot(years, y, 'bo-', linewidth=2, markersize=8, label='真实值')

    # 预测值折线（红色虚线）
    plt.plot(years, y_pred, 'r--', linewidth=2, label=f'预测值 (y = {slope:.2f}x + {intercept:.2f})')

    # 添加连接线（灰色虚线）
    for yr, true, pred in zip(years, y, y_pred):
        plt.plot([yr, yr], [true, pred], 'gray', linestyle=':', alpha=0.5)

    # 设置图形属性
    plt.title('人口数量与能源消费量的线性回归拟合（2010-2020）', fontsize=14, pad=20)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('能源消费量（万吨标准煤）', fontsize=12)
    plt.xticks(years, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示回归方程和R²
    plt.legend(loc='upper left', fontsize=10)
    plt.text(0.02, 0.95, f'$R^2 = {r_squared:.3f}$', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()

    # 打印回归结果
    print(f"回归方程: 能源消费量 = {slope:.2f} × 人口 + {intercept:.2f}")
    print(f"R²值: {r_squared:.3f}")

    return model


# ================== 人口与年份的回归拟合 ==================
def population_year_regression():
    X = years.reshape(-1, 1)
    y = population

    # 线性回归
    model = LinearRegression()
    model.fit(X, y)

    # 获取参数
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    y_pred = model.predict(X)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(years, y, 'bo-', label='真实人口', markersize=8)
    plt.plot(years, y_pred, 'r--', label=f'拟合曲线: y = {slope:.2f}x + {intercept:.2f}')

    # 图表美化
    plt.title('人口数量与年份的线性回归拟合（2010-2020）', fontsize=14)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('人口数量（万人）', fontsize=12)
    plt.xticks(years, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.text(0.02, 0.95, f'$R^2 = {r_squared:.3f}$', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()
    print(f"回归方程: 人口 = {slope:.2f} × 年份 + {intercept:.2f}")
    return model


def future_prediction(pop_model, energy_model):
    # 预测2021-2060年
    years_future = np.arange(2021, 2061)
    X_future = years_future.reshape(-1, 1)

    # 预测人口
    pop_future = pop_model.predict(X_future)

    # 预测能源消费量
    energy_future = energy_model.predict(pop_future.reshape(-1, 1))

    # 创建结果表格
    results = pd.DataFrame({
        '年份': years_future,
        '预测人口(万人)': np.round(pop_future, 2),
        '预测能源消费量(万吨标准煤)': np.round(energy_future, 2)
    })
    return results

# 执行函数
print("\n=== 能源-人口回归模型 ===")
energy_model = plot_linear_regression()
print("=== 人口-年份回归分析 ===")
pop_model = population_year_regression()
print("\n=== 2021-2060年预测结果 ===")
future_results = future_prediction(pop_model, energy_model)
print("每10年预测数据：")
print(future_results.iloc[::])
