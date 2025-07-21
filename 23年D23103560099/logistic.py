import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 更新后的历史数据（单位：万人）
years = np.array([2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
population = np.array([7869.34,8022.99,8119.81,8192.44,8281.09,8315.11,8381.47,8423.50,8446.19,8469.09,8477.26]) 

# 转换为亿人单位（根据需求可选）
# population = population / 10000  

# Logistic函数定义
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# 参数初始猜测（关键调整）
# K: 最大承载人口（建议取历史最大值的1.05-1.2倍）
# r: 增长率（根据数据调整，中国近年约0.3%-0.5%）
# t0: 拐点年份（观察数据在2018-2020增长放缓）
initial_guess = [8549.75, 0.0024, 2018]  # 示例初始值

# 拟合模型
popt, pcov = curve_fit(logistic, years, population, p0=initial_guess, maxfev=5000)
K, r, t0 = popt  # 最优参数
print(f"拟合参数：K={K:.2f}万人, r={r:.4f}, t0={t0:.1f}")

# 生成预测年份（2010-2060年）
future_years = np.arange(2010, 2061)
predicted_pop = logistic(future_years, K, r, t0)

# 关键年份预测
target_years = [2025, 2030, 2040, 2050, 2060]
for year in target_years:
    print(f"{year}年预测人口: {logistic(year, K, r, t0):.2f}万人")

# 可视化
plt.figure(figsize=(12, 6))
plt.scatter(years, population, label='实际数据', color='blue')
plt.plot(future_years, predicted_pop, 'r-', label='Logistic预测')
plt.axvline(x=t0, linestyle='--', color='gray', label=f'拐点年份: {t0:.1f}')
plt.axhline(y=K, linestyle=':', color='green', label=f'人口上限: {K:.2f}万人')

# 标记关键年份
for year in target_years:
    plt.axvline(x=year, linestyle=':', color='orange', alpha=0.3)
    plt.text(year, predicted_pop[year-2010], f'{predicted_pop[year-2010]:.1f}', 
             ha='center', va='bottom')

plt.title('中国人口预测(2010-2060)', fontsize=14)
plt.xlabel('年份', fontsize=12)
plt.ylabel('人口（万人）', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()