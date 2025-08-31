import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# 读取Excel数据
df = pd.read_excel('数据处理q1_b.xlsx')

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 筛选前100个患者
df_subset = df[df['患者id'].str.extract('(\d+)')[0].astype(int) <= 100].copy()

# 存储所有数据点
time_points = []
volume_values = []
patient_ids = []

# 处理每个患者的数据
for idx, row in df_subset.iterrows():
    patient_id = row['患者id']

    # 获取发病到首次影像检查的时间间隔（小时）
    onset_to_first_interval = row['发病到首次影像检查时间间隔']

    if pd.isna(onset_to_first_interval):
        continue

    # 获取入院首次检查时间点
    first_check_time = row['入院首次检查时间点']

    if pd.isna(first_check_time):
        continue

    try:
        # 解析首次检查时间
        first_check_datetime = pd.to_datetime(first_check_time)

        # 计算发病时间 = 首次检查时间 - 发病到首次检查的间隔
        onset_time = first_check_datetime - timedelta(hours=float(onset_to_first_interval))

        # 处理首次检查（ED_volume0）
        if not pd.isna(row['ED_volume0']):
            hours_from_onset = float(onset_to_first_interval)
            time_points.append(hours_from_onset)
            volume_values.append(row['ED_volume0'])
            patient_ids.append(patient_id)

        # 处理随访检查（ED_volume1 到 ED_volume8）
        for i in range(1, 9):
            volume_col = f'ED_volume{i}'
            time_col = f'随访{i}时间点'

            if volume_col in row and time_col in row:
                if not pd.isna(row[volume_col]) and not pd.isna(row[time_col]):
                    try:
                        # 解析随访时间
                        followup_time = pd.to_datetime(row[time_col])

                        # 计算相对于发病的时间（小时）
                        time_diff = followup_time - onset_time
                        hours_from_onset = time_diff.total_seconds() / 3600

                        time_points.append(hours_from_onset)
                        volume_values.append(row[volume_col])
                        patient_ids.append(patient_id)

                    except:
                        continue

    except Exception as e:
        print(f"处理患者 {patient_id} 时出错: {e}")
        continue

# 转换为numpy数组便于处理
time_points = np.array(time_points)
volume_values = np.array(volume_values)

print(f"总共收集到 {len(time_points)} 个数据点")
print(f"时间范围: {time_points.min():.1f} - {time_points.max():.1f} 小时")
print(f"水肿体积范围: {volume_values.min():.1f} - {volume_values.max():.1f}")

# 创建散点图
plt.figure(figsize=(12, 8))
plt.scatter(time_points, volume_values, alpha=0.6, s=30, c='steelblue', edgecolors='none')

plt.xlabel('发病至影像检查时间 (小时)', fontsize=12)
plt.ylabel('水肿体积 (ED_volume)', fontsize=12)
plt.title('前100名患者水肿体积随时间进展散点图', fontsize=14, fontweight='bold')

# 设置网格
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示统计信息
plt.text(0.02, 0.98, f'数据点数: {len(time_points)}\n患者数: {len(set(patient_ids))}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.show()

# 打印一些统计信息
print(f"\n统计信息:")
print(f"参与分析的患者数量: {len(set(patient_ids))}")
print(f"平均每个患者的检查次数: {len(time_points) / len(set(patient_ids)):.1f}")
print(f"时间点统计 (小时):")
print(f"  最小值: {time_points.min():.1f}")
print(f"  最大值: {time_points.max():.1f}")
print(f"  平均值: {time_points.mean():.1f}")
print(f"  中位数: {np.median(time_points):.1f}")
print(f"水肿体积统计:")
print(f"  最小值: {volume_values.min():.1f}")
print(f"  最大值: {volume_values.max():.1f}")
print(f"  平均值: {volume_values.mean():.1f}")
print(f"  中位数: {np.median(volume_values):.1f}")

# 可选：添加最小二乘法多项式拟合
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

if len(time_points) > 1:
    # 尝试不同阶数的多项式拟合（最小二乘法）
    degrees = [1, 2]  # 1次到4次多项式
    best_degree = 1
    best_r2 = 0

    plt.figure(figsize=(15, 10))

    # 创建子图比较不同阶数的拟合效果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    polynomial_results = {}

    # 准备数据
    X = time_points.reshape(-1, 1)
    y = volume_values

    for i, degree in enumerate(degrees):
        if len(time_points) > degree:  # 确保数据点数量足够
            # 使用sklearn进行最小二乘法多项式拟合
            poly_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])

            # 拟合模型
            poly_pipeline.fit(X, y)

            # 预测
            y_pred = poly_pipeline.predict(X)

            # 计算评估指标
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)

            polynomial_results[degree] = {
                'model': poly_pipeline,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'y_pred': y_pred
            }

            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree

            # 绘制子图
            axes[i].scatter(time_points, volume_values, alpha=0.6, s=20, c='steelblue', edgecolors='none')

            # 拟合曲线
            x_smooth = np.linspace(time_points.min(), time_points.max(), 200)
            X_smooth = x_smooth.reshape(-1, 1)
            y_smooth = poly_pipeline.predict(X_smooth)
            axes[i].plot(x_smooth, y_smooth, 'r-', linewidth=2)

            axes[i].set_xlabel('发病至影像检查时间 (小时)')
            axes[i].set_ylabel('水肿体积')
            axes[i].set_title(f'{degree}次多项式拟合 (R²={r2:.4f}, RMSE={rmse:.2f})')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 绘制最佳拟合结果的大图
    plt.figure(figsize=(12, 8))
    plt.scatter(time_points, volume_values, alpha=0.6, s=30, c='steelblue', edgecolors='none', label='数据点')

    # 使用最佳多项式拟合
    best_model = polynomial_results[best_degree]['model']
    x_smooth = np.linspace(time_points.min(), time_points.max(), 200)
    X_smooth = x_smooth.reshape(-1, 1)
    y_smooth = best_model.predict(X_smooth)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=3,
             label=f'{best_degree}次多项式拟合 (R²={best_r2:.4f})')

    # 计算并显示置信区间（可选）
    residuals = volume_values - polynomial_results[best_degree]['y_pred']
    std_residual = np.std(residuals)
    y_upper = y_smooth + 1.96 * std_residual
    y_lower = y_smooth - 1.96 * std_residual
    plt.fill_between(x_smooth, y_lower, y_upper, alpha=0.2, color='red',
                     label='95%置信区间')

    plt.xlabel('发病至影像检查时间 (小时)', fontsize=12)
    plt.ylabel('水肿体积 (ED_volume)', fontsize=12)
    plt.title('前100名患者水肿体积随时间进展散点图（最小二乘法多项式拟合）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"\n最小二乘法多项式拟合分析:")
    print(f"最佳拟合阶数: {best_degree}")
    print(f"最佳R²值: {best_r2:.4f}")

    # 打印所有拟合结果
    print(f"\n各阶数拟合结果 (最小二乘法):")
    for degree in degrees:
        if degree in polynomial_results:
            result = polynomial_results[degree]
            print(f"{degree}次多项式:")
            print(f"  R² = {result['r2']:.4f}")
            print(f"  MSE = {result['mse']:.4f}")
            print(f"  RMSE = {result['rmse']:.4f}")

            # 获取多项式系数并构建方程
            poly_features = PolynomialFeatures(degree=degree)
            linear_reg = result['model'].named_steps['linear']
            coefficients = linear_reg.coef_
            intercept = linear_reg.intercept_

            # 构建多项式方程字符串
            equation = "y = "
            if degree == 1:
                equation += f"{coefficients[1]:.4f}x + {intercept:.4f}"
            else:
                # 对于高次多项式，系数顺序是 [intercept, x, x², x³, ...]
                terms = []
                for j in range(len(coefficients)):
                    coef = coefficients[j]
                    if j == 0:  # 常数项已经在intercept中
                        continue
                    elif j == 1:  # x项
                        if abs(coef) > 1e-10:  # 避免显示很小的系数
                            terms.append(f"{coef:.4f}x")
                    else:  # x^n项
                        if abs(coef) > 1e-10:
                            terms.append(f"{coef:.4f}x^{j}")

                if intercept != 0:
                    terms.append(f"{intercept:.4f}")

                equation += " + ".join(terms).replace("+ -", "- ")

            print(f"  方程: {equation}")
            print()

    # 残差分析
    best_residuals = volume_values - polynomial_results[best_degree]['y_pred']

    # 绘制残差图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(time_points, best_residuals, alpha=0.6, s=30, c='green')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('发病至影像检查时间 (小时)')
    plt.ylabel('残差')
    plt.title(f'{best_degree}次多项式拟合残差图')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(best_residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('残差值')
    plt.ylabel('频数')
    plt.title('残差分布直方图')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"残差统计:")
    print(f"残差均值: {np.mean(best_residuals):.4f}")
    print(f"残差标准差: {np.std(best_residuals):.4f}")
    print(f"残差范围: [{np.min(best_residuals):.4f}, {np.max(best_residuals):.4f}]")