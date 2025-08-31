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

                        # if hours_from_onset >1500:
                        #     print(f"患者 {patient_id} 的第 {i} 次检查时间超过1000小时")

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

# 可选：添加多项式拟合曲线
from scipy import stats

if len(time_points) > 1:
    # 尝试不同阶数的多项式拟合
    degrees = [1, 2, 3, 4]  # 1次到4次多项式
    best_degree = 1
    best_r2 = 0

    plt.figure(figsize=(15, 10))

    # 创建子图比较不同阶数的拟合效果
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    polynomial_results = {}

    for i, degree in enumerate(degrees):
        if len(time_points) > degree:  # 确保数据点数量足够
            # 多项式拟合
            coeffs = np.polyfit(time_points, volume_values, degree)
            poly_func = np.poly1d(coeffs)

            # 计算R²
            y_pred = poly_func(time_points)
            ss_res = np.sum((volume_values - y_pred) ** 2)
            ss_tot = np.sum((volume_values - np.mean(volume_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            polynomial_results[degree] = {
                'coeffs': coeffs,
                'poly_func': poly_func,
                'r2': r2
            }

            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree

            # 绘制子图
            axes[i].scatter(time_points, volume_values, alpha=0.6, s=20, c='steelblue', edgecolors='none')

            # 拟合曲线
            x_smooth = np.linspace(time_points.min(), time_points.max(), 200)
            y_smooth = poly_func(x_smooth)
            axes[i].plot(x_smooth, y_smooth, 'r-', linewidth=2)

            axes[i].set_xlabel('发病至影像检查时间 (小时)')
            axes[i].set_ylabel('水肿体积')
            axes[i].set_title(f'{degree}次多项式拟合 (R²={r2:.4f})')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 绘制最佳拟合结果的大图
    plt.figure(figsize=(12, 8))
    plt.scatter(time_points, volume_values, alpha=0.6, s=30, c='steelblue', edgecolors='none', label='数据点')

    # 使用最佳多项式拟合
    best_poly = polynomial_results[best_degree]['poly_func']
    x_smooth = np.linspace(time_points.min(), time_points.max(), 200)
    y_smooth = best_poly(x_smooth)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=3,
             label=f'{best_degree}次多项式拟合 (R²={best_r2:.4f})')

    plt.xlabel('发病至影像检查时间 (小时)', fontsize=12)
    plt.ylabel('水肿体积 (ED_volume)', fontsize=12)
    plt.title('前100名患者水肿体积随时间进展散点图（多项式拟合）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"\n多项式拟合分析:")
    print(f"最佳拟合阶数: {best_degree}")
    print(f"最佳R²值: {best_r2:.4f}")

    # 打印所有拟合结果
    print(f"\n各阶数拟合结果:")
    for degree in degrees:
        if degree in polynomial_results:
            r2 = polynomial_results[degree]['r2']
            coeffs = polynomial_results[degree]['coeffs']
            print(f"{degree}次多项式: R² = {r2:.4f}")

            # 构建多项式方程字符串
            equation = "y = "
            for j, coeff in enumerate(coeffs):
                power = len(coeffs) - 1 - j
                if j > 0:
                    equation += " + " if coeff >= 0 else " - "
                    equation += f"{abs(coeff):.4f}"
                else:
                    equation += f"{coeff:.4f}"

                if power > 0:
                    equation += f"x^{power}" if power > 1 else "x"
            print(f"   方程: {equation}")

