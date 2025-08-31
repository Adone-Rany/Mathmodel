# 读取五个聚类患者表格，画出拟合曲线
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_patient_time_volume_data(df_cluster):
    """
    从聚类数据中提取所有时间点和体积数据
    """
    time_points = []
    volume_values = []
    patient_ids = []

    for idx, row in df_cluster.iterrows():
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

    return np.array(time_points), np.array(volume_values), patient_ids


def polynomial_fitting_analysis(time_points, volume_values, cluster_id):
    """
    对时间和体积数据进行多项式拟合分析
    """
    if len(time_points) <= 2:
        print(f"聚类 {cluster_id} 数据点太少，无法进行拟合分析")
        return None

    # 尝试不同阶数的多项式拟合
    degrees = [1, 2]
    best_degree = 1
    best_r2 = -np.inf

    X = time_points.reshape(-1, 1)
    y = volume_values

    polynomial_results = {}

    for degree in degrees:
        if len(time_points) > degree:
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

    return polynomial_results, best_degree


# 读取五个聚类的数据文件
cluster_data = {}
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

print("正在读取五个聚类数据文件...")

for cluster_id in range(5):
    filename = f'聚类{cluster_id}_患者数据.xlsx'
    try:
        df_cluster = pd.read_excel(filename)

        # 提取时间-体积数据
        time_points, volume_values, patient_ids = extract_patient_time_volume_data(df_cluster)

        cluster_data[cluster_id] = {
            'df': df_cluster,
            'time_points': time_points,
            'volume_values': volume_values,
            'patient_ids': patient_ids,
            'unique_patients': len(set(patient_ids))
        }

        print(f"聚类 {cluster_id}: {len(df_cluster)} 名患者, {len(time_points)} 个数据点")

    except FileNotFoundError:
        print(f"警告: 找不到文件 {filename}")
        continue
    except Exception as e:
        print(f"读取聚类 {cluster_id} 数据时出错: {e}")
        continue

if not cluster_data:
    print("错误: 无法读取任何聚类数据文件，请确保文件存在")
    exit()

# 1. 综合散点图：所有聚类在一张图上
plt.figure(figsize=(14, 10))

for cluster_id, data in cluster_data.items():
    plt.scatter(data['time_points'], data['volume_values'],
                alpha=0.7, s=40, c=cluster_colors[cluster_id],
                label=f'聚类 {cluster_id} ({data["unique_patients"]}患者)',
                edgecolors='white', linewidth=0.5)

plt.xlabel('发病至影像检查时间 (小时)', fontsize=12)
plt.ylabel('水肿体积 (ED_volume)', fontsize=12)
plt.title('五个聚类患者水肿体积随时间进展综合散点图', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. 为每个聚类分别进行拟合分析和可视化
fig, axes = plt.subplots(5, 3, figsize=(18, 20))

cluster_fitting_results = {}

for cluster_id, data in cluster_data.items():
    time_points = data['time_points']
    volume_values = data['volume_values']

    print(f"\n=== 聚类 {cluster_id} 分析 ===")
    print(f"患者数量: {data['unique_patients']}")
    print(f"数据点数量: {len(time_points)}")
    print(f"时间范围: {time_points.min():.1f} - {time_points.max():.1f} 小时")
    print(f"体积范围: {volume_values.min():.1f} - {volume_values.max():.1f}")

    # 进行多项式拟合分析
    fitting_results, best_degree = polynomial_fitting_analysis(time_points, volume_values, cluster_id)

    if fitting_results is None:
        continue

    cluster_fitting_results[cluster_id] = {
        'fitting_results': fitting_results,
        'best_degree': best_degree,
        'time_points': time_points,
        'volume_values': volume_values
    }

    # 绘制散点图和拟合曲线
    axes[cluster_id, 0].scatter(time_points, volume_values,
                                alpha=0.7, s=30, c=cluster_colors[cluster_id],
                                edgecolors='white', linewidth=0.5)

    # 绘制最佳拟合曲线
    if len(time_points) > 2:
        best_model = fitting_results[best_degree]['model']
        x_smooth = np.linspace(time_points.min(), time_points.max(), 100)
        X_smooth = x_smooth.reshape(-1, 1)
        y_smooth = best_model.predict(X_smooth)

        axes[cluster_id, 0].plot(x_smooth, y_smooth, 'r-', linewidth=2,
                                 label=f'{best_degree}次拟合 (R²={fitting_results[best_degree]["r2"]:.3f})')

        # 添加置信区间
        residuals = volume_values - fitting_results[best_degree]['y_pred']
        std_residual = np.std(residuals)
        y_upper = y_smooth + 1.96 * std_residual
        y_lower = y_smooth - 1.96 * std_residual
        axes[cluster_id, 0].fill_between(x_smooth, y_lower, y_upper, alpha=0.2, color='red')

    axes[cluster_id, 0].set_xlabel('发病至影像检查时间 (小时)')
    axes[cluster_id, 0].set_ylabel('水肿体积')
    axes[cluster_id, 0].set_title(f'聚类 {cluster_id} - 散点图与拟合曲线')
    axes[cluster_id, 0].legend()
    axes[cluster_id, 0].grid(True, alpha=0.3)

    # 绘制残差散点图
    if len(time_points) > 2:
        best_residuals = volume_values - fitting_results[best_degree]['y_pred']
        axes[cluster_id, 1].scatter(time_points, best_residuals,
                                    alpha=0.7, s=30, c='green', edgecolors='white', linewidth=0.5)
        axes[cluster_id, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[cluster_id, 1].set_xlabel('发病至影像检查时间 (小时)')
        axes[cluster_id, 1].set_ylabel('残差')
        axes[cluster_id, 1].set_title(f'聚类 {cluster_id} - 残差散点图')
        axes[cluster_id, 1].grid(True, alpha=0.3)

        # 绘制残差直方图
        axes[cluster_id, 2].hist(best_residuals, bins=15, alpha=0.7,
                                 color='green', edgecolor='black')
        axes[cluster_id, 2].set_xlabel('残差值')
        axes[cluster_id, 2].set_ylabel('频数')
        axes[cluster_id, 2].set_title(f'聚类 {cluster_id} - 残差分布')
        axes[cluster_id, 2].grid(True, alpha=0.3)

        # 打印拟合结果
        print(f"最佳拟合: {best_degree}次多项式")
        print(f"R² = {fitting_results[best_degree]['r2']:.4f}")
        print(f"RMSE = {fitting_results[best_degree]['rmse']:.4f}")
        print(f"残差统计: 均值={np.mean(best_residuals):.4f}, 标准差={np.std(best_residuals):.4f}")
    else:
        # 如果数据点太少，显示提示信息
        for col in [1, 2]:
            axes[cluster_id, col].text(0.5, 0.5, '数据点不足\n无法拟合',
                                       ha='center', va='center',
                                       transform=axes[cluster_id, col].transAxes,
                                       fontsize=12)
            axes[cluster_id, col].set_title(f'聚类 {cluster_id} - 数据不足')

plt.tight_layout()
plt.show()

# 3. 绘制所有聚类的拟合曲线对比图
plt.figure(figsize=(16, 10))

for cluster_id, cluster_results in cluster_fitting_results.items():
    time_points = cluster_results['time_points']
    volume_values = cluster_results['volume_values']
    fitting_results = cluster_results['fitting_results']
    best_degree = cluster_results['best_degree']

    # 绘制散点图
    plt.scatter(time_points, volume_values,
                alpha=0.6, s=30, c=cluster_colors[cluster_id],
                label=f'聚类 {cluster_id} 数据点')

    # 绘制拟合曲线
    if len(time_points) > 2:
        best_model = fitting_results[best_degree]['model']
        x_smooth = np.linspace(time_points.min(), time_points.max(), 100)
        X_smooth = x_smooth.reshape(-1, 1)
        y_smooth = best_model.predict(X_smooth)

        plt.plot(x_smooth, y_smooth, color=cluster_colors[cluster_id],
                 linewidth=3, linestyle='-',
                 label=f'聚类 {cluster_id} 拟合 (R²={fitting_results[best_degree]["r2"]:.3f})')

plt.xlabel('发病至影像检查时间 (小时)', fontsize=12)
plt.ylabel('水肿体积 (ED_volume)', fontsize=12)
plt.title('五个聚类拟合曲线对比图', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. 创建拟合结果汇总表
summary_data = []

for cluster_id, cluster_results in cluster_fitting_results.items():
    fitting_results = cluster_results['fitting_results']
    best_degree = cluster_results['best_degree']
    time_points = cluster_results['time_points']
    volume_values = cluster_results['volume_values']

    if fitting_results:
        best_result = fitting_results[best_degree]
        residuals = volume_values - best_result['y_pred']

        # 获取拟合方程系数
        linear_reg = best_result['model'].named_steps['linear']
        coefficients = linear_reg.coef_
        intercept = linear_reg.intercept_

        # 构建方程字符串
        if best_degree == 1:
            equation = f"y = {coefficients[1]:.4f}x + {intercept:.4f}"
        else:
            terms = []
            for j in range(1, len(coefficients)):
                coef = coefficients[j]
                if abs(coef) > 1e-10:
                    if j == 1:
                        terms.append(f"{coef:.4f}x")
                    else:
                        terms.append(f"{coef:.4f}x^{j}")
            if intercept != 0:
                terms.append(f"{intercept:.4f}")
            equation = "y = " + " + ".join(terms).replace("+ -", "- ")

        summary_info = {
            '聚类': cluster_id,
            '患者数': cluster_data[cluster_id]['unique_patients'],
            '数据点数': len(time_points),
            '最佳拟合阶数': best_degree,
            'R²': f"{best_result['r2']:.4f}",
            'RMSE': f"{best_result['rmse']:.4f}",
            '时间范围(小时)': f"[{time_points.min():.1f}, {time_points.max():.1f}]",
            '体积范围': f"[{volume_values.min():.1f}, {volume_values.max():.1f}]",
            '残差均值': f"{np.mean(residuals):.4f}",
            '残差标准差': f"{np.std(residuals):.4f}",
            '拟合方程': equation
        }
        summary_data.append(summary_info)

# 创建汇总DataFrame并保存
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel('五个聚类拟合分析汇总.xlsx', index=False)

    print(f"\n=== 五个聚类拟合分析汇总 ===")
    print(summary_df.to_string(index=False))
    print(f"\n汇总结果已保存到: 五个聚类拟合分析汇总.xlsx")

# 5. 绘制每个聚类的详细分析图（单独显示）
for cluster_id, cluster_results in cluster_fitting_results.items():
    time_points = cluster_results['time_points']
    volume_values = cluster_results['volume_values']
    fitting_results = cluster_results['fitting_results']
    best_degree = cluster_results['best_degree']

    if len(time_points) <= 2:
        continue

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 散点图和拟合曲线
    axes[0].scatter(time_points, volume_values,
                    alpha=0.7, s=50, c=cluster_colors[cluster_id],
                    edgecolors='white', linewidth=1)

    best_model = fitting_results[best_degree]['model']
    x_smooth = np.linspace(time_points.min(), time_points.max(), 100)
    X_smooth = x_smooth.reshape(-1, 1)
    y_smooth = best_model.predict(X_smooth)

    axes[0].plot(x_smooth, y_smooth, 'r-', linewidth=3,
                 label=f'{best_degree}次拟合 (R²={fitting_results[best_degree]["r2"]:.4f})')

    # 置信区间
    residuals = volume_values - fitting_results[best_degree]['y_pred']
    std_residual = np.std(residuals)
    y_upper = y_smooth + 1.96 * std_residual
    y_lower = y_smooth - 1.96 * std_residual
    axes[0].fill_between(x_smooth, y_lower, y_upper, alpha=0.2, color='red', label='95%置信区间')

    axes[0].set_xlabel('发病至影像检查时间 (小时)')
    axes[0].set_ylabel('水肿体积')
    axes[0].set_title(f'聚类 {cluster_id} - 拟合曲线图')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 残差散点图
    axes[1].scatter(time_points, residuals,
                    alpha=0.7, s=50, c='green', edgecolors='white', linewidth=1)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
    axes[1].set_xlabel('发病至影像检查时间 (小时)')
    axes[1].set_ylabel('残差')
    axes[1].set_title(f'聚类 {cluster_id} - 残差散点图')
    axes[1].grid(True, alpha=0.3)

    # 残差直方图
    axes[2].hist(residuals, bins=max(5, len(residuals) // 3), alpha=0.7,
                 color='green', edgecolor='black')
    axes[2].set_xlabel('残差值')
    axes[2].set_ylabel('频数')
    axes[2].set_title(f'聚类 {cluster_id} - 残差分布')
    axes[2].grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'均值: {np.mean(residuals):.3f}\n标准差: {np.std(residuals):.3f}\n范围: [{np.min(residuals):.3f}, {np.max(residuals):.3f}]'
    axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'聚类 {cluster_id} 详细分析 ({data["unique_patients"]} 名患者, {len(time_points)} 个数据点)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 6. 创建聚类对比分析
if len(cluster_fitting_results) > 1:
    print(f"\n=== 聚类间拟合效果对比 ===")

    comparison_data = []
    for cluster_id, results in cluster_fitting_results.items():
        best_result = results['fitting_results'][results['best_degree']]
        comparison_data.append({
            '聚类': cluster_id,
            'R²': best_result['r2'],
            'RMSE': best_result['rmse'],
            '拟合阶数': results['best_degree'],
            '数据点数': len(results['time_points'])
        })

    comparison_df = pd.DataFrame(comparison_data)

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R²对比
    axes[0].bar(comparison_df['聚类'], comparison_df['R²'],
                color=cluster_colors[:len(comparison_df)], alpha=0.7)
    axes[0].set_xlabel('聚类')
    axes[0].set_ylabel('R² 值')
    axes[0].set_title('各聚类拟合效果对比 (R²)')
    axes[0].grid(True, alpha=0.3)

    # 在柱状图上标注数值
    for i, (cluster_id, r2) in enumerate(zip(comparison_df['聚类'], comparison_df['R²'])):
        axes[0].text(cluster_id, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom')

    # RMSE对比
    axes[1].bar(comparison_df['聚类'], comparison_df['RMSE'],
                color=cluster_colors[:len(comparison_df)], alpha=0.7)
    axes[1].set_xlabel('聚类')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('各聚类拟合误差对比 (RMSE)')
    axes[1].grid(True, alpha=0.3)

    # 在柱状图上标注数值
    for i, (cluster_id, rmse) in enumerate(zip(comparison_df['聚类'], comparison_df['RMSE'])):
        axes[1].text(cluster_id, rmse + comparison_df['RMSE'].max() * 0.02,
                     f'{rmse:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print(comparison_df.to_string(index=False))

    # 保存对比结果
    comparison_df.to_excel('聚类拟合效果对比.xlsx', index=False)
    print(f"\n聚类拟合效果对比已保存到: 聚类拟合效果对比.xlsx")

print(f"\n=== 分析完成 ===")
print(f"生成的文件:")
print(f"1. 五个聚类拟合分析汇总.xlsx - 详细拟合结果汇总")
print(f"2. 聚类拟合效果对比.xlsx - 聚类间拟合效果对比")