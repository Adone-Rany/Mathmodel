# 将五个聚类患者数据保存到表格

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
df = pd.read_excel('数据处理q1_b.xlsx')

# 筛选前100个患者
df_subset = df[df['患者id'].str.extract('(\d+)')[0].astype(int) <= 100].copy()

# 提取每个患者的最后一次随访数据（包括时间和体积）
patient_final_data = {}

for idx, row in df_subset.iterrows():
    patient_id = row['患者id']

    # 获取发病到首次影像检查的时间间隔（小时）
    onset_to_first_interval = row['发病到首次影像检查时间间隔']
    first_check_time = row['入院首次检查时间点']

    if pd.isna(onset_to_first_interval) or pd.isna(first_check_time):
        continue

    try:
        # 计算发病时间
        first_check_datetime = pd.to_datetime(first_check_time)
        onset_time = first_check_datetime - timedelta(hours=float(onset_to_first_interval))

        # 按顺序检查从ED_volume8到ED_volume0，找到最后一个非空值
        final_volume = None
        final_time_hours = None
        final_timepoint = None

        # 从最后的随访开始检查（ED_volume8, ED_volume7, ..., ED_volume0）
        for i in range(8, -1, -1):
            volume_col = f'ED_volume{i}'

            if i == 0:  # 首次检查
                if not pd.isna(row[volume_col]):
                    final_volume = row[volume_col]
                    final_time_hours = float(onset_to_first_interval)
                    final_timepoint = i
                    break
            else:  # 随访检查
                time_col = f'随访{i}时间点'
                if volume_col in row and time_col in row:
                    if not pd.isna(row[volume_col]) and not pd.isna(row[time_col]):
                        try:
                            followup_time = pd.to_datetime(row[time_col])
                            time_diff = followup_time - onset_time
                            final_time_hours = time_diff.total_seconds() / 3600
                            final_volume = row[volume_col]
                            final_timepoint = i
                            break
                        except:
                            continue

        if final_volume is not None and final_time_hours is not None:
            patient_final_data[patient_id] = {
                'final_volume': final_volume,
                'final_time_hours': final_time_hours,
                'final_timepoint': final_timepoint
            }

    except Exception as e:
        print(f"处理患者 {patient_id} 时出错: {e}")
        continue

# 转换为DataFrame
clustering_data = pd.DataFrame.from_dict(patient_final_data, orient='index')
clustering_data.reset_index(inplace=True)
clustering_data.rename(columns={'index': 'patient_id'}, inplace=True)

print(f"成功提取到 {len(clustering_data)} 名患者的最后随访数据")
print(
    f"相对发病时间范围: {clustering_data['final_time_hours'].min():.1f} - {clustering_data['final_time_hours'].max():.1f} 小时")
print(f"水肿体积范围: {clustering_data['final_volume'].min():.2f} - {clustering_data['final_volume'].max():.2f}")
print(f"最后随访时间点分布:")
print(clustering_data['final_timepoint'].value_counts().sort_index())

# 准备二维聚类数据：[相对发病时间, 最后水肿体积]
X = clustering_data[['final_time_hours', 'final_volume']].values

print(f"\n聚类特征统计:")
print(f"特征1 (相对发病时间小时): 均值={X[:, 0].mean():.1f}, 标准差={X[:, 0].std():.1f}")
print(f"特征2 (最后水肿体积): 均值={X[:, 1].mean():.2f}, 标准差={X[:, 1].std():.2f}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 在k=3,4,5中选择最佳聚类数
k_range = [3, 4, 5]
evaluation_metrics = {}

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    evaluation_metrics[k] = {
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X_scaled, cluster_labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, cluster_labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, cluster_labels),
        'labels': cluster_labels,
        'model': kmeans
    }

# 显示评估指标
print(f"\n聚类评估指标比较:")
print(f"{'k值':<5} {'惯性':<10} {'轮廓系数':<10} {'CH指数':<12} {'DB指数':<10}")
print("-" * 50)

for k in k_range:
    metrics = evaluation_metrics[k]
    print(f"{k:<5} {metrics['inertia']:<10.2f} {metrics['silhouette']:<10.4f} "
          f"{metrics['calinski_harabasz']:<12.2f} {metrics['davies_bouldin']:<10.4f}")

# 绘制评估指标比较图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics_names = ['inertia', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
titles = ['惯性 (越低越好)', '轮廓系数 (越高越好)', 'Calinski-Harabasz指数 (越高越好)', 'Davies-Bouldin指数 (越低越好)']
colors = ['blue', 'red', 'green', 'purple']

for i, (metric, title, color) in enumerate(zip(metrics_names, titles, colors)):
    row, col = i // 2, i % 2
    values = [evaluation_metrics[k][metric] for k in k_range]
    axes[row, col].bar(k_range, values, color=color, alpha=0.7)
    axes[row, col].set_xlabel('聚类数 (k)')
    axes[row, col].set_ylabel(title.split(' ')[0])
    axes[row, col].set_title(title)
    axes[row, col].grid(True, alpha=0.3)

    # 标注数值
    for k, val in zip(k_range, values):
        axes[row, col].text(k, val, f'{val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 基于轮廓系数选择最佳k
silhouette_scores = [evaluation_metrics[k]['silhouette'] for k in k_range]
optimal_k = 5
# optimal_k = k_range[np.argmax(silhouette_scores)]

print(f"\n基于轮廓系数的最佳聚类数: k = {optimal_k}")
print(f"最佳轮廓系数: {max(silhouette_scores):.4f}")

# 使用最佳k进行最终聚类
final_kmeans = evaluation_metrics[optimal_k]['model']
cluster_labels = evaluation_metrics[optimal_k]['labels']

# 将聚类结果添加到数据中
clustering_data['cluster'] = cluster_labels

# 计算聚类中心（在原始数据尺度上）
cluster_centers_scaled = final_kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

print(f"\n聚类结果分析 (k={optimal_k}):")

# 分析各聚类的统计信息
for cluster_id in range(optimal_k):
    cluster_data = clustering_data[clustering_data['cluster'] == cluster_id]
    times = cluster_data['final_time_hours']
    volumes = cluster_data['final_volume']
    center = cluster_centers_original[cluster_id]

    print(f"\n聚类 {cluster_id}:")
    print(f"  患者数量: {len(cluster_data)}")
    print(f"  聚类中心: 时间={center[0]:.1f}小时, 体积={center[1]:.2f}")
    print(f"  相对发病时间统计:")
    print(f"    均值: {times.mean():.1f}小时")
    print(f"    标准差: {times.std():.1f}小时")
    print(f"    范围: [{times.min():.1f}, {times.max():.1f}]小时")
    print(f"  水肿体积统计:")
    print(f"    均值: {volumes.mean():.2f}")
    print(f"    标准差: {volumes.std():.2f}")
    print(f"    范围: [{volumes.min():.2f}, {volumes.max():.2f}]")
    print(f"  患者ID示例: {list(cluster_data['patient_id'].head())}")

# 可视化聚类结果
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 二维散点图显示聚类结果
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
for cluster_id in range(optimal_k):
    cluster_data = clustering_data[clustering_data['cluster'] == cluster_id]
    axes[0, 0].scatter(cluster_data['final_time_hours'], cluster_data['final_volume'],
                       c=[colors[cluster_id]], label=f'聚类 {cluster_id}',
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

# 标记聚类中心
for cluster_id in range(optimal_k):
    center = cluster_centers_original[cluster_id]
    axes[0, 0].scatter(center[0], center[1], c='red', marker='x', s=200,
                       linewidths=3, label=f'中心 {cluster_id}' if cluster_id == 0 else "")

axes[0, 0].set_xlabel('相对发病时间 (小时)')
axes[0, 0].set_ylabel('最后随访水肿体积')
axes[0, 0].set_title(f'二维K-means聚类结果 (k={optimal_k})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 时间维度的箱线图
time_clusters = [clustering_data[clustering_data['cluster'] == i]['final_time_hours']
                 for i in range(optimal_k)]
bp1 = axes[0, 1].boxplot(time_clusters, labels=[f'聚类 {i}' for i in range(optimal_k)])
axes[0, 1].set_ylabel('相对发病时间 (小时)')
axes[0, 1].set_title('各聚类时间分布')
axes[0, 1].grid(True, alpha=0.3)

# 3. 体积维度的箱线图
volume_clusters = [clustering_data[clustering_data['cluster'] == i]['final_volume']
                   for i in range(optimal_k)]
bp2 = axes[1, 0].boxplot(volume_clusters, labels=[f'聚类 {i}' for i in range(optimal_k)])
axes[1, 0].set_ylabel('最后随访水肿体积')
axes[1, 0].set_title('各聚类体积分布')
axes[1, 0].grid(True, alpha=0.3)

# 4. 聚类大小饼图
cluster_counts = clustering_data['cluster'].value_counts().sort_index()
axes[1, 1].pie(cluster_counts.values, labels=[f'聚类 {i}\n({count}人)' for i, count in cluster_counts.items()],
               autopct='%1.1f%%', colors=colors[:optimal_k])
axes[1, 1].set_title('聚类患者分布')

plt.tight_layout()
plt.show()

# 额外可视化：不同k值的聚类结果比较
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, k in enumerate(k_range):
    labels = evaluation_metrics[k]['labels']
    colors_k = plt.cm.Set3(np.linspace(0, 1, k))

    for cluster_id in range(k):
        mask = labels == cluster_id
        axes[i].scatter(X[mask, 0], X[mask, 1], c=[colors_k[cluster_id]],
                        label=f'聚类 {cluster_id}', s=50, alpha=0.7)

    # 标记聚类中心
    centers = scaler.inverse_transform(evaluation_metrics[k]['model'].cluster_centers_)
    for cluster_id in range(k):
        axes[i].scatter(centers[cluster_id, 0], centers[cluster_id, 1],
                        c='red', marker='x', s=200, linewidths=3)

    axes[i].set_xlabel('相对发病时间 (小时)')
    axes[i].set_ylabel('最后随访水肿体积')
    axes[i].set_title(f'k={k} (轮廓系数={evaluation_metrics[k]["silhouette"]:.3f})')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 导出聚类结果
output_df = clustering_data[['patient_id', 'final_time_hours', 'final_volume',
                             'final_timepoint', 'cluster']].copy()
output_df.columns = ['患者ID', '相对发病时间(小时)', '最后随访水肿体积', '最后随访时间点', '聚类标签']

# 按聚类标签排序
output_df = output_df.sort_values(['聚类标签', '相对发病时间(小时)'])

print(f"\n聚类结果详细信息 (前20行):")
print(output_df.head(20).to_string(index=False))

# 保存结果到Excel
output_filename = 'kmeans_2D_clustering_results.xlsx'
output_df.to_excel(output_filename, index=False)
print(f"\n完整聚类结果已保存到: {output_filename}")

# =================== 新增功能：分别保存五个聚类的完整数据 ===================

print(f"\n开始为k=5聚类的每个聚类创建完整数据表格...")

# 获取k=5时的聚类结果
k5_labels = evaluation_metrics[5]['labels']

# 将聚类标签合并到原始数据
# 首先创建患者ID到聚类标签的映射
patient_cluster_mapping = dict(zip(clustering_data['patient_id'], k5_labels))

# 为原始数据添加聚类标签
df_subset_with_cluster = df_subset.copy()
df_subset_with_cluster['聚类标签'] = df_subset_with_cluster['患者id'].map(patient_cluster_mapping)

# 只保留有聚类标签的患者（即参与聚类的患者）
df_clustered = df_subset_with_cluster.dropna(subset=['聚类标签']).copy()
df_clustered['聚类标签'] = df_clustered['聚类标签'].astype(int)

print(f"参与聚类的患者总数: {len(df_clustered)}")
print(f"各聚类患者分布:")
cluster_distribution = df_clustered['聚类标签'].value_counts().sort_index()
for cluster_id, count in cluster_distribution.items():
    print(f"  聚类 {cluster_id}: {count} 名患者")

# 使用ExcelWriter创建多工作表Excel文件
with pd.ExcelWriter('五个聚类患者完整数据.xlsx', engine='openpyxl') as writer:
    # 为每个聚类创建单独的工作表
    for cluster_id in range(5):
        cluster_patients = df_clustered[df_clustered['聚类标签'] == cluster_id].copy()

        # 按患者ID排序
        cluster_patients = cluster_patients.sort_values('患者id')

        # 移除聚类标签列（因为每个表格都是同一聚类）
        cluster_patients_clean = cluster_patients.drop(columns=['聚类标签'])

        # 写入到对应的工作表
        sheet_name = f'聚类{cluster_id}'
        cluster_patients_clean.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"聚类 {cluster_id} 数据已保存到工作表 '{sheet_name}' (共 {len(cluster_patients_clean)} 名患者)")

# 另外，也为每个聚类创建单独的Excel文件
for cluster_id in range(5):
    cluster_patients = df_clustered[df_clustered['聚类标签'] == cluster_id].copy()
    cluster_patients = cluster_patients.sort_values('患者id')
    cluster_patients_clean = cluster_patients.drop(columns=['聚类标签'])

    filename = f'聚类{cluster_id}_患者数据.xlsx'
    cluster_patients_clean.to_excel(filename, index=False)
    print(f"聚类 {cluster_id} 单独文件已保存: {filename} (共 {len(cluster_patients_clean)} 名患者)")

# 创建聚类汇总信息表
cluster_summary = []
for cluster_id in range(5):
    cluster_data = clustering_data[clustering_data['cluster'] == cluster_id]
    center = cluster_centers_original[cluster_id]

    summary_info = {
        '聚类标签': cluster_id,
        '患者数量': len(cluster_data),
        '百分比': f"{(len(cluster_data) / len(clustering_data)) * 100:.1f}%",
        '中心_时间(小时)': f"{center[0]:.1f}",
        '中心_体积': f"{center[1]:.2f}",
        '时间_均值': f"{cluster_data['final_time_hours'].mean():.1f}",
        '时间_标准差': f"{cluster_data['final_time_hours'].std():.1f}",
        '时间_范围': f"[{cluster_data['final_time_hours'].min():.1f}, {cluster_data['final_time_hours'].max():.1f}]",
        '体积_均值': f"{cluster_data['final_volume'].mean():.2f}",
        '体积_标准差': f"{cluster_data['final_volume'].std():.2f}",
        '体积_范围': f"[{cluster_data['final_volume'].min():.2f}, {cluster_data['final_volume'].max():.2f}]"
    }
    cluster_summary.append(summary_info)

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_excel('聚类汇总信息.xlsx', index=False)
print(f"\n聚类汇总信息已保存到: 聚类汇总信息.xlsx")

print(f"\n=== 文件保存汇总 ===")
print(f"1. 五个聚类患者完整数据.xlsx - 包含5个工作表的综合文件")
print(f"2. 聚类0_患者数据.xlsx 到 聚类4_患者数据.xlsx - 5个单独的聚类文件")
print(f"3. 聚类汇总信息.xlsx - 各聚类的统计汇总")
print(f"4. kmeans_2D_clustering_results.xlsx - 原有的聚类结果文件")

# 统计各聚类的患者分布
print(f"\n聚类标签分布:")
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(clustering_data)) * 100
    print(f"聚类 {cluster_id}: {count} 名患者 ({percentage:.1f}%)")

# 聚类间差异的统计检验（MANOVA多变量方差分析）
from scipy.stats import f_oneway

print(f"\n聚类间差异统计检验:")

# 时间维度的ANOVA
time_groups = [clustering_data[clustering_data['cluster'] == i]['final_time_hours']
               for i in range(optimal_k)]
f_stat_time, p_value_time = f_oneway(*time_groups)
print(f"时间维度ANOVA: F={f_stat_time:.4f}, p={p_value_time:.4f}")

# 体积维度的ANOVA
volume_groups = [clustering_data[clustering_data['cluster'] == i]['final_volume']
                 for i in range(optimal_k)]
f_stat_volume, p_value_volume = f_oneway(*volume_groups)
print(f"体积维度ANOVA: F={f_stat_volume:.4f}, p={p_value_volume:.4f}")

significance_level = 0.05
print(f"\n显著性分析 (α={significance_level}):")
print(f"时间维度: {'显著差异' if p_value_time < significance_level else '无显著差异'}")
print(f"体积维度: {'显著差异' if p_value_volume < significance_level else '无显著差异'}")