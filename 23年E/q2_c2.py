# 热力图，堆叠柱状图，扇形图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from scipy import stats

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 治疗手段列名映射
treatment_columns = {
    '脑室引流': 'Q',
    '止血治疗': 'R',
    '降颅压治疗': 'S',
    '降压治疗': 'T',
    '镇静、镇痛治疗': 'U',
    '止吐护胃': 'V',
    '营养神经': 'W'
}


def read_cluster_patient_ids():
    """读取五个聚类的患者ID"""
    cluster_patient_ids = {}

    for cluster_id in range(5):
        filename = f'聚类{cluster_id}_患者数据.xlsx'
        try:
            df_cluster = pd.read_excel(filename)
            patient_ids = df_cluster['患者id'].unique().tolist()
            cluster_patient_ids[cluster_id] = patient_ids
            print(f"聚类 {cluster_id}: 读取到 {len(patient_ids)} 名患者")
        except FileNotFoundError:
            print(f"警告: 找不到文件 {filename}")
        except Exception as e:
            print(f"读取聚类 {cluster_id} 文件时出错: {e}")

    return cluster_patient_ids


def load_clinical_data():
    """读取临床信息表"""
    try:
        clinical_df = pd.read_excel('表1-患者列表及临床信息.xlsx')
        print(f"成功读取临床信息表，共 {len(clinical_df)} 名患者")
        return clinical_df
    except FileNotFoundError:
        print("错误: 找不到文件 '表1-患者列表及临床信息.xlsx'")
        return None
    except Exception as e:
        print(f"读取临床信息表时出错: {e}")
        return None


def analyze_treatment_by_cluster(cluster_patient_ids, clinical_df):
    """分析各聚类的治疗手段使用情况"""

    # 创建结果存储字典
    cluster_treatment_stats = {}
    all_cluster_data = []

    # 为每个聚类分析治疗手段
    for cluster_id, patient_ids in cluster_patient_ids.items():
        # 筛选出当前聚类的患者数据
        cluster_clinical = clinical_df[clinical_df['患者id'].isin(patient_ids)].copy()

        if len(cluster_clinical) == 0:
            print(f"警告: 聚类 {cluster_id} 没有找到匹配的临床数据")
            continue

        print(f"\n聚类 {cluster_id} 治疗手段分析:")
        print(f"患者数量: {len(cluster_clinical)}")

        # 统计各治疗手段的使用情况
        treatment_stats = {}

        for treatment_name, col_letter in treatment_columns.items():
            # 通过列索引获取数据（Q列是第16列，从0开始计数是15）
            col_index = ord(col_letter) - ord('A')  # 将字母转换为列索引

            if col_index < len(cluster_clinical.columns):
                col_name = cluster_clinical.columns[col_index]
                treatment_data = cluster_clinical.iloc[:, col_index]

                # 统计使用和未使用的数量
                used_count = (treatment_data == 1).sum()
                not_used_count = (treatment_data == 0).sum()
                missing_count = treatment_data.isna().sum()
                total_valid = used_count + not_used_count

                if total_valid > 0:
                    usage_rate = (used_count / total_valid) * 100
                else:
                    usage_rate = 0

                treatment_stats[treatment_name] = {
                    '使用人数': used_count,
                    '未使用人数': not_used_count,
                    '缺失数据': missing_count,
                    '有效数据': total_valid,
                    '使用率(%)': usage_rate
                }

                print(f"  {treatment_name}: {used_count}/{total_valid} ({usage_rate:.1f}%)")
            else:
                print(f"  警告: 找不到 {treatment_name} 对应的列")

        cluster_treatment_stats[cluster_id] = {
            'patient_count': len(cluster_clinical),
            'treatments': treatment_stats,
            'clinical_data': cluster_clinical
        }

        # 为总体分析准备数据
        for _, patient_row in cluster_clinical.iterrows():
            patient_data = {'聚类': cluster_id, '患者id': patient_row['患者id']}
            for treatment_name, col_letter in treatment_columns.items():
                col_index = ord(col_letter) - ord('A')
                if col_index < len(cluster_clinical.columns):
                    patient_data[treatment_name] = patient_row.iloc[col_index]
            all_cluster_data.append(patient_data)

    return cluster_treatment_stats, pd.DataFrame(all_cluster_data)


def create_treatment_summary_table(cluster_treatment_stats):
    """创建治疗手段汇总表"""
    summary_data = []

    for cluster_id, cluster_info in cluster_treatment_stats.items():
        for treatment_name, stats in cluster_info['treatments'].items():
            summary_data.append({
                '聚类': cluster_id,
                '治疗手段': treatment_name,
                '使用人数': stats['使用人数'],
                '未使用人数': stats['未使用人数'],
                '有效数据总数': stats['有效数据'],
                '使用率(%)': f"{stats['使用率(%)']:.1f}%",
                '缺失数据': stats['缺失数据']
            })

    return pd.DataFrame(summary_data)


def statistical_analysis(all_cluster_data):
    """进行统计学分析"""
    print(f"\n=== 统计学分析 ===")

    # 为每个治疗手段进行卡方检验
    chi2_results = {}

    for treatment_name in treatment_columns.keys():
        # 创建列联表
        contingency_table = pd.crosstab(all_cluster_data['聚类'],
                                        all_cluster_data[treatment_name],
                                        dropna=False)

        print(f"\n{treatment_name} - 列联表:")
        print(contingency_table)

        # 只对有效数据进行统计检验（排除NaN）
        valid_data = all_cluster_data.dropna(subset=[treatment_name])
        if len(valid_data) > 0:
            contingency_valid = pd.crosstab(valid_data['聚类'], valid_data[treatment_name])

            # 检查是否适合卡方检验（期望频数≥5）
            if contingency_valid.size > 0 and (contingency_valid.values >= 5).all():
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_valid)
                    chi2_results[treatment_name] = {
                        'chi2': chi2,
                        'p_value': p_value,
                        'dof': dof,
                        'significant': p_value < 0.05
                    }
                    print(f"卡方检验: χ² = {chi2:.4f}, p = {p_value:.4f}, 自由度 = {dof}")
                    print(f"结果: {'有显著差异' if p_value < 0.05 else '无显著差异'}")
                except:
                    print("卡方检验失败")
            else:
                print("数据不满足卡方检验条件（期望频数<5）")

    return chi2_results


def visualize_treatment_analysis(cluster_treatment_stats, all_cluster_data):
    """可视化治疗手段分析结果"""

    # 1. 创建使用率热力图
    heatmap_data = []
    cluster_ids = sorted(cluster_treatment_stats.keys())

    for cluster_id in cluster_ids:
        cluster_rates = []
        for treatment_name in treatment_columns.keys():
            if treatment_name in cluster_treatment_stats[cluster_id]['treatments']:
                rate = cluster_treatment_stats[cluster_id]['treatments'][treatment_name]['使用率(%)']
                cluster_rates.append(rate)
            else:
                cluster_rates.append(0)
        heatmap_data.append(cluster_rates)

    heatmap_df = pd.DataFrame(heatmap_data,
                              index=[f'聚类{i}' for i in cluster_ids],
                              columns=list(treatment_columns.keys()))

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': '使用率 (%)'})
    plt.title('各聚类治疗手段使用率热力图', fontsize=14, fontweight='bold')
    plt.ylabel('聚类')
    plt.xlabel('治疗手段')
    plt.tight_layout()
    plt.show()

    # 2. 创建堆叠柱状图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i, treatment_name in enumerate(treatment_columns.keys()):
        if i < len(axes):
            used_counts = []
            not_used_counts = []
            cluster_labels = []

            for cluster_id in cluster_ids:
                if treatment_name in cluster_treatment_stats[cluster_id]['treatments']:
                    stats = cluster_treatment_stats[cluster_id]['treatments'][treatment_name]
                    used_counts.append(stats['使用人数'])
                    not_used_counts.append(stats['未使用人数'])
                    cluster_labels.append(f'聚类{cluster_id}')

            x = np.arange(len(cluster_labels))
            width = 0.6

            axes[i].bar(x, used_counts, width, label='使用', color='#2E8B57', alpha=0.8)
            axes[i].bar(x, not_used_counts, width, bottom=used_counts,
                        label='未使用', color='#CD5C5C', alpha=0.8)

            axes[i].set_xlabel('聚类')
            axes[i].set_ylabel('患者数量')
            axes[i].set_title(f'{treatment_name}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(cluster_labels)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # 添加使用率标注
            for j, (used, total) in enumerate(zip(used_counts,
                                                  [u + nu for u, nu in zip(used_counts, not_used_counts)])):
                if total > 0:
                    rate = (used / total) * 100
                    axes[i].text(j, total + 0.5, f'{rate:.1f}%',
                                 ha='center', va='bottom', fontweight='bold')

    # 删除多余的子图
    for i in range(len(treatment_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('各聚类治疗手段使用情况对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3. 创建总体使用率对比图
    plt.figure(figsize=(14, 8))

    treatment_names = list(treatment_columns.keys())
    x = np.arange(len(treatment_names))
    width = 0.15

    for i, cluster_id in enumerate(cluster_ids):
        rates = []
        for treatment_name in treatment_names:
            if treatment_name in cluster_treatment_stats[cluster_id]['treatments']:
                rate = cluster_treatment_stats[cluster_id]['treatments'][treatment_name]['使用率(%)']
                rates.append(rate)
            else:
                rates.append(0)

        plt.bar(x + i * width, rates, width, label=f'聚类{cluster_id}',
                color=colors[i], alpha=0.8)

    plt.xlabel('治疗手段')
    plt.ylabel('使用率 (%)')
    plt.title('各聚类治疗手段使用率对比', fontsize=14, fontweight='bold')
    plt.xticks(x + width * 2, treatment_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def create_pie_charts_for_clusters(cluster_treatment_stats):
    """为每个聚类创建治疗手段使用情况扇形图"""

    # 创建子图布局 - 2行3列，第6个位置放总体对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 为前5个聚类创建扇形图
    for cluster_id, cluster_info in cluster_treatment_stats.items():
        if cluster_id < 5:  # 确保不超过5个聚类
            ax = axes[cluster_id]

            # 准备扇形图数据
            treatment_names = []
            usage_counts = []
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98']

            for i, treatment_name in enumerate(treatment_columns.keys()):
                if treatment_name in cluster_info['treatments']:
                    stats = cluster_info['treatments'][treatment_name]
                    if stats['使用人数'] > 0:  # 只显示有人使用的治疗手段
                        treatment_names.append(treatment_name)
                        usage_counts.append(stats['使用人数'])

            if usage_counts:  # 如果有数据
                # 创建扇形图
                wedges, texts, autotexts = ax.pie(usage_counts, labels=treatment_names, autopct='%1.1f%%',
                                                  colors=colors[:len(usage_counts)], startangle=90)

                # 美化文本
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)

                for text in texts:
                    text.set_fontsize(8)

                ax.set_title(f'聚类 {cluster_id} 治疗手段使用分布\n(总患者数: {cluster_info["patient_count"]})',
                             fontsize=12, fontweight='bold', pad=20)
            else:
                ax.text(0.5, 0.5, '该聚类无治疗数据', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'聚类 {cluster_id}', fontsize=12, fontweight='bold')

    # 在第6个位置创建总体治疗手段使用率对比
    if len(cluster_treatment_stats) > 0:
        ax = axes[5]

        # 计算总体各治疗手段的平均使用率
        overall_usage = {}
        for treatment_name in treatment_columns.keys():
            total_used = 0
            total_patients = 0

            for cluster_info in cluster_treatment_stats.values():
                if treatment_name in cluster_info['treatments']:
                    stats = cluster_info['treatments'][treatment_name]
                    total_used += stats['使用人数']
                    total_patients += stats['有效数据']

            if total_patients > 0:
                overall_rate = (total_used / total_patients) * 100
                overall_usage[treatment_name] = overall_rate

        # 创建总体使用率扇形图
        if overall_usage:
            treatment_names = list(overall_usage.keys())
            usage_rates = list(overall_usage.values())

            wedges, texts, autotexts = ax.pie(usage_rates, labels=treatment_names, autopct='%1.1f%%',
                                              colors=colors[:len(usage_rates)], startangle=90)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)

            for text in texts:
                text.set_fontsize(8)

            ax.set_title('总体治疗手段使用率分布', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('各聚类治疗手段使用情况扇形图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_individual_cluster_pie_charts(cluster_treatment_stats):
    """为每个聚类单独创建大尺寸扇形图"""

    for cluster_id, cluster_info in cluster_treatment_stats.items():
        plt.figure(figsize=(10, 8))

        # 准备数据
        treatment_names = []
        usage_counts = []
        usage_rates = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98']

        for treatment_name in treatment_columns.keys():
            if treatment_name in cluster_info['treatments']:
                stats = cluster_info['treatments'][treatment_name]
                if stats['有效数据'] > 0:  # 有有效数据的治疗手段
                    treatment_names.append(treatment_name)
                    usage_counts.append(stats['使用人数'])
                    usage_rates.append(stats['使用率(%)'])

        if usage_counts:
            # 创建扇形图
            wedges, texts, autotexts = plt.pie(usage_counts, labels=treatment_names,
                                               autopct=lambda
                                                   pct: f'{pct:.1f}%\n({int(pct / 100 * sum(usage_counts))}人)',
                                               colors=colors[:len(usage_counts)],
                                               startangle=90,
                                               textprops={'fontsize': 10})

            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            for text in texts:
                text.set_fontweight('bold')

            plt.title(f'聚类 {cluster_id} 治疗手段使用分布\n总患者数: {cluster_info["patient_count"]}人',
                      fontsize=14, fontweight='bold', pad=20)

            # 添加图例，显示具体使用率
            legend_labels = [f'{name}: {rate:.1f}% ({count}人)'
                             for name, rate, count in zip(treatment_names, usage_rates, usage_counts)]
            plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        else:
            plt.text(0.5, 0.5, '该聚类无有效治疗数据', ha='center', va='center',
                     fontsize=16, transform=plt.gca().transAxes)
            plt.title(f'聚类 {cluster_id}', fontsize=14, fontweight='bold')

        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def create_detailed_analysis_tables(cluster_treatment_stats, all_cluster_data, chi2_results):
    """创建详细分析表格"""

    # 1. 聚类治疗手段汇总表
    summary_data = []
    for cluster_id, cluster_info in cluster_treatment_stats.items():
        row_data = {'聚类': cluster_id, '患者总数': cluster_info['patient_count']}

        for treatment_name in treatment_columns.keys():
            if treatment_name in cluster_info['treatments']:
                stats = cluster_info['treatments'][treatment_name]
                row_data[f'{treatment_name}_使用人数'] = stats['使用人数']
                row_data[f'{treatment_name}_使用率'] = f"{stats['使用率(%)']:.1f}%"
            else:
                row_data[f'{treatment_name}_使用人数'] = 0
                row_data[f'{treatment_name}_使用率'] = "0.0%"

        summary_data.append(row_data)

    summary_df = pd.DataFrame(summary_data)

    # 2. 统计检验结果表
    test_results_data = []
    for treatment_name, result in chi2_results.items():
        test_results_data.append({
            '治疗手段': treatment_name,
            'χ²值': f"{result['chi2']:.4f}",
            'p值': f"{result['p_value']:.4f}",
            '自由度': result['dof'],
            '是否显著差异': '是' if result['significant'] else '否',
            '显著性': '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result[
                                                                                                               'p_value'] < 0.05 else 'ns'
        })

    test_results_df = pd.DataFrame(test_results_data)

    # 3. 详细交叉表
    cross_tables = {}
    for treatment_name in treatment_columns.keys():
        # 创建交叉表
        valid_data = all_cluster_data.dropna(subset=[treatment_name])
        if len(valid_data) > 0:
            cross_table = pd.crosstab(valid_data['聚类'], valid_data[treatment_name],
                                      margins=True, margins_name='总计')
            cross_table.columns = ['未使用', '使用', '总计']
            cross_tables[treatment_name] = cross_table

    return summary_df, test_results_df, cross_tables


def save_results_to_excel(summary_df, test_results_df, cross_tables, cluster_treatment_stats):
    """保存结果到Excel文件"""

    with pd.ExcelWriter('聚类治疗手段统计分析结果.xlsx', engine='openpyxl') as writer:
        # 保存汇总表
        summary_df.to_excel(writer, sheet_name='聚类治疗汇总', index=False)

        # 保存统计检验结果
        test_results_df.to_excel(writer, sheet_name='统计检验结果', index=False)

        # 保存每个治疗手段的交叉表
        for treatment_name, cross_table in cross_tables.items():
            sheet_name = f'{treatment_name}交叉表'[:31]  # Excel工作表名长度限制
            cross_table.to_excel(writer, sheet_name=sheet_name)

        # 保存各聚类详细数据
        for cluster_id, cluster_info in cluster_treatment_stats.items():
            # 提取治疗相关列
            clinical_data = cluster_info['clinical_data']

            # 正确提取患者id列和治疗手段列
            treatment_columns_list = ['患者id']
            treatment_data_dict = {'患者id': clinical_data['患者id']}

            # 根据列字母索引提取对应的治疗数据
            for treatment_name, col_letter in treatment_columns.items():
                col_index = ord(col_letter) - ord('A')  # 将字母转换为列索引
                if col_index < len(clinical_data.columns):
                    column_data = clinical_data.iloc[:, col_index]
                    treatment_data_dict[treatment_name] = column_data
                    treatment_columns_list.append(treatment_name)

            # 创建治疗数据DataFrame
            treatment_data = pd.DataFrame(treatment_data_dict)

            sheet_name = f'聚类{cluster_id}治疗详情'
            treatment_data.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n详细分析结果已保存到: 聚类治疗手段统计分析结果.xlsx")


# 主程序执行
print("=== 聚类患者治疗手段统计分析 ===")

# 1. 读取聚类患者ID
print("1. 读取五个聚类患者ID...")
cluster_patient_ids = read_cluster_patient_ids()

if not cluster_patient_ids:
    print("错误: 无法读取聚类数据")
    exit()

# 2. 读取临床信息
print("\n2. 读取临床信息表...")
clinical_df = load_clinical_data()

if clinical_df is None:
    print("错误: 无法读取临床信息表")
    exit()

# 3. 分析治疗手段使用情况
print("\n3. 分析各聚类治疗手段使用情况...")
cluster_treatment_stats, all_cluster_data = analyze_treatment_by_cluster(cluster_patient_ids, clinical_df)

# 4. 统计学检验
print("\n4. 进行统计学检验...")
chi2_results = statistical_analysis(all_cluster_data)

# 5. 创建汇总表
print("\n5. 创建分析表格...")
summary_df, test_results_df, cross_tables = create_detailed_analysis_tables(
    cluster_treatment_stats, all_cluster_data, chi2_results)

# 6. 可视化分析
print("\n6. 生成可视化图表...")
visualize_treatment_analysis(cluster_treatment_stats, all_cluster_data)
create_pie_charts_for_clusters(cluster_treatment_stats)
create_individual_cluster_pie_charts(cluster_treatment_stats)

# 7. 保存结果
print("\n7. 保存分析结果...")
save_results_to_excel(summary_df, test_results_df, cross_tables, cluster_treatment_stats)

# 8. 打印关键发现
print(f"\n=== 关键发现总结 ===")

# 计算总体统计
total_patients = sum([info['patient_count'] for info in cluster_treatment_stats.values()])
print(f"分析的患者总数: {total_patients}")

# 找出使用率差异最大的治疗手段
max_diff_treatment = None
max_diff_value = 0

for treatment_name in treatment_columns.keys():
    rates = []
    for cluster_id in cluster_treatment_stats.keys():
        if treatment_name in cluster_treatment_stats[cluster_id]['treatments']:
            rate = cluster_treatment_stats[cluster_id]['treatments'][treatment_name]['使用率(%)']
            rates.append(rate)

    if rates:
        diff = max(rates) - min(rates)
        if diff > max_diff_value:
            max_diff_value = diff
            max_diff_treatment = treatment_name

if max_diff_treatment:
    print(f"聚类间差异最大的治疗手段: {max_diff_treatment} (差异: {max_diff_value:.1f}%)")

# 统计显著差异的治疗手段数量
significant_treatments = [name for name, result in chi2_results.items() if result['significant']]
print(f"存在显著聚类差异的治疗手段数量: {len(significant_treatments)}")
if significant_treatments:
    print(f"具体包括: {', '.join(significant_treatments)}")

print(f"\n分析完成！详细结果已保存到Excel文件中。")