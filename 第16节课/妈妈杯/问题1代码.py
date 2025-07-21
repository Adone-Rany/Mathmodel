import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import os

# 创建结果文件夹
output_dir = '问题1结果'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置整体风格
plt.style.use('default')  # 使用默认样式
sns.set_theme(style="whitegrid")  # 设置seaborn主题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 读取数据
file_path = r'E:\桌面\mathflies\python\Mathmodel\第17节课相关演示代码\音频分析结果\音频分析结果.csv'
df = pd.read_csv(file_path)

# 定义最大值用于归一化
MSE_max = df['MSE'].max()
complexity_max = df['复杂度'].max()
sampling_rate_max = df['采样率(Hz)'].max()
channels_max = df['声道数'].max()
bitrate_max = df['比特率'].max()

# 定义权重
w1, w2, w3, w4, w5, w6, w7 = 0.2, 0.25, 0.2, 0.1, 0.1, 0.1, 0.05

# 适用场景的权重映射
scene_weights = {
    '高质量音频存储、专业录音': 1.0,
    '通用场景': 0.8,
    '流媒体传输、网络传输': 0.6,
    '流媒体传输、移动设备、网络传输': 0.5,
    '移动设备': 0.7
}

def calculate_CPI(row):
    MSE_normalized = 1 - (row['MSE'] / MSE_max)
    complexity_normalized = row['复杂度'] / complexity_max
    sampling_rate_normalized = row['采样率(Hz)'] / sampling_rate_max
    channels_normalized = row['声道数'] / channels_max
    bitrate_normalized = row['比特率'] / bitrate_max
    file_size_normalized = 1 / row['大小(KB)']
    scene_weight = scene_weights.get(row['场景'], 0.8)

    CPI = (w1 * file_size_normalized) + (w2 * MSE_normalized) + (w3 * complexity_normalized) + \
          (w4 * sampling_rate_normalized) + (w5 * channels_normalized) + (w6 * bitrate_normalized) + \
          (w7 * scene_weight)
    return CPI

# 计算CPI
df['CPI'] = df.apply(calculate_CPI, axis=1)

# 1. CPI对比图 - 使用点图
plt.figure(figsize=(12, 6))
cpi_by_format = df.groupby('格式')['CPI'].mean().sort_values(ascending=True)
sns.pointplot(x=cpi_by_format.index, y=cpi_by_format.values, color='#2E86C1', 
              markers='o', linewidth=2, markersize=8)
plt.title('各音频格式CPI综合评分对比', fontsize=14, pad=20)
for i, value in enumerate(cpi_by_format.values):
    plt.text(i, value, f'{value:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_CPI对比图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. 各指标箱型图 - 使用小提琴图
plt.figure(figsize=(12, 6))
metrics = ['MSE', '复杂度', '采样率(Hz)', '声道数', '比特率']
df_melted = df.melt(id_vars=['格式'], value_vars=metrics)
sns.violinplot(data=df_melted, x='variable', y='value', hue='variable', 
               inner='box', legend=False)
plt.title('各音频格式指标分布', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_指标分布图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. 文件大小与CPI的散点图 - 使用气泡图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='大小(KB)', y='CPI', hue='格式', 
                size='复杂度', sizes=(100, 1000), alpha=0.6,
                palette='deep')
plt.title('文件大小与CPI关系散点图', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_文件大小与CPI关系图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. 适用场景分布 - 使用环形图
plt.figure(figsize=(10, 8))
scene_counts = df['场景'].value_counts()
colors = sns.color_palette('pastel')[0:len(scene_counts)]
plt.pie(scene_counts, labels=scene_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors, wedgeprops=dict(width=0.5))
plt.title('音频格式适用场景分布', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_场景分布图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. 各格式在不同场景下的CPI表现 - 使用改进的热力图
plt.figure(figsize=(12, 8))
pivot_table = df.pivot_table(values='CPI', index='格式', columns='场景', aggfunc='mean')

# 创建自定义颜色映射
colors = sns.color_palette("RdYlBu_r", as_cmap=True)
sns.heatmap(pivot_table, 
            annot=True,  # 显示数值
            fmt='.2f',   # 数值格式
            cmap=colors, # 使用自定义颜色映射
            center=pivot_table.mean().mean(),  # 设置中心值
            square=True,  # 使用正方形单元格
            cbar_kws={'label': 'CPI值', 'orientation': 'vertical'},
            annot_kws={'size': 10},  # 设置数值标签大小
            linewidths=0.5,  # 添加网格线
            linecolor='white')  # 网格线颜色

plt.title('各格式在不同场景下的CPI表现', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_场景CPI热力图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 保存分析报告到文本文件
with open(os.path.join(output_dir, '分析报告.txt'), 'w', encoding='utf-8') as f:
    f.write("=== 音频格式分析报告 ===\n\n")
    
    f.write("1. CPI综合评分排名:\n")
    for format_name, cpi in cpi_by_format.items():
        f.write(f"{format_name}: {cpi:.3f}\n")
    
    f.write("\n2. 各格式特点分析:\n")
    for format_name in df['格式'].unique():
        format_data = df[df['格式'] == format_name]
        f.write(f"\n{format_name}:\n")
        f.write(f"- 平均文件大小: {format_data['大小(KB)'].mean():.2f} KB\n")
        f.write(f"- 平均MSE: {format_data['MSE'].mean():.4f}\n")
        f.write(f"- 平均复杂度: {format_data['复杂度'].mean():.2f}\n")
        f.write(f"- 主要适用场景: {format_data['场景'].mode().iloc[0]}\n")
    
    f.write("\n3. 性能指标相关性分析:\n")
    correlation_matrix = df[['CPI', 'MSE', '复杂度', '采样率(Hz)', '声道数', '比特率', '大小(KB)']].corr()
    f.write(str(correlation_matrix['CPI'].sort_values(ascending=False)))

print(f"分析结果已保存到 {output_dir} 文件夹中")
