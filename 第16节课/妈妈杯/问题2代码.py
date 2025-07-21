import pandas as pd
import numpy as np
import os
from datetime import datetime

# 创建结果文件夹
results_folder = "问题2结果"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 读取数据
file_path = r'C:\Users\18344\Desktop\C题 -数学建模老哥\音频分析结果\音频分析结果.csv'
df = pd.read_csv(file_path)

# 筛选去除原始音乐和原始语音文件 - 创建副本而不是视图
df_filtered = df[~df['文件名'].str.contains("原始", na=False)].copy()

# 如果 '比特深度' 列不存在，使用 '采样率(Hz)' 和 '声道数' 来代替
if '比特深度' not in df_filtered.columns:
    df_filtered['比特深度'] = df_filtered['采样率(Hz)']  # 临时使用采样率代替比特深度

# 定义格式映射
format_map = {1: 'mp3', 2: 'wav', 3: 'aac'}

# 使用 .loc 修改数据，避免 SettingWithCopyWarning
df_filtered.loc[:, '格式'] = df_filtered['格式'].map({'mp3': 1, 'wav': 2, 'aac': 3})
df_filtered.loc[:, '格式名称'] = df_filtered['格式'].map(format_map)

# 创建音质和文件大小的性价比指标（QSI）
df_filtered.loc[:, 'QSI'] = df_filtered['PSNR(dB)'] / df_filtered['大小(KB)']

# 根据QSI对音频文件进行排序
df_filtered_sorted = df_filtered.sort_values(by='QSI', ascending=False)

# 分析语音和音乐文件
df_voice = df_filtered_sorted[df_filtered_sorted['场景'].str.contains('通用场景', na=False)]
df_music = df_filtered_sorted[~df_filtered_sorted['场景'].str.contains('通用场景', na=False)]

# 生成统计报告
best_voice = df_voice.iloc[0]
best_music = df_music.iloc[0]

# 创建详细的统计报告
report = f"""音频分析报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 最佳语音参数组合:
------------------------
格式: {format_map[best_voice['格式']]}
采样率: {best_voice['采样率(Hz)']} Hz
声道数: {best_voice['声道数']}
比特率: {best_voice['比特率']} kbps
PSNR: {best_voice['PSNR(dB)']:.2f} dB
文件大小: {best_voice['大小(KB)']:.2f} KB
QSI: {best_voice['QSI']:.4f}

2. 最佳音乐参数组合:
------------------------
格式: {format_map[best_music['格式']]}
采样率: {best_music['采样率(Hz)']} Hz
声道数: {best_music['声道数']}
比特率: {best_music['比特率']} kbps
PSNR: {best_music['PSNR(dB)']:.2f} dB
文件大小: {best_music['大小(KB)']:.2f} KB
QSI: {best_music['QSI']:.4f}

3. 统计分析:
------------------------
语音文件平均QSI: {df_voice['QSI'].mean():.4f}
音乐文件平均QSI: {df_music['QSI'].mean():.4f}
语音文件QSI标准差: {df_voice['QSI'].std():.4f}
音乐文件QSI标准差: {df_music['QSI'].std():.4f}

4. 格式性能排名:
------------------------
{df_filtered.groupby('格式名称')['QSI'].mean().sort_values(ascending=False).to_string()}

5. 详细数据分析:
------------------------
语音文件数量: {len(df_voice)}
音乐文件数量: {len(df_music)}

各格式文件数量:
{df_filtered['格式名称'].value_counts().to_string()}

采样率分布:
{df_filtered['采样率(Hz)'].value_counts().sort_index().to_string()}

比特率分布:
{df_filtered['比特率'].value_counts().sort_index().to_string()}
"""

# 保存报告
with open(os.path.join(results_folder, 'audio_analysis_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

# 保存处理后的数据
df_filtered.to_csv(os.path.join(results_folder, 'processed_audio_data.csv'), index=False, encoding='utf-8')

print(f"分析完成！结果已保存到 {results_folder} 文件夹中。")
print("生成的文件包括：")
print("1. audio_analysis_report.txt - 详细分析报告")
print("2. processed_audio_data.csv - 处理后的数据")

