import os
from pydub.utils import mediainfo
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
import matplotlib as mpl

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

sns.set_theme(style="whitegrid")
sns.set_palette("husl")

AudioSegment.ffmpeg = r"C:\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-7.0.2-essentials_build\bin\ffprobe.exe"

audio_directory = r'E:\桌面\mathflies\python\Mathmodel\第17节课相关演示代码\附件1'

# 创建结果文件夹
result_dir = "音频分析结果"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def get_format_name(file_info, file_path):
    ext = os.path.splitext(file_path)[1].lower()
    format_map = {'.aac': 'aac', '.mp3': 'mp3', '.wav': 'wav'}
    return format_map.get(ext, file_info['format_name'].lower())

def get_complexity_score(file_info, format_name):
    sample_rate = int(file_info['sample_rate'])
    bit_rate = int(file_info.get('bit_rate', '0')) if isinstance(file_info.get('bit_rate', '0'), str) and file_info.get('bit_rate', '0').isdigit() else 0
    
    format_scores = {'wav': 2, 'mp3': 3, 'aac': 4}
    base_score = 5 + format_scores.get(format_name, 0)
    
    if sample_rate >= 48000: base_score += 2
    elif sample_rate >= 44100: base_score += 1
    if bit_rate > 192000: base_score += 1
    
    return min(10, max(1, base_score))

def get_application_scenarios(file_info, format_name):
    sample_rate = int(file_info['sample_rate'])
    bit_rate = int(file_info.get('bit_rate', '0')) if isinstance(file_info.get('bit_rate', '0'), str) and file_info.get('bit_rate', '0').isdigit() else 0
    
    scenarios = set()
    if format_name == 'wav' and sample_rate >= 44100: scenarios.add("专业录音")
    if format_name in ['mp3', 'aac']: scenarios.update(["流媒体传输", "网络传输"])
    if bit_rate <= 128000 or format_name == 'aac': scenarios.add("移动设备")
    if sample_rate >= 48000 and format_name == 'wav': scenarios.add("高质量音频存储")
    
    return "、".join(scenarios) if scenarios else "通用场景"

def compute_audio_metrics(original_file, compressed_file):
    original_audio = AudioSegment.from_file(original_file)
    compressed_audio = AudioSegment.from_file(compressed_file)
    
    original_samples = np.array(original_audio.get_array_of_samples())
    compressed_samples = np.array(compressed_audio.get_array_of_samples())
    
    min_length = min(len(original_samples), len(compressed_samples))
    diff = np.abs(original_samples[:min_length] - compressed_samples[:min_length])
    mse = np.mean(diff ** 2)
    
    max_pixel = 2**16 - 1
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse) if mse > 0 else 100.0
    psnr = max(0, min(100, psnr))
    
    return {'MSE': mse, 'PSNR': psnr}

def get_audio_info(directory):
    audio_files = []
    reference_file = next((os.path.join(root, file) for root, _, files in os.walk(directory) 
                          for file in files if '原始' in file and file.endswith('.wav')), None)
    
    if not reference_file:
        print("未找到参考音频文件！")
        return []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.aac')):
                audio_path = os.path.join(root, file)
                file_info = mediainfo(audio_path)
                format_name = get_format_name(file_info, audio_path)
                metrics = compute_audio_metrics(reference_file, audio_path)
                
                audio_files.append({
                    '文件名': file,
                    '格式': format_name,
                    '时长(s)': float(file_info['duration']),
                    '大小(KB)': os.path.getsize(audio_path) / 1024,
                    '采样率(Hz)': int(file_info['sample_rate']),
                    '声道数': int(file_info['channels']),
                    '比特率': file_info.get('bit_rate', 'N/A'),
                    'MSE': metrics['MSE'],
                    'PSNR(dB)': metrics['PSNR'],
                    '复杂度': get_complexity_score(file_info, format_name),
                    '场景': get_application_scenarios(file_info, format_name)
                })
    return audio_files

def visualize_results(audio_files_info):
    df = pd.DataFrame(audio_files_info)
    
    # 文件大小分布
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='大小(KB)', y='文件名')
    plt.title('音频文件大小分布', pad=15)
    plt.xlabel('大小 (KB)')
    plt.ylabel('文件名')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '1_文件大小分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PSNR分布
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='PSNR(dB)', y='文件名')
    plt.title('音频质量(PSNR)分布', pad=15)
    plt.xlabel('PSNR (dB)')
    plt.ylabel('文件名')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '2_音频质量分布.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 文件大小与音质关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='大小(KB)', y='PSNR(dB)', 
                    hue='格式', size='复杂度', sizes=(100, 400))
    plt.title('文件大小与音质关系', pad=15)
    plt.xlabel('大小 (KB)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '3_大小音质关系.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 不同格式音质分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='格式', y='PSNR(dB)')
    plt.title('不同格式音质分布', pad=15)
    plt.xlabel('格式')
    plt.ylabel('PSNR (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '4_格式音质分布.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    audio_files_info = get_audio_info(audio_directory)
    
    df = pd.DataFrame(audio_files_info)
    
    # 保存CSV文件
    csv_path = os.path.join(result_dir, '音频分析结果.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 保存统计信息到文本文件
    stats_path = os.path.join(result_dir, '统计信息.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=== 音频分析结果摘要 ===\n")
        f.write(df.to_string(index=False))
        f.write("\n\n=== 统计信息 ===\n")
        f.write(df.describe().to_string())
    
    # 生成可视化
    visualize_results(audio_files_info)
    
    print(f"\n分析结果已保存到 {result_dir} 文件夹：")
    print(f"1. 音频分析结果.csv - 详细数据")
    print(f"2. 统计信息.txt - 统计摘要")
    print(f"3. 可视化图表：")
    print(f"   - 1_文件大小分布.png")
    print(f"   - 2_音频质量分布.png")
    print(f"   - 3_大小音质关系.png")
    print(f"   - 4_格式音质分布.png")

if __name__ == "__main__":
    main()
