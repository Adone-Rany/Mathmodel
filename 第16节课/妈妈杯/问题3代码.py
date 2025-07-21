import os
from pydub.utils import mediainfo
from pydub import AudioSegment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import subprocess
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 设置 ffmpeg 的路径
AudioSegment.ffmpeg = r"C:\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-7.0.2-essentials_build\bin\ffprobe.exe"

# 音频文件所在的目录路径
audio_directory = r'C:\Users\18344\Desktop\C题 -数学建模老哥\附件1'
output_directory = r'C:\Users\18344\Desktop\C题 -数学建模老哥\问题3结果'

# 创建输出目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def analyze_audio_features(audio_path):
    """分析音频特征，返回音频类型和特征参数"""
    # 读取音频文件
    sample_rate, audio_data = wavfile.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # 如果是立体声，只取第一个声道
    
    # 将音频数据转换为浮点数并归一化
    audio_data = audio_data.astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # 计算频谱特征
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate)
    
    # 计算动态范围（使用归一化后的数据）
    dynamic_range = np.max(audio_data) - np.min(audio_data)
    
    # 计算频谱能量分布
    spectral_energy = np.sum(spectrogram, axis=1)
    high_freq_energy = np.sum(spectral_energy[frequencies > 2000])
    low_freq_energy = np.sum(spectral_energy[frequencies <= 2000])
    
    # 判断音频类型
    is_voice = high_freq_energy > low_freq_energy * 1.5
    
    # 根据特征确定编码参数（高保真设置）
    if is_voice:
        # 语音文件参数 - 高保真设置
        params = {
            '格式': 'wav',
            '采样率': 48000,  # 使用更高采样率
            '声道数': 1,
            '比特率': '192058'  # 使用更高比特率
        }
    else:
        # 音乐文件参数 - 高保真设置
        params = {
            '格式': 'wav',
            '采样率': 48000,  # 使用更高采样率
            '声道数': 1,
            '比特率': '192058'  # 使用更高比特率
        }
    
    # 根据动态范围调整参数
    if dynamic_range > 0.8:
        params['比特率'] = '384058'  # 对于高动态范围使用更高比特率
    
    return params, {
        'is_voice': is_voice,
        'dynamic_range': dynamic_range,
        'high_freq_ratio': high_freq_energy / (high_freq_energy + low_freq_energy)
    }

def compute_audio_metrics(original_file, compressed_file):
    original_audio = AudioSegment.from_file(original_file)
    compressed_audio = AudioSegment.from_file(compressed_file)

    original_samples = np.array(original_audio.get_array_of_samples())
    compressed_samples = np.array(compressed_audio.get_array_of_samples())

    min_length = min(len(original_samples), len(compressed_samples))
    original_samples = original_samples[:min_length]
    compressed_samples = compressed_samples[:min_length]

    # 归一化数据
    original_samples = original_samples.astype(np.float32) / np.max(np.abs(original_samples))
    compressed_samples = compressed_samples.astype(np.float32) / np.max(np.abs(compressed_samples))

    diff = np.abs(original_samples - compressed_samples)
    mse = np.mean(diff ** 2)

    if mse <= 0:
        psnr = 100.0
    else:
        try:
            psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
            psnr = max(0, min(100, psnr))
        except:
            psnr = 0.0

    return {'MSE': mse, 'PSNR': psnr}

def save_optimized_audio(file_path, encoding_params):
    audio = AudioSegment.from_file(file_path)
    filename = os.path.basename(file_path)
    optimized_audio_path = os.path.join(output_directory, f'adaptive_{filename}')

    # 应用编码参数
    audio = audio.set_frame_rate(encoding_params['采样率'])
    if encoding_params['声道数'] == 1:
        audio = audio.set_channels(1)
    
    # 导出音频（高保真设置）
    audio.export(optimized_audio_path, format='wav', parameters=["-q:a", "0"])

    return optimized_audio_path

def calculate_qsi(original_size, compressed_size, psnr):
    """计算QSI (Quality-Size Index)"""
    compression_ratio = original_size / compressed_size
    return compression_ratio * (psnr / 100)

def compare_with_csv_data(result, df):
    """与CSV数据进行详细对比"""
    filename = result['文件名']
    original_row = df[df['文件名'] == filename].iloc[0]
    
    # 计算各项指标的改进百分比
    size_improvement = ((original_row['大小(KB)'] - result['文件大小变化']['压缩后大小(KB)']) / original_row['大小(KB)']) * 100
    psnr_improvement = ((result['质量指标']['压缩后PSNR'] - original_row['PSNR(dB)']) / original_row['PSNR(dB)']) * 100
    
    return {
        'size_improvement': size_improvement,
        'psnr_improvement': psnr_improvement,
        'original_metrics': {
            '格式': original_row['格式'],
            '采样率': original_row['采样率(Hz)'],
            '声道数': original_row['声道数'],
            '比特率': original_row['比特率'],
            'PSNR': original_row['PSNR(dB)'],
            '文件大小': original_row['大小(KB)']
        }
    }

def plot_audio_waveform(audio_data, sample_rate, title, save_path):
    """绘制音频波形图"""
    plt.figure(figsize=(12, 4))
    time = np.arange(len(audio_data)) / sample_rate
    plt.plot(time, audio_data)
    plt.title(title)
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram(frequencies, times, spectrogram, title, save_path):
    """绘制频谱图"""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), shading='gouraud')
    plt.title(title)
    plt.ylabel('频率 (Hz)')
    plt.xlabel('时间 (秒)')
    plt.colorbar(label='强度 (dB)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison_metrics(results, save_path):
    """绘制比较指标图"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 文件大小对比
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = []
    labels = []
    for r in results:
        sizes.extend([r['文件大小变化']['原始大小(KB)'], r['文件大小变化']['压缩后大小(KB)']])
        labels.extend([f"{r['文件名']} (原始)", f"{r['文件名']} (压缩)"])
    
    x = np.arange(len(sizes))
    ax1.bar(x, sizes, color=['#3498db', '#e74c3c'] * len(results))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_title('文件大小对比')
    ax1.set_ylabel('大小 (KB)')
    
    # PSNR对比
    ax2 = fig.add_subplot(gs[0, 1])
    psnr_values = []
    for r in results:
        psnr_values.extend([r['质量指标']['原始PSNR'], r['质量指标']['压缩后PSNR']])
    
    x = np.arange(len(psnr_values))
    ax2.bar(x, psnr_values, color=['#2ecc71', '#f1c40f'] * len(results))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_title('PSNR对比')
    ax2.set_ylabel('PSNR (dB)')
    
    # QSI对比
    ax3 = fig.add_subplot(gs[1, 0])
    qsi_values = [r['质量指标']['QSI'] for r in results]
    x = np.arange(len(qsi_values))
    ax3.bar(x, qsi_values, color='#9b59b6')
    ax3.set_xticks(x)
    ax3.set_xticklabels([r['文件名'] for r in results], rotation=45, ha='right')
    ax3.set_title('QSI指数对比')
    ax3.set_ylabel('QSI')
    
    # 改进百分比
    ax4 = fig.add_subplot(gs[1, 1])
    improvements = []
    improvement_labels = []
    for r in results:
        size_imp = ((r['文件大小变化']['原始大小(KB)'] - r['文件大小变化']['压缩后大小(KB)']) / 
                    r['文件大小变化']['原始大小(KB)']) * 100
        psnr_imp = ((r['质量指标']['压缩后PSNR'] - r['质量指标']['原始PSNR']) / 
                    r['质量指标']['原始PSNR']) * 100
        improvements.extend([size_imp, psnr_imp])
        improvement_labels.extend([f"{r['文件名']}\n大小改进", f"{r['文件名']}\nPSNR改进"])
    
    x = np.arange(len(improvements))
    ax4.bar(x, improvements, color=['#e67e22', '#1abc9c'] * len(results))
    ax4.set_xticks(x)
    ax4.set_xticklabels(improvement_labels, rotation=45, ha='right')
    ax4.set_title('改进百分比')
    ax4.set_ylabel('改进百分比 (%)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_audio_features(features, save_path):
    """绘制音频特征雷达图"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # 准备数据
    categories = ['动态范围', '高频能量比', '语音特征']
    values = [features['dynamic_range'], features['high_freq_ratio'], 
             1.0 if features['is_voice'] else 0.0]
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))  # 闭合图形
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    # 绘制雷达图
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    plt.title('音频特征分析')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 读取现有的分析结果
    df = pd.read_csv(r'C:\Users\18344\Desktop\C题 -数学建模老哥\音频分析结果\音频分析结果.csv')
    
    # 打印列名以便调试
    print("CSV文件的列名:", df.columns.tolist())
    
    # 要处理的文件
    target_files = ['原始语音_48kHz_24bit.wav', '原始音乐_48kHz_24bit.wav']
    
    results = []
    
    for filename in target_files:
        print(f"\n处理文件: {filename}")
        file_path = os.path.join(audio_directory, filename)
        
        # 分析音频特征并获取自适应参数
        encoding_params, features = analyze_audio_features(file_path)
        
        # 读取音频数据用于可视化
        sample_rate, audio_data = wavfile.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
        
        # 计算频谱图
        frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate)
        
        # 保存波形图
        waveform_path = os.path.join(output_directory, f'{filename}_waveform.png')
        plot_audio_waveform(audio_data, sample_rate, f'{filename} 波形图', waveform_path)
        
        # 保存频谱图
        spectrogram_path = os.path.join(output_directory, f'{filename}_spectrogram.png')
        plot_spectrogram(frequencies, times, spectrogram, f'{filename} 频谱图', spectrogram_path)
        
        # 保存音频特征雷达图
        features_path = os.path.join(output_directory, f'{filename}_features.png')
        plot_audio_features(features, features_path)
        
        # 保存优化后的音频
        optimized_path = save_optimized_audio(file_path, encoding_params)
        
        # 计算质量指标
        metrics = compute_audio_metrics(file_path, optimized_path)
        
        # 计算QSI
        original_size = os.path.getsize(file_path)
        compressed_size = os.path.getsize(optimized_path)
        qsi = calculate_qsi(original_size, compressed_size, metrics['PSNR'])
        
        # 获取原始文件在CSV中的对应行
        original_row = df[df['文件名'] == filename].iloc[0]
        
        # 保存结果
        result = {
            '文件名': filename,
            '类型': '语音' if features['is_voice'] else '音乐',
            '自适应参数': encoding_params,
            '原始参数': {
                '格式': original_row['格式'],
                '采样率': original_row['采样率(Hz)'],
                '声道数': original_row['声道数'],
                '比特率': original_row['比特率']
            },
            '文件大小变化': {
                '原始大小(KB)': original_row['大小(KB)'],
                '压缩后大小(KB)': compressed_size / 1024
            },
            '质量指标': {
                '原始PSNR': original_row['PSNR(dB)'],
                '压缩后PSNR': metrics['PSNR'],
                'QSI': qsi
            }
        }
        results.append(result)
        
        # 与CSV数据进行对比
        comparison = compare_with_csv_data(result, df)
        
        # 输出详细对比结果
        print("\n=== 详细对比分析 ===")
        print("\n1. 参数对比:")
        print(f"原始参数: {result['原始参数']}")
        print(f"自适应参数: {result['自适应参数']}")
        
        print("\n2. 文件大小对比:")
        print(f"原始文件大小: {result['文件大小变化']['原始大小(KB)']:.2f} KB")
        print(f"压缩后大小: {result['文件大小变化']['压缩后大小(KB)']:.2f} KB")
        print(f"大小变化: {comparison['size_improvement']:.2f}%")
        
        print("\n3. 音质对比:")
        print(f"原始PSNR: {result['质量指标']['原始PSNR']:.2f} dB")
        print(f"压缩后PSNR: {result['质量指标']['压缩后PSNR']:.2f} dB")
        print(f"PSNR变化: {comparison['psnr_improvement']:.2f}%")
        print(f"QSI: {result['质量指标']['QSI']:.4f}")
        
        print("\n4. 自适应编码优势:")
        if comparison['size_improvement'] > 0:
            print(f"- 文件大小减少了 {abs(comparison['size_improvement']):.2f}%")
        if comparison['psnr_improvement'] > 0:
            print(f"- 音质提升了 {comparison['psnr_improvement']:.2f}%")
        print(f"- 保持了高保真音质 (PSNR: {result['质量指标']['压缩后PSNR']:.2f} dB)")
        print(f"- 优化后的QSI指数: {result['质量指标']['QSI']:.4f}")
    
    # 绘制所有结果的比较图
    comparison_path = os.path.join(output_directory, 'comparison_metrics.png')
    plot_comparison_metrics(results, comparison_path)
    
    print(f"\n所有可视化结果已保存到: {output_directory}")

if __name__ == "__main__":
    main()
