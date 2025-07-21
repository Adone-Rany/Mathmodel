import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, wiener
from scipy.io.wavfile import write
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 音频文件路径
audio_paths = [
    r"C:\Users\18344\Desktop\C题 -数学建模老哥\附件2\part1.wav",
    r"C:\Users\18344\Desktop\C题 -数学建模老哥\附件2\part2.wav"
]

# 创建结果保存目录
output_directory = "问题4结果"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 创建子目录
for subdir in ["音频", "图像", "数据"]:
    subdir_path = os.path.join(output_directory, subdir)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)


# 读取音频文件
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


# 时频分析：计算短时傅里叶变换（STFT）
def time_frequency_analysis(y, sr):
    D = librosa.stft(y)  # 计算短时傅里叶变换（STFT）
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # 转换为分贝表示
    return D, S_db


# 去噪策略：背景噪声去除 - 改进版
def background_noise_removal(D):
    magnitude, phase = librosa.magphase(D)
    # 使用更复杂的噪声估计方法
    noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
    # 应用软阈值
    gain = np.maximum(0, magnitude - noise_profile) / (magnitude + 1e-8)
    magnitude_denoised = magnitude * gain
    return magnitude_denoised * phase


# 去噪策略：突发噪声去除 - 改进版
def impulsive_noise_removal(y):
    # 结合中值滤波和维纳滤波
    y_med = medfilt(y, kernel_size=5)
    
    # 使用更稳健的维纳滤波实现
    y_wiener = np.zeros_like(y)
    window_size = 5
    half_window = window_size // 2
    
    for i in range(len(y)):
        start = max(0, i - half_window)
        end = min(len(y), i + half_window + 1)
        window = y[start:end]
        
        # 计算局部方差，避免除零
        local_var = np.var(window)
        if local_var < 1e-10:  # 如果方差太小，直接使用中值滤波结果
            y_wiener[i] = y_med[i]
        else:
            # 计算局部均值
            local_mean = np.mean(window)
            # 计算局部信噪比
            snr = local_var / (local_var + 1e-10)  # 添加小量避免除零
            # 应用维纳滤波
            y_wiener[i] = local_mean + snr * (y[i] - local_mean)
    
    # 自适应混合
    diff_med = np.abs(y - y_med)
    diff_wiener = np.abs(y - y_wiener)
    total_diff = diff_med + diff_wiener + 1e-10  # 添加小量避免除零
    
    alpha = diff_med / total_diff
    return alpha * y_med + (1 - alpha) * y_wiener


# 去噪策略：带状噪声去除 - 改进版
def band_noise_removal(D, sr, low_freq=1000, high_freq=5000):
    freqs = librosa.fft_frequencies(sr=sr)
    low_bin = np.searchsorted(freqs, low_freq)
    high_bin = np.searchsorted(freqs, high_freq)
    
    # 使用平滑过渡而不是直接置零
    transition_width = 50
    for i in range(low_bin, high_bin):
        if i < low_bin + transition_width:
            D[:, i] *= (i - low_bin) / transition_width
        elif i > high_bin - transition_width:
            D[:, i] *= (high_bin - i) / transition_width
        else:
            D[:, i] *= 0.1  # 降低而不是完全消除
    
    return D


# 信噪比计算
def calculate_snr(original, denoised):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - denoised) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# 计算音频特征
def calculate_audio_features(y, sr):
    # 计算RMS能量
    rms = np.sqrt(np.mean(y**2))
    
    # 计算过零率
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(y))))
    
    # 计算频谱质心
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # 计算频谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    
    # 计算频谱平坦度
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    spectral_flatness_mean = np.mean(spectral_flatness)
    
    # 计算频谱滚降
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    
    return {
        "RMS能量": rms,
        "过零率": zero_crossings,
        "频谱质心": spectral_centroid_mean,
        "频谱带宽": spectral_bandwidth_mean,
        "频谱平坦度": spectral_flatness_mean,
        "频谱滚降": spectral_rolloff_mean
    }


# 保存音频文件
def save_audio(y, sr, output_path):
    write(output_path, sr, y)


# 新的可视化函数：音频分析
def visualize_audio_analysis(y, D, sr, title, output_path=None):
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # 1. 波形图
    ax1 = plt.subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, alpha=0.7)
    ax1.set_title('波形图')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('振幅')
    ax1.grid(True)
    
    # 2. 频谱图
    ax2 = plt.subplot(gs[1, 0])
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    im = ax2.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('频谱图')
    ax2.set_xlabel('帧')
    ax2.set_ylabel('频率')
    plt.colorbar(im, ax=ax2, format="%+2.0f dB")
    
    # 3. 频率分布
    ax3 = plt.subplot(gs[1, 1])
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.mean(np.abs(D), axis=1)
    ax3.plot(freqs[:len(mag)], mag, alpha=0.7)
    ax3.set_title('频率分布')
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('幅度')
    ax3.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


# 生成数据表格
def generate_data_tables(results_data, output_path):
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 保存为CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return df


# 主函数
def process_audio_files():
    all_results = []
    
    for audio_path in audio_paths:
        print(f"处理音频文件: {audio_path}")
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        y, sr = load_audio(audio_path)
        
        # 计算原始音频特征
        original_features = calculate_audio_features(y, sr)
        
        # 时频分析
        D, S_db = time_frequency_analysis(y, sr)
        
        # 保存原始音频分析图
        visualize_audio_analysis(
            y, D, sr,
            f"音频{base_filename} - 去噪前分析",
            os.path.join(output_directory, "图像", f"音频{base_filename}_去噪前分析.png")
        )
        
        # 应用所有去噪方法
        # 1. 背景噪声去除
        D_denoised = background_noise_removal(D)
        
        # 2. 带状噪声去除
        D_denoised = band_noise_removal(D_denoised, sr)
        
        # 3. 转换回时域
        y_denoised = librosa.istft(D_denoised)
        
        # 4. 突发噪声去除
        y_denoised = impulsive_noise_removal(y_denoised)
        
        # 计算去噪后音频特征
        denoised_features = calculate_audio_features(y_denoised, sr)
        
        # 计算SNR
        snr = calculate_snr(y, y_denoised)
        
        # 保存去噪后的音频文件
        output_audio = os.path.join(output_directory, "音频", f"denoised_{os.path.basename(audio_path)}")
        save_audio(y_denoised, sr, output_audio)
        
        print(f"去噪后的音频文件已保存为：{output_audio}")
        
        # 保存去噪后音频分析图
        D_denoised, _ = time_frequency_analysis(y_denoised, sr)
        visualize_audio_analysis(
            y_denoised, D_denoised, sr,
            f"音频{base_filename} - 去噪后分析",
            os.path.join(output_directory, "图像", f"音频{base_filename}_去噪后分析.png")
        )
        
        # 收集结果数据
        file_results = {
            "文件名": base_filename,
            "信噪比(SNR)": f"{snr:.2f} dB",
            "原始RMS能量": f"{original_features['RMS能量']:.6f}",
            "去噪后RMS能量": f"{denoised_features['RMS能量']:.6f}",
            "RMS能量变化率": f"{((denoised_features['RMS能量'] - original_features['RMS能量']) / original_features['RMS能量'] * 100):.2f}%",
            "原始过零率": f"{original_features['过零率']:.2f}",
            "去噪后过零率": f"{denoised_features['过零率']:.2f}",
            "过零率变化率": f"{((denoised_features['过零率'] - original_features['过零率']) / original_features['过零率'] * 100):.2f}%",
            "原始频谱质心": f"{original_features['频谱质心']:.2f} Hz",
            "去噪后频谱质心": f"{denoised_features['频谱质心']:.2f} Hz",
            "频谱质心变化率": f"{((denoised_features['频谱质心'] - original_features['频谱质心']) / original_features['频谱质心'] * 100):.2f}%",
            "原始频谱带宽": f"{original_features['频谱带宽']:.2f} Hz",
            "去噪后频谱带宽": f"{denoised_features['频谱带宽']:.2f} Hz",
            "频谱带宽变化率": f"{((denoised_features['频谱带宽'] - original_features['频谱带宽']) / original_features['频谱带宽'] * 100):.2f}%",
            "原始频谱平坦度": f"{original_features['频谱平坦度']:.6f}",
            "去噪后频谱平坦度": f"{denoised_features['频谱平坦度']:.6f}",
            "频谱平坦度变化率": f"{((denoised_features['频谱平坦度'] - original_features['频谱平坦度']) / original_features['频谱平坦度'] * 100):.2f}%",
            "原始频谱滚降": f"{original_features['频谱滚降']:.2f} Hz",
            "去噪后频谱滚降": f"{denoised_features['频谱滚降']:.2f} Hz",
            "频谱滚降变化率": f"{((denoised_features['频谱滚降'] - original_features['频谱滚降']) / original_features['频谱滚降'] * 100):.2f}%"
        }
        
        all_results.append(file_results)
    
    # 生成数据表格
    data_table_path = os.path.join(output_directory, "数据", "音频处理结果.csv")
    df = generate_data_tables(all_results, data_table_path)
    
    print(f"数据表格已保存到: {data_table_path}")
    
    return df


# 运行主函数
results_df = process_audio_files()
