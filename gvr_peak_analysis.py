"""
GVR Peak Analysis Module
This module extracts and visualizes the peak detection functionality from read_data.py
for easier debugging and troubleshooting.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
import h5py
import os
import re
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import json
import random
from scipy.signal import find_peaks 


# ============================================================
# 新增：ANSYS数据加载器
# ============================================================
class ANSYSDataLoader:
    """
    从外部目录加载ANSYS仿真加速度数据
    """
    def __init__(self, data_root: str = './ansys_data', num_degrees: int = 15, num_steps: int = 30000):
        """
        Args:
            data_root: ansys_data根目录
            num_degrees: 传感器/通道数量 (15)
            num_steps: 采样点数量 (30000)
        """
        self.data_root = data_root
        self.num_degrees = num_degrees
        self.num_steps = num_steps
        self.dt = 0.001  # 根据tree.txt中的时间步长推断
        self.healthy_folder_name = '无损'
        
        # 检查目录是否存在
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"数据目录不存在: {self.data_root}")

    def _parse_folder_name(self, folder_name: str):
        """
        从文件夹名称解析损伤信息
        例如: "3号30%损伤" -> {3: 0.3}
        例如: "4号40%+8号40%损伤" -> {4: 0.4, 8: 0.4}
        """
        damaged_dofs = []
        severity_ratios = []
        
        # 移除"损伤"后缀
        temp_name = folder_name.replace("损伤", "")
        
        # 分割多个损伤 (如 "4号40%+8号40%")
        segments = temp_name.split('+')
        
        for seg in segments:
            # 正则匹配 "数字号数字%"
            match = re.search(r'(\d+)号(\d+)%', seg)
            if match:
                dof = int(match.group(1))
                severity = float(match.group(2)) / 100.0
                damaged_dofs.append(dof)
                severity_ratios.append(severity)
        
        return damaged_dofs, severity_ratios

    def load_single_file(self, file_path: str):
        """
        读取单个ANSYS导出的txt文件
        格式: 索引 时间 加速度
        跳过第一行表头
        """
        try:
            # 使用numpy读取，跳过第一行
            # 假设列之间用空白字符分隔
            data = np.loadtxt(file_path, skiprows=1)
            # data shape: (N, 3) or (N, 2)
            # 我们只需要加速度值 (最后一列)
            # 如果数据不足30000点，进行截断或填充? 假设都是完整的
            acc = data[:, -1]
            
            # 确保长度一致
            if len(acc) > self.num_steps:
                acc = acc[:self.num_steps]
            elif len(acc) < self.num_steps:
                # 如果数据不足，进行边缘填充
                acc = np.pad(acc, (0, self.num_steps - len(acc)), 'edge')
                
            return acc
        except Exception as e:
            print(f"读取文件错误 {file_path}: {e}")
            return np.zeros(self.num_steps)

    def load_scenario(self, folder_name: str) -> Dict:
        """
        加载特定场景的所有通道数据
        """
        folder_path = os.path.join(self.data_root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"目录不存在，跳过: {folder_path}")
            return None

        # 初始化响应矩阵
        response = np.zeros((self.num_steps, self.num_degrees))
        
        # 遍历1到15个通道
        # 假设文件名格式为: "文件夹名1.txt", "文件夹名2.txt" ...
        # 或者是 "无损1.txt" 等
        found_files = 0
        for i in range(1, self.num_degrees + 1):
            # 尝试匹配文件名，例如 "无损1.txt" 或 "3号30%损伤1.txt"
            # 注意：tree.txt中显示文件名包含文件夹名前缀
            possible_names = [
                f"{folder_name}{i}.txt",
                f"{i}.txt"
            ]
            
            file_path = None
            for name in possible_names:
                full_path = os.path.join(folder_path, name)
                if os.path.exists(full_path):
                    file_path = full_path
                    break
            
            if file_path:
                response[:, i-1] = self.load_single_file(file_path)
                found_files += 1
            else:
                print(f"警告: 场景 {folder_name} 中找不到通道 {i} 的数据文件")

        if found_files == 0:
            print(f"错误: 场景 {folder_name} 中没有找到任何有效数据文件")
            return None

        # 解析标签
        if folder_name == self.healthy_folder_name:
            damaged_dofs = []
            severity_ratios = []
            damage_class = 0
        else:
            damaged_dofs, severity_ratios = self._parse_folder_name(folder_name)
            damage_class = 1 if len(damaged_dofs) > 0 else 0

        return {
            'acceleration': response,
            'damaged_dofs': damaged_dofs,
            'severity_ratios': severity_ratios,
            'damage_class': damage_class,
            'folder_name': folder_name
        }


def analyze_gvr_peaks(damaged_signal: np.ndarray, 
                      healthy_signal: np.ndarray,
                      dt: float = 0.001,
                      window_length: int = 3000,
                      step_size: int = 50,
                      cutoff_freq: float = 5.0,
                      prob_threshold: float = 5.0,
                      visualize: bool = True,
                      output_dir: str = './gvr_analysis_output'):
    """
    Analyze GVR peaks for damage detection
    
    Args:
        damaged_signal: Acceleration signal from damaged structure
        healthy_signal: Acceleration signal from healthy structure
        dt: Time step
        window_length: Length of analysis window
        step_size: Step size between windows
        cutoff_freq: Low-pass filter cutoff frequency
        prob_threshold: Probability threshold for damage classification
        visualize: Whether to generate visualization plots
        output_dir: Directory to save plots
    
    Returns:
        tuple: (auto_labels, probabilities, DI_double_prime, analysis_data)
    """
    
    # Create output directory if needed
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize filter
    nyquist = 0.5 / dt
    b, a = signal.butter(4, cutoff_freq / nyquist, btype='low')
    
    # 1. Preprocessing: filter signals
    filtered_damaged = signal.filtfilt(b, a, damaged_signal, axis=0)
    filtered_healthy = signal.filtfilt(b, a, healthy_signal, axis=0)
    
    n_channels = damaged_signal.shape[1]
    num_windows = (filtered_damaged.shape[0] - window_length) // step_size + 1
    
    # 2. Calculate DI_series (must be computed in loop)
    DI_series = np.zeros((num_windows, n_channels))
    for win_idx in range(num_windows):
        start = win_idx * step_size
        end = start + window_length
        
        win_damaged = filtered_damaged[start:end]
        win_healthy = filtered_healthy[start:end]
        
        # Paper formula (8)
        for ch in range(n_channels):
            numerator = np.sum((win_damaged[:, ch] - win_healthy[:, ch]) ** 2)
            denominator = np.sum(win_healthy[:, ch] ** 2) + 1e-10
            DI_series[win_idx, ch] = np.sqrt(numerator) / np.sqrt(denominator)

    # Spatial first derivative: calculate difference between adjacent sensors' DI
    # Logic: DI[i] - DI[i-1]
    DI_prime = np.zeros_like(DI_series)
    DI_prime[:, 1:] = DI_series[:, 1:] - DI_series[:, :-1]
    
    # Spatial second derivative: calculate rate of change of spatial gradient (detects peaks)
    # Logic: abs((DI[i]-DI[i-1]) - (DI[i-1]-DI[i-2]))
    DI_double_prime = np.zeros_like(DI_prime)
    # Note: After taking first derivative then second derivative, effective length is (n_channels - 2)
    DI_double_prime[:, 1:] = np.abs(DI_prime[:, 1:] - DI_prime[:, :-1])
    
    # 4. Count fault occurrences across channels
    fault_occurrences = np.zeros(n_channels)
    
    for win_idx in range(num_windows):
        # Get spatial GVR distribution for current window
        spatial_gvr = DI_double_prime[win_idx]
        
        if np.max(spatial_gvr) > 1e-8:
            prominence_threshold = np.max(spatial_gvr) * 0.1
        else:
            prominence_threshold = 0
        
        # 2. Find all peaks that meet conditions
        # distance=2: Prevents adjacent sensors (like 4 and 5) from being identified as two separate damage points
        peaks, properties = find_peaks(
            spatial_gvr, 
            prominence=prominence_threshold, 
            distance=2,
        ) # e.g., above mean+2*std
        
        # 3. Count
        for ch in peaks:
            fault_occurrences[ch] += 1
    
    # 5. Calculate damage probability
    probabilities = (fault_occurrences / num_windows) * 100
    
    # 6. Generate labels based on probability threshold
    auto_labels = (probabilities > prob_threshold).astype(int)
    
    # Visualization if requested
    if visualize:
        visualize_gvr_analysis(
            DI_series, DI_prime, DI_double_prime, 
            fault_occurrences, probabilities, auto_labels,
            num_windows, output_dir
        )
    
    analysis_data = {
        'DI_series': DI_series,
        'DI_prime': DI_prime,
        'DI_double_prime': DI_double_prime,
        'fault_occurrences': fault_occurrences,
        'num_windows': num_windows,
        'prominence_threshold': prominence_threshold if 'prominence_threshold' in locals() else 0
    }
    
    return auto_labels, probabilities, DI_double_prime, analysis_data


def visualize_gvr_analysis(DI_series, DI_prime, DI_double_prime, 
                          fault_occurrences, probabilities, auto_labels,
                          num_windows, output_dir):
    """
    Visualize GVR analysis results
    """
    n_channels = DI_series.shape[1]
    
    # Plot 1: DI Series over time for all channels
    plt.figure(figsize=(15, 10))
    for ch in range(min(15, n_channels)):  # Only plot first 15 channels to avoid overcrowding
        plt.subplot(5, 3, ch+1)
        plt.plot(DI_series[:, ch], alpha=0.7, label=f'Channel {ch+1}')
        plt.title(f'DI Series - Channel {ch+1}')
        plt.xlabel('Window Index')
        plt.ylabel('DI Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'di_series_all_channels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: DI Prime (first derivative) over time
    plt.figure(figsize=(15, 10))
    for ch in range(min(15, n_channels)):
        plt.subplot(5, 3, ch+1)
        plt.plot(DI_prime[:, ch], alpha=0.7, color='orange', label=f'Channel {ch+1}')
        plt.title(f'DI Prime - Channel {ch+1}')
        plt.xlabel('Window Index')
        plt.ylabel("DI'")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'di_prime_all_channels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: DI Double Prime (second derivative) over time
    plt.figure(figsize=(15, 10))
    for ch in range(min(15, n_channels)):
        plt.subplot(5, 3, ch+1)
        plt.plot(DI_double_prime[:, ch], alpha=0.7, color='red', label=f'Channel {ch+1}')
        plt.title(f'DI Double Prime - Channel {ch+1}')
        plt.xlabel('Window Index')
        plt.ylabel("DI''")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'di_double_prime_all_channels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fault occurrences per channel
    axes[0, 0].bar(range(len(fault_occurrences)), fault_occurrences)
    axes[0, 0].set_title('Fault Occurrences Per Channel')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Damage probabilities per channel
    axes[0, 1].bar(range(len(probabilities)), probabilities)
    axes[0, 1].set_title('Damage Probability Per Channel (%)')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Probability (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Auto labels
    colors = ['green' if label == 0 else 'red' for label in auto_labels]
    axes[1, 0].bar(range(len(auto_labels)), auto_labels, color=colors)
    axes[1, 0].set_title('Auto Labels (0=Healthy, 1=Damaged)')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Label')
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall statistics
    stats_text = f"""Statistics:
Total Windows: {num_windows}
Channels: {len(fault_occurrences)}
Avg Fault Occurrences: {np.mean(fault_occurrences):.2f}
Avg Probability: {np.mean(probabilities):.2f}%
Damaged Channels: {np.sum(auto_labels)} / {len(auto_labels)}
"""
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Heatmap of DI Double Prime
    plt.figure(figsize=(12, 8))
    im = plt.imshow(DI_double_prime.T, aspect='auto', origin='lower', 
                    extent=[0, num_windows, 0, n_channels], cmap='viridis')
    plt.colorbar(im, label="DI'' Value")
    plt.title('Heatmap of DI Double Prime Over Time and Channels')
    plt.xlabel('Window Index')
    plt.ylabel('Channel')
    plt.savefig(os.path.join(output_dir, 'di_double_prime_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 在gvr_peak_analysis.py文件末尾添加以下代码

if __name__ == '__main__':
    # 设置参数
    data_root = './ansys_data'  # ANSYS数据根目录
    output_dir = './gvr_analysis_output'  # 输出目录
    num_degrees = 15  # 传感器/通道数量
    num_steps = 30000  # 采样点数量
    
    # 分析参数
    window_length = 3000  # 分析窗口长度
    step_size = 50  # 步长
    cutoff_freq = 6.0  # 低通滤波截止频率
    prob_threshold = 15.0  # 损伤分类概率阈值
    
    try:
        # 初始化数据加载器
        print("初始化ANSYS数据加载器...")
        loader = ANSYSDataLoader(data_root=data_root, num_degrees=num_degrees, num_steps=num_steps)
        
        # 加载健康状态数据
        print("加载健康状态数据...")
        healthy_data = loader.load_scenario(loader.healthy_folder_name)
        if healthy_data is None:
            raise ValueError("无法加载健康状态数据")
        
        # 获取所有损伤场景
        print("获取损伤场景列表...")
        damage_scenarios = []
        for item in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, item)) and item != loader.healthy_folder_name:
                damage_scenarios.append(item)
        
        if not damage_scenarios:
            raise ValueError("未找到任何损伤场景数据")
        
        print(f"找到 {len(damage_scenarios)} 个损伤场景")
        
        # 分析每个损伤场景
        for scenario in damage_scenarios:
            print(f"\n分析场景: {scenario}")
            
            # 加载损伤数据
            damaged_data = loader.load_scenario(scenario)
            if damaged_data is None:
                print(f"警告: 无法加载场景 {scenario} 的数据，跳过")
                continue
            
            # 创建场景特定的输出目录
            scenario_output_dir = os.path.join(output_dir, scenario)
            
            # 执行GVR峰值分析
            print("执行GVR峰值分析...")
            auto_labels, probabilities, DI_double_prime, analysis_data = analyze_gvr_peaks(
                damaged_signal=damaged_data['acceleration'],
                healthy_signal=healthy_data['acceleration'],
                dt=loader.dt,
                window_length=window_length,
                step_size=step_size,
                cutoff_freq=cutoff_freq,
                prob_threshold=prob_threshold,
                visualize=True,
                output_dir=scenario_output_dir
            )
            
            # 保存分析结果
            result = {
                'scenario': scenario,
                'damaged_dofs': damaged_data['damaged_dofs'],
                'severity_ratios': damaged_data['severity_ratios'],
                'auto_labels': auto_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'analysis_parameters': {
                    'window_length': window_length,
                    'step_size': step_size,
                    'cutoff_freq': cutoff_freq,
                    'prob_threshold': prob_threshold
                }
            }
            
            # 保存结果为JSON文件
            result_file = os.path.join(scenario_output_dir, 'analysis_result.json')
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=4)
            
            print(f"分析完成，结果已保存到: {scenario_output_dir}")
            
            # 打印每个通道的受损概率
            print("\n各通道受损概率:")
            print("通道\t概率(%)")
            for i, prob in enumerate(probabilities):
                print(f"{i+1}\t{prob:.2f}")

            # 打印简要结果
            detected_dofs = [i+1 for i, label in enumerate(auto_labels) if label == 1]
            print(f"检测到的损伤通道: {detected_dofs}")
            print(f"实际损伤通道: {damaged_data['damaged_dofs']}")
            
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
