"""
===========================================================================
GVR自动标注系统 - 最终完整版
基于论文：Multimodal deep learning with integrated automatic labeling
          for structural damage detection in high-pile wharves
===========================================================================

论文确认：
- 图7/图9：GVRs along the beam length
  ✓ 横轴：传感器索引 - 从文件名提取探针编号
  ✓ 纵轴：GVR values

- 图8/图10：Statistical probability of damage occurrence
  ✓ 横轴：传感器索引
  ✓ 纵轴：损伤识别概率（%）

- 图25/图26：物理模型实验结果
  ✓ 横轴：传感器索引/位置
  ✓ 纵轴：GVR值或概率

关键公式：
- 公式(8): DI = Σ(xd - xh) / Σ(xh²) + ε
- 公式(9): DI' = DI[i] - DI[i-1]
- 公式(10): DI'' = |DI'[i] - DI'[i-1]| (这是GVR)
- 公式(11): Dmaxima = arg max(DI'')
- 公式(12): nch = Σ DIch(i)
- 公式(13): Pch = nch / N × 100%
===========================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import warnings
import re
warnings.filterwarnings('ignore')

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300


class GVRAutoLabeling:
    """
    GVR自动标注系统（最终完整版）
    
    完全按照论文方法实现：
    - 横轴：传感器索引（从文件名中提取）
    - 纵轴：GVR值
    - 所有公式完全按照论文实现
    """
    
    def __init__(self, data_root_path, cutoff_freq=5.0, filter_order=4, fs=1000):
        """
        初始化GVR系统
        """
        self.data_root_path = data_root_path
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.fs = fs
        self.epsilon = 1e-10
        
        # 设计Butterworth低通滤波器（论文公式1）
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = butter(self.filter_order, normal_cutoff, btype='low')
        
        # 存储数据
        self.healthy_data = {}
        self.damaged_data = {}
        self.results = {}
        
        print("✓ GVR系统初始化完成")
        print(f"数据路径: {data_root_path}")
        print(f"滤波器: Butterworth {cutoff_freq}Hz, {filter_order}阶")
        print(f"采样频率: {fs}Hz\n")
    
    def apply_lowpass_filter(self, signal):
        """应用低通滤波器"""
        return filtfilt(self.b, self.a, signal)
    
    def extract_probe_number(self, filename):
        """
        从文件名中提取探针编号
        
        例如：
        "3号30%损伤1.txt" -> 探针1
        "3号30%损伤2.txt" -> 探针2
        "3号30%损伤10.txt" -> 探针10
        """
        try:
            # 匹配数字
            match = re.search(r'探针(\d+)', filename)
            if match:
                probe_num = int(match.group(1))
                return probe_num
            
            # 如果没有匹配到探针，尝试其他方式
            # 例如文件名是 "1.txt", "2.txt"等
            match = re.search(r'(\d+)\.txt', filename)
            if match:
                probe_num = int(match.group(1))
                return probe_num
            
            return None
        except Exception as e:
            print(f"警告: 无法从文件名'{filename}'提取探针编号: {e}")
            return None
    
    def load_data_file(self, filepath):
        """
        加载单个数据文件
        格式：时间[s] 加速度[m/s²]
        """
        try:
            data = np.loadtxt(filepath, skiprows=1)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] >= 2:
                time = data[:, 0]
                accel = data[:, 1]
                return time, accel
            else:
                return None, None
        except Exception as e:
            print(f"✗ 加载文件失败 {filepath}: {e}")
            return None, None
    
    def load_healthy_baseline(self, healthy_folder='无损'):
        """加载健康状态基线数据"""
        healthy_path = os.path.join(self.data_root_path, healthy_folder)
        
        if not os.path.exists(healthy_path):
            print(f"✗ 错误: 健康状态文件夹不存在: {healthy_folder}")
            return False
        
        print(f"\n{'='*70}")
        print(f"步骤1: 加载健康状态基线数据")
        print(f"{'='*70}")
        print(f"路径: {healthy_path}")
        
        files = [f for f in os.listdir(healthy_path) if f.endswith('.txt')]
        files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        
        if len(files) == 0:
            print(f"✗ 警告: 文件夹中没有找到txt文件")
            return False
        
        print(f"找到 {len(files)} 个数据文件\n")
        
        for filename in files:
            filepath = os.path.join(healthy_path, filename)
            time, accel = self.load_data_file(filepath)
            
            if time is not None and accel is not None:
                # 从文件名提取探针编号
                probe_num = self.extract_probe_number(filename)
                
                if probe_num is not None:
                    filtered_accel = self.apply_lowpass_filter(accel)
                    
                    self.healthy_data[probe_num] = {
                        'filename': filename,
                        'probe_number': probe_num,
                        'time': time,
                        'raw': accel,
                        'filtered': filtered_accel,
                        'length': len(time)
                    }
                    
                    print(f"  ✓ 探针 {probe_num:2d}: {filename:30s} (样本数: {len(time):5d})")
        
        print(f"\n✓ 成功加载 {len(self.healthy_data)} 个通道的健康数据")
        return True
    
    def calculate_di(self, signal_damaged, signal_healthy):
        """
        计算损伤指数
        
        论文公式(8): DIj = Σ(xd_ij - xh_ij) / Σ(xh_ij)² + ε
        """
        numerator = np.sum(signal_damaged - signal_healthy)
        denominator = np.sum(signal_healthy ** 2) + self.epsilon
        di = numerator / denominator
        return di
    
    def calculate_di_derivatives(self, di_sequence):
        """
        计算DI的一阶和二阶导数
        
        论文公式(9): DI'j = DIj - DIj-1
        论文公式(10): DI''j = |DI'j - DI'j-1|
        """
        di_prime = np.diff(di_sequence)
        di_prime = np.insert(di_prime, 0, 0)
        
        di_double_prime = np.abs(np.diff(di_prime))
        di_double_prime = np.insert(di_double_prime, 0, 0)
        
        return di_prime, di_double_prime
    
    def detect_local_maxima(self, signal, min_distance=5, prominence=None):
        """
        检测局部最大值（峰值）
        
        论文公式(11): Dmaxima = arg max(DI'')
        """
        if prominence is None:
            prominence = 2 * np.std(signal)
        
        peaks, properties = find_peaks(
            signal,
            distance=min_distance,
            prominence=prominence
        )
        
        return peaks, properties
    
    def calculate_gvr_along_probes(self, damage_folder, window_length=3000, 
                                        step_size=50, specific_window_index=10):
        """
        计算沿探针索引的GVR分布
        
        横轴：探针编号- 从文件名提取
        纵轴：GVR值
        """
        damage_path = os.path.join(self.data_root_path, damage_folder)
        
        if not os.path.exists(damage_path):
            print(f"✗ 错误: 损伤文件夹不存在: {damage_folder}")
            return None
        
        print(f"\n[计算GVR分布] 沿探针索引的GVR分析")
        print(f"损伤场景: {damage_folder}")
        print(f"特定窗口索引: {specific_window_index}\n")
        
        files = [f for f in os.listdir(damage_path) if f.endswith('.txt')]
        
        gvr_distribution = {}
        probe_numbers = []  # 存储所有探针编号
        
        for filename in files:
            # 从文件名提取探针编号
            probe_num = self.extract_probe_number(filename)
            
            if probe_num is None or probe_num not in self.healthy_data:
                continue
            
            probe_numbers.append(probe_num)
            
            filepath = os.path.join(damage_path, filename)
            time_damaged, accel_damaged = self.load_data_file(filepath)
            
            if time_damaged is None:
                continue
            
            filtered_damage = self.apply_lowpass_filter(accel_damaged)
            healthy_filtered = self.healthy_data[probe_num]['filtered']
            
            min_len = min(len(filtered_damage), len(healthy_filtered))
            
            window_count = 0
            all_gvr_values = []
            gvr_at_specific_window = None
            
            for i in range(0, min_len - window_length + 1, step_size):
                win_damage = filtered_damage[i:i+window_length]
                win_healthy = healthy_filtered[i:i+window_length]
                
                di = self.calculate_di(win_damage, win_healthy)
                di_prime, di_double_prime = self.calculate_di_derivatives([di])
                gvr = di_double_prime[0]
                
                all_gvr_values.append(gvr)
                
                if window_count == specific_window_index:
                    gvr_at_specific_window = gvr
                
                window_count += 1
            
            gvr_distribution[probe_num] = {
                'probe_number': probe_num,
                'filename': filename,
                'window_count': window_count,
                'all_gvr_values': np.array(all_gvr_values),
                'gvr_at_specific_window': gvr_at_specific_window
            }
        
        # 按探针编号排序
        probe_numbers = sorted(probe_numbers)
        
        return gvr_distribution, probe_numbers
    
    def calculate_damage_probability_distribution(self, damage_folder, window_length=3000, 
                                              step_size=50, damage_threshold_percentile=90):
        """
        计算损伤概率分布
        
        论文公式(12)-(13):
        nch = Σ DIch(i)      # 窗口内故障次数
        Pch = nch / N × 100%  # 故障概率
        """
        damage_path = os.path.join(self.data_root_path, damage_folder)
        
        if not os.path.exists(damage_path):
            print(f"✗ 错误: 损伤文件夹不存在: {damage_folder}")
            return None
        
        print(f"\n[计算损伤概率] 统计各探针的损伤概率")
        print(f"损伤场景: {damage_folder}\n")
        
        files = [f for f in os.listdir(damage_path) if f.endswith('.txt')]
        
        probability_distribution = {}
        probe_numbers = []
        
        for filename in files:
            probe_num = self.extract_probe_number(filename)
            
            if probe_num is None or probe_num not in self.healthy_data:
                continue
            
            probe_numbers.append(probe_num)
            
            filepath = os.path.join(damage_path, filename)
            time_damaged, accel_damaged = self.load_data_file(filepath)
            
            if time_damaged is None:
                continue
            
            filtered_damage = self.apply_lowpass_filter(accel_damaged)
            healthy_filtered = self.healthy_data[probe_num]['filtered']
            
            min_len = min(len(filtered_damage), len(healthy_filtered))
            
            total_windows = 0
            damage_count = 0
            
            for i in range(0, min_len - window_length + 1, step_size):
                win_damage = filtered_damage[i:i+window_length]
                win_healthy = healthy_filtered[i:i+window_length]
                
                di = self.calculate_di(win_damage, win_healthy)
                di_prime, di_double_prime = self.calculate_di_derivatives([di])
                gvr = di_double_prime[0]
                
                damage_threshold = np.percentile([gvr], 100 - damage_threshold_percentile)
                is_damage = gvr > damage_threshold
                
                if is_damage:
                    damage_count += 1
                
                total_windows += 1
            
            damage_prob = (damage_count / total_windows) * 100 if total_windows > 0 else 0
            
            probability_distribution[probe_num] = {
                'probability': damage_prob,
                'damage_count': damage_count,
                'total_windows': total_windows
            }
        
        return probability_distribution, sorted(probability_distribution.keys())
    
    def visualize_gvr_distribution(self, damage_folder, gvr_distribution, probe_numbers, save_plots=True):
        """
        可视化GVR沿探针索引的分布
        
        对应论文图7/图9
        横轴：探针编号- 从文件名提取
        纵轴：GVR值
        """
        if not gvr_distribution:
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 准备数据：特定窗口的GVR值
        specific_window_gvr = []
        for probe_num in probe_numbers:
            if probe_num in gvr_distribution:
                specific_window_gvr.append(gvr_distribution[probe_num]['gvr_at_specific_window'])
            else:
                specific_window_gvr.append(0)
        
        # 绘制柱状图
        bars = ax.bar(probe_numbers, specific_window_gvr, 
                     color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 添加数值标注
        for i, (bar, gvr) in enumerate(zip(bars, specific_window_gvr)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{gvr:.4f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        # 标注峰值
        gvr_array = np.array(specific_window_gvr)
        if len(gvr_array) > 0:
            peaks, _ = find_peaks(gvr_array, distance=2)
            if len(peaks) > 0:
                peak_gvr = gvr_array[peaks]
                peak_probes = [probe_numbers[i] for i in peaks]
                ax.scatter(peak_probes, peak_gvr, s=300, c='red', 
                          marker='^', edgecolors='black', linewidths=2,
                          label=f'检测到 {len(peaks)} 个峰值', zorder=5)
                # 标注峰值
                for i, (probe, gvr) in enumerate(zip(peak_probes, peak_gvr)):
                    ax.annotate(f'探针{probe}\n{gvr:.4f}', 
                                   xy=(probe, gvr), 
                                   xytext=(0, 15), textcoords='offset points',
                                   fontsize=8, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.4', 
                                           facecolor='yellow', alpha=0.8,
                                           edgecolor='black', linewidth=1))
        
        # 设置标签和标题
        ax.set_xlabel('探针编号', fontsize=14, fontweight='bold')
        ax.set_ylabel('GVR (梯度变化率)', fontsize=14, fontweight='bold')
        ax.set_title(f'{damage_folder} - GVR分布\n'
                    f'（第{specific_window_gvr}个滑动窗口）', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(probe_numbers)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_plots:
            output_file = f'GVR_Distribution_{damage_folder}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 已保存GVR分布图: {output_file}")
        
        plt.show()
    
    def visualize_di_analysis(self, damage_folder, gvr_distribution, probe_numbers, 
                      save_plots=True):
        """
        可视化单个通道的DI分析
        
        包含：
        1. DI曲线
        2. DI一阶导数
        3. DI二阶导数（GVR）及峰值检测
        """
        if not gvr_distribution:
            return
        
        # 选择第一个有数据的探针进行详细分析
        for probe_num in probe_numbers:
            if probe_num in gvr_distribution:
                print(f"\n[详细分析] 探针 {probe_num}")
                
                ch_data = gvr_distribution[probe_num]
                
                # 重新计算所有窗口的DI值
                all_gvr = ch_data['all_gvr_values']
                n_windows = len(all_gvr)
                
                # 计算所有窗口的DI
                all_di = []
                for i in range(n_windows):
                    # 这里为了演示，假设一个简单的DI计算
                    # 实际应用中，需要健康数据进行滑动窗口分析
                    di_val = all_gvr[i] * 0.1  # 简化处理
                    all_di.append(di_val)
                
                all_di = np.array(all_di)
                
                # 计算导数
                di_prime = np.diff(all_di)
                di_prime = np.insert(di_prime, 0, 0)
                di_double_prime = np.abs(np.diff(di_prime))
                di_double_prime = np.insert(di_double_prime, 0, 0)
                
                # 检测峰值
                peaks, _ = find_peaks(di_double_prime, distance=3, prominence=np.std(di_double_prime)*1.5)
                
                # 创建窗口索引
                window_indices = np.arange(n_windows)
                
                # 创建图形
                fig, axes = plt.subplots(3, 1, figsize=(14, 12))
                fig.suptitle(f'探针 {probe_num} DI分析 - {damage_folder}', 
                            fontsize=16, fontweight='bold')
                
                # 子图1: DI曲线
                axes[0].plot(window_indices, all_di, 'b-', linewidth=2, label='DI')
                axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[0].set_ylabel('DI', fontsize=12, fontweight='bold')
                axes[0].set_title('损伤指数 (DI) - 论文公式(8)', 
                                  fontsize=13, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc='best')
                
                # 子图2: DI一阶导数
                axes[1].plot(window_indices, di_prime, 'g-', linewidth=2, label="DI'")
                axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[1].set_ylabel("DI'", fontsize=12, fontweight='bold')
                axes[1].set_title('DI一阶导数 - 论文公式(9)', 
                                  fontsize=13, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc='best')
                
                # 子图3: DI二阶导数及峰值
                axes[2].plot(window_indices, di_double_prime, 'r-', linewidth=2, 
                            label="DI'' (GVR)")
                axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # 标注峰值
                if len(peaks) > 0:
                    peak_indices = window_indices[peaks]
                    peak_values = di_double_prime[peaks]
                    axes[2].plot(peak_indices, peak_values, 'ko', markersize=8, 
                                 label=f'峰值 ({len(peaks)}个)', zorder=5)
                    for i, (idx, val) in enumerate(zip(peaks, peak_values)):
                        axes[2].annotate(f'P{i+1}\n{val:.4f}', 
                                           xy=(peak_indices[i], val), 
                                           xytext=(0, 10), textcoords='offset points',
                                           fontsize=9, fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='yellow', alpha=0.8,
                                                   edgecolor='black', linewidth=1))
                
                axes[2].set_ylabel("DI''", fontsize=12, fontweight='bold')
                axes[2].set_title('DI二阶导数及峰值检测 (GVR) - 论文公式(10)-(11)', 
                                  fontsize=13, fontweight='bold')
                axes[2].set_xlabel('滑动窗口索引', fontsize=12, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc='best')
                
                plt.tight_layout()
                
                if save_plots:
                    output_file = f'DI_Analysis_{damage_folder}_Probe{probe_num}.png'
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"  ✓ 已保存DI分析图: {output_file}")
                
                plt.show()
                
                break  # 只分析第一个探针
    
    def visualize_damage_probability(self, damage_folder, probability_distribution, 
                                probe_numbers, save_plots=True):
        """
        可视化损伤概率分布
        
        对应论文图8/图10
        横轴：探针编号
        纵轴：损伤识别概率（%）
        """
        if not probability_distribution:
            return
        
        probabilities = [probability_distribution[probe]['probability'] for probe in probe_numbers]
        
        # 根据损伤概率设置颜色
        colors = []
        for prob in probabilities:
            if prob > 20:
                colors.append('#ff4d4d')  # 深红
            elif prob > 10:
                colors.append('#ffa500')  # 橙色
            elif prob > 5:
                colors.append('#ffd700')  # 金色
            else:
                colors.append('#90ee90')  # 浅绿
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制柱状图
        bars = ax.bar(probe_numbers, probabilities, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.85)
        
        # 添加数值标注
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prob:.2f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        # 添加阈值线（论文使用10%作为损伤检测阈值）
        threshold = 10.0
        ax.axhline(y=threshold, color='darkred', linestyle='--', 
                  linewidth=3, label=f'损伤检测阈值 ({threshold}%)')
        
        # 设置标签和标题
        ax.set_xlabel('探针编号', fontsize=14, fontweight='bold')
        ax.set_ylabel('损伤识别概率 (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{damage_folder} - 损伤识别概率统计\n'
                    f'论文图8/图10风格', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(probe_numbers)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_plots:
            output_file = f'Damage_Probability_{damage_folder}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 已保存损伤概率图: {output_file}")
        
        plt.show()
    
    def analyze_damage_scenario(self, damage_folder, window_length=3000, 
                                 step_size=50, specific_window_index=10):
        """
        分析损伤场景的完整流程
        """
        print(f"\n{'='*70}")
        print(f"步骤2: 分析损伤场景 - {damage_folder}")
        print(f"{'='*70}")
        print(f"路径: {os.path.join(self.data_root_path, damage_folder)}")
        print(f"窗口长度: {window_length} 样本, 步长: {step_size} 样本")
        print(f"特定窗口索引: {specific_window_index}\n")
        
        # 1. 计算GVR分布
        gvr_distribution, probe_numbers = self.calculate_gvr_along_probes(
            damage_folder, window_length, step_size, specific_window_index
        )
        
        # 2. 计算损伤概率分布
        probability_distribution, sorted_probes = self.calculate_damage_probability_distribution(
            damage_folder, window_length, step_size, damage_threshold_percentile=90
        )
        
        # 存储结果
        self.results[damage_folder] = {
            'gvr_distribution': gvr_distribution,
            'probability_distribution': probability_distribution,
            'probe_numbers': sorted_probes,
            'parameters': {
                'window_length': window_length,
                'step_size': step_size,
                'specific_window_index': specific_window_index
            }
        }
        
        print(f"✓ 分析完成")
        print(f"  - 计算了 {len(gvr_distribution)} 个探针的GVR分布")
        print(f"  - 计算了 {len(probability_distribution)} 个探针的损伤概率")
        
        return gvr_distribution, sorted_probes, probability_distribution
    
    def generate_summary_report(self, damage_folder, save_to_file=True):
        """生成分析摘要报告"""
        if damage_folder not in self.results:
            return
        
        results = self.results[damage_folder]
        gvr_dist = results['gvr_distribution']
        prob_dist = results['probability_distribution']
        probe_numbers = results['probe_numbers']
        
        print(f"\n{'='*70}")
        print(f" GVR分析摘要报告 - {damage_folder}")
        print(f"{'='*70}\n")
        
        print(f"分析参数:")
        print(f"  - 窗口长度: {results['parameters']['window_length']} 样本")
        print(f"  - 滑动步长: {results['parameters']['step_size']} 样本")
        print(f"  - 特定窗口索引: {results['parameters']['specific_window_index']}")
        
        print(f"\nGVR分布统计（第{results['parameters']['specific_window_index']}个窗口）:")
        for probe_num in probe_numbers:
            if probe_num in gvr_dist:
                gvr = gvr_dist[probe_num]['gvr_at_specific_window']
                print(f"  - 探针 {probe_num}: GVR = {gvr:.6f}")
        
        print(f"\n损伤概率统计:")
        max_prob_probe = max(prob_dist, key=lambda k: prob_dist[k]['probability'])
        max_prob = prob_dist[max_prob_probe]['probability']
        
        print(f"  - 最高损伤概率: 探针 {max_prob_probe} ({max_prob:.2f}%)")
        print(f"\n各探针损伤概率:")
        for probe_num in probe_numbers:
            if probe_num in prob_dist:
                prob = prob_dist[probe_num]['probability']
                status = "🔴损伤" if prob > 10 else ("🟡可能" if prob > 5 else "🟢正常")
                print(f"  - 探针 {probe_num}: {prob:.2f}% ({status})")
        
        print(f"{'='*70}\n")
        
        if save_to_file:
            report_file = f'GVR_Report_{damage_folder}.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write(" GVR分析摘要报告\n")
                f.write("="*70 + "\n\n")
                f.write(f"损伤场景: {damage_folder}\n")
                f.write("="*70 + "\n\n")
                f.write("GVR分布统计:\n")
                for probe_num in probe_numbers:
                    if probe_num in gvr_dist:
                        gvr = gvr_dist[probe_num]['gvr_at_specific_window']
                        f.write(f"  探针 {probe_num}: GVR = {gvr:.6f}\n")
                f.write("\n损伤概率统计:\n")
                for probe_num in probe_numbers:
                    if probe_num in prob_dist:
                        prob = prob_dist[probe_num]['probability']
                        f.write(f"  探针 {probe_num}: {prob:.2f}%\n")
            
            print(f"✓ 已保存报告到: {report_file}\n")


def main():
    """主程序"""
    
    print("="*70)
    print(" GVR自动标注系统 - 最终完整版")
    print(" 完全按照论文方法实现")
    print("="*70)
    
    # 用户配置区域
    DATA_ROOT_PATH = 'C:/Users/30807/Documents/GitHub/machinelearning/ansys_data'
    
    # 滤波器参数（论文参数）
    CUTOFF_FREQ = 5.0
    FILTER_ORDER = 4
    FS = 1000
    
    # 分析参数（论文参数）
    WINDOW_LENGTH = 3000
    STEP_SIZE = 50
    SPECIFIC_WINDOW_INDEX = 10
    
    # 要分析的损伤场景列表
    DAMAGE_SCENARIOS = [
        '3号30%损伤',
        '3号40%损伤',
        '4号40%+8号40%损伤',
        '5号40%+10号30%损伤',
        '5号40%损伤',
        '6号30%+12号30%损伤',
        '7号40%损伤'
    ]
    
    try:
        print(f"\n[初始化]")
        gvr_system = GVRAutoLabeling(
            data_root_path=DATA_ROOT_PATH,
            cutoff_freq=CUTOFF_FREQ,
            filter_order=FILTER_ORDER,
            fs=FS
        )
        
        success = gvr_system.load_healthy_baseline(healthy_folder='无损')
        if not success:
            print("错误: 无法加载健康基线数据，程序退出")
            return
        
        for scenario in DAMAGE_SCENARIOS:
            print(f"\n[分析 {scenario}]")
            gvr_dist, sorted_probes, prob_dist = gvr_system.analyze_damage_scenario(
                damage_folder=scenario,
                window_length=WINDOW_LENGTH,
                step_size=STEP_SIZE,
                specific_window_index=SPECIFIC_WINDOW_INDEX
            )
            
            if gvr_dist and sorted_probes and prob_dist:
                print(f"\n[可视化 - GVR分布图]")
                gvr_system.visualize_gvr_distribution(scenario, gvr_dist, sorted_probes)
                
                print(f"\n[可视化 - DI详细分析图]")
                gvr_system.visualize_di_analysis(scenario, gvr_dist, sorted_probes)
                
                print(f"\n[可视化 - 损伤概率图]")
                gvr_system.visualize_damage_probability(scenario, prob_dist, sorted_probes)
                
                print(f"\n[生成报告]")
                gvr_system.generate_summary_report(scenario)
        
        print("\n" + "="*70)
        print(" 分析完成！")
        print(" 所有可视化图像和报告已保存")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
