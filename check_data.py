"""
=============================================================================
MDOF仿真数据质量检查可视化工具
MDOF Simulation Data Quality Inspection Visualizer

功能:
1. 数据基本统计信息汇总
2. 时域信号可视化与对比
3. 频域分析 (FFT功率谱)
4. 图像质量检查
5. 标签分布分析
6. 跨模态一致性验证
7. 异常值检测
8. 类间可分性分析
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
import pandas as pd
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')



# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MDOFDataQualityInspector:
    """
    MDOF仿真数据质量检查器
    
    提供全面的数据质量诊断和可视化功能
    """
    
    def __init__(self, signals: np.ndarray, images: np.ndarray, 
                 labels: np.ndarray, fs: float = 100.0):
        """
        初始化检查器
        
        参数:
            signals: (n_samples, n_dof, n_timepoints)
            images: (n_samples, 3, H, W) 或 (n_samples, H, W, C)
            labels: (n_samples,)
            fs: 采样频率 (Hz)
        """
        self.signals = signals
        self.images = self._normalize_images(images)
        self.labels = labels
        self.fs = fs
        
        # 基本信息
        self.n_samples, self.n_dof, self.n_timepoints = signals.shape
        self.n_classes = len(np.unique(labels))
        self.class_names = ['健康', '中间损伤', '底部损伤', '多处损伤'][:self.n_classes]
        
        # 计算频域信息
        self.freqs = np.fft.rfftfreq(self.n_timepoints, d=1.0/fs)
        self.nyquist = self.fs / 2
        
        print(f"✓ 数据质量检查器初始化完成")
        print(f"  - 样本数: {self.n_samples}")
        print(f"  - 自由度: {self.n_dof}")
        print(f"  - 时间点: {self.n_timepoints}")
        print(f"  - 类别数: {self.n_classes}")
    
    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        """确保图像格式为 (n_samples, 3, H, W)"""
        if images.ndim == 4:
            # (n_samples, 3, H, W)
            return images
        elif images.ndim == 4 and images.shape[-1] == 3:
            # (n_samples, H, W, 3) -> (n_samples, 3, H, W)
            return np.transpose(images, (0, 3, 1, 2))
        else:
            raise ValueError(f"不支持的图像格式: {images.shape}")
    
    # ======================================================================
    # 1. 基本统计信息汇总
    # ======================================================================
    
    def print_basic_stats(self):
        """打印基本统计信息"""
        print("\n" + "="*70)
        print("数据基本信息汇总")
        print("="*70)
        
        # 信号统计
        print("\n【信号数据统计】")
        print(f"  形状: {self.signals.shape}")
        print(f"  数据类型: {self.signals.dtype}")
        print(f"  取值范围: [{self.signals.min():.4f}, {self.signals.max():.4f}]")
        print(f"  均值: {self.signals.mean():.4f}")
        print(f"  标准差: {self.signals.std():.4f}")
        
        # 缺失值检查
        nan_count = np.isnan(self.signals).sum()
        inf_count = np.isinf(self.signals).sum()
        print(f"  NaN值: {nan_count}")
        print(f"  Inf值: {inf_count}")
        
        # 图像统计
        print("\n【图像数据统计】")
        print(f"  形状: {self.images.shape}")
        print(f"  数据类型: {self.images.dtype}")
        print(f"  取值范围: [{self.images.min():.4f}, {self.images.max():.4f}]")
        
        # 标签分布
        print("\n【标签分布】")
        label_counts = np.bincount(self.labels)
        for i, count in enumerate(label_counts):
            print(f"  类别 {i} ({self.class_names[i] if i < len(self.class_names) else '未知'}): {count} ({count/len(self.labels)*100:.1f}%)")
        
        print("="*70 + "\n")
    
    # ======================================================================
    # 2. 完整的数据质量报告
    # ======================================================================
    
    def generate_full_report(self, save_path: str = 'mdof_data_quality_report.png'):
        """
        生成完整的数据质量报告（综合多个图表）
        """
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # ========== 第1行: 标签分布 + 信号统计分布 ==========
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_label_distribution(ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_signal_distribution(ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_image_pixel_distribution(ax3)
        
        # ========== 第2行: 时域信号对比 (健康 vs 损伤) ==========
        for i in range(min(3, self.n_dof)):
            ax = fig.add_subplot(gs[1, i])
            self._plot_time_domain_comparison(ax, sensor_idx=i)
        
        # ========== 第3行: 频域分析对比 ==========
        for i in range(min(3, self.n_dof)):
            ax = fig.add_subplot(gs[2, i])
            self._plot_frequency_domain_comparison(ax, sensor_idx=i)
        
        # ========== 第4行: 时域信号统计特征 ==========
        ax4 = fig.add_subplot(gs[3, 0])
        self._plot_signal_stats_by_class(ax4, stat='mean')
        
        ax5 = fig.add_subplot(gs[3, 1])
        self._plot_signal_stats_by_class(ax5, stat='std')
        
        ax6 = fig.add_subplot(gs[3, 2])
        self._plot_signal_stats_by_class(ax6, stat='rms')
        
        # ========== 第5行: 图像样本展示 ==========
        for i in range(min(3, self.n_classes)):
            ax = fig.add_subplot(gs[4, i])
            self._plot_image_sample(ax, class_idx=i)
        
        # ========== 第6行: 频域特征分布 + 异常检测 ==========
        ax7 = fig.add_subplot(gs[5, 0])
        self._plot_frequency_stats(ax7)
        
        ax8 = fig.add_subplot(gs[5, 1])
        self._plot_snr_estimation(ax8)
        
        ax9 = fig.add_subplot(gs[5, 2])
        self._plot_outlier_detection(ax9)
        
        plt.suptitle('MDOF仿真数据质量综合报告', fontsize=20, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 完整数据质量报告已保存: {save_path}")
        
        return fig
    
    # ======================================================================
    # 3. 标签分布分析
    # ======================================================================
    
    def _plot_label_distribution(self, ax):
        """绘制标签分布"""
        label_counts = np.bincount(self.labels)
        colors = sns.color_palette("husl", len(label_counts))
        
        bars = ax.bar(range(len(label_counts)), label_counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('类别', fontsize=11)
        ax.set_ylabel('样本数量', fontsize=11)
        ax.set_title('标签分布', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(label_counts)))
        ax.set_xticklabels([f'{i}\n{self.class_names[i] if i < len(self.class_names) else ""}' 
                           for i in range(len(label_counts))], fontsize=9)
        
        # 添加数值标签
        for bar, count in zip(bars, label_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/len(self.labels)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3)
    
    # ======================================================================
    # 4. 信号值分布分析
    # ======================================================================
    
    def _plot_signal_distribution(self, ax):
        """绘制信号值分布（直方图）"""
        signal_flat = self.signals.flatten()
        
        ax.hist(signal_flat, bins=100, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black')
        
        # 添加正态分布拟合
        mu, sigma = signal_flat.mean(), signal_flat.std()
        x = np.linspace(signal_flat.min(), signal_flat.max(), 200)
        y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
        ax.plot(x, y, 'r--', linewidth=2, label=f'正态分布\nμ={mu:.3f}\nσ={sigma:.3f}')
        
        ax.set_xlabel('信号值', fontsize=11)
        ax.set_ylabel('概率密度', fontsize=11)
        ax.set_title('信号值分布', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = (f'偏度: {skew(signal_flat):.3f}\n'
                     f'峰度: {kurtosis(signal_flat):.3f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               ha='right', va='top', bbox=dict(facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    # ======================================================================
    # 5. 图像像素值分布分析
    # ======================================================================
    
    def _plot_image_pixel_distribution(self, ax):
        """绘制图像像素值分布"""
        pixel_flat = self.images.flatten()
        
        # 分别绘制RGB通道
        for i, color in enumerate(['red', 'green', 'blue']):
            channel_flat = self.images[:, i, :, :].flatten()
            ax.hist(channel_flat, bins=50, alpha=0.5, color=color, 
                   label=f'通道 {i}', density=True)
        
        ax.set_xlabel('像素值', fontsize=11)
        ax.set_ylabel('概率密度', fontsize=11)
        ax.set_title('图像像素值分布 (RGB通道)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # 添加范围信息
        ax.text(0.95, 0.95, f'范围:\n[{pixel_flat.min():.3f},\n{pixel_flat.max():.3f}]',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=9)
    
    # ======================================================================
    # 6. 时域信号对比
    # ======================================================================
    
    def _plot_time_domain_comparison(self, ax, sensor_idx: int = 0):
        """绘制健康与损伤样本的时域信号对比"""
        time_axis = np.arange(self.n_timepoints) / self.fs
        
        # 获取健康和损伤样本
        healthy_mask = (self.labels == 0)
        damage_mask = (self.labels != 0)
        
        # 计算均值和标准差
        healthy_mean = self.signals[healthy_mask, sensor_idx, :].mean(axis=0)
        healthy_std = self.signals[healthy_mask, sensor_idx, :].std(axis=0)
        damage_mean = self.signals[damage_mask, sensor_idx, :].mean(axis=0)
        damage_std = self.signals[damage_mask, sensor_idx, :].std(axis=0)
        
        # 绘制
        ax.fill_between(time_axis, 
                       healthy_mean - healthy_std, 
                       healthy_mean + healthy_std,
                       alpha=0.3, color='green', label='健康 ±σ')
        ax.plot(time_axis, healthy_mean, color='green', linewidth=2)
        
        ax.fill_between(time_axis,
                       damage_mean - damage_std,
                       damage_mean + damage_std,
                       alpha=0.3, color='red', label='损伤 ±σ')
        ax.plot(time_axis, damage_mean, color='red', linewidth=2)
        
        ax.set_xlabel('时间 (s)', fontsize=10)
        ax.set_ylabel('位移', fontsize=10)
        ax.set_title(f'传感器 {sensor_idx+1}: 时域对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, min(5, time_axis[-1]))  # 只显示前5秒
    
    # ======================================================================
    # 7. 频域分析对比
    # ======================================================================
    
    def _plot_frequency_domain_comparison(self, ax, sensor_idx: int = 0):
        """绘制健康与损伤样本的频域对比"""
        # 计算FFT
        healthy_mask = (self.labels == 0)
        damage_mask = (self.labels != 0)
        
        # 计算功率谱
        def compute_power_spectrum(mask):
            signals_subset = self.signals[mask, sensor_idx, :]
            fft_vals = np.fft.rfft(signals_subset, axis=1)
            power = np.abs(fft_vals) ** 2
            return power.mean(axis=0), power.std(axis=0)
        
        healthy_power, healthy_std = compute_power_spectrum(healthy_mask)
        damage_power, damage_std = compute_power_spectrum(damage_mask)
        
        # 转换为dB
        eps = 1e-10
        healthy_power_db = 10 * np.log10(healthy_power + eps)
        healthy_std_db = 10 * np.log10(healthy_std + eps)
        damage_power_db = 10 * np.log10(damage_power + eps)
        damage_std_db = 10 * np.log10(damage_std + eps)
        
        # 绘制
        ax.fill_between(self.freqs,
                       healthy_power_db - healthy_std_db,
                       healthy_power_db + healthy_std_db,
                       alpha=0.3, color='green', label='健康')
        ax.plot(self.freqs, healthy_power_db, color='green', linewidth=1.5)
        
        ax.fill_between(self.freqs,
                       damage_power_db - damage_std_db,
                       damage_power_db + damage_std_db,
                       alpha=0.3, color='red', label='损伤')
        ax.plot(self.freqs, damage_power_db, color='red', linewidth=1.5)
        
        ax.set_xlabel('频率 (Hz)', fontsize=10)
        ax.set_ylabel('功率谱密度 (dB)', fontsize=10)
        ax.set_title(f'传感器 {sensor_idx+1}: 频域对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, min(10, self.nyquist))  # 显示0-10Hz
    
    # ======================================================================
    # 8. 信号统计特征按类别分析
    # ======================================================================
    
    def _plot_signal_stats_by_class(self, ax, stat: str = 'mean'):
        """绘制不同类别的信号统计特征对比"""
        stats_data = []
        class_labels = []
        
        for class_idx in range(self.n_classes):
            mask = (self.labels == class_idx)
            signals_class = self.signals[mask, :, :]
            
            if stat == 'mean':
                values = signals_class.mean(axis=(1, 2))
            elif stat == 'std':
                values = signals_class.std(axis=(1, 2))
            elif stat == 'rms':
                values = np.sqrt(np.mean(signals_class**2, axis=(1, 2)))
            elif stat == 'max':
                values = signals_class.max(axis=(1, 2))
            elif stat == 'min':
                values = signals_class.min(axis=(1, 2))
            else:
                values = signals_class.mean(axis=(1, 2))
            
            stats_data.append(values)
            class_labels.append(self.class_names[class_idx] if class_idx < len(self.class_names) else f'类别{class_idx}')
        
        # 绘制箱线图
        parts = ax.boxplot(stats_data, patch_artist=True, labels=class_labels)
        
        # 设置颜色
        colors = sns.color_palette("husl", self.n_classes)
        for patch, color in zip(parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        stat_name_dict = {'mean': '均值', 'std': '标准差', 'rms': 'RMS',
                          'max': '最大值', 'min': '最小值'}
        ax.set_ylabel(stat_name_dict.get(stat, stat), fontsize=11)
        ax.set_title(f'信号 {stat_name_dict.get(stat, stat)} 按类别分布', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    # ======================================================================
    # 9. 图像样本展示
    # ======================================================================
    
    def _plot_image_sample(self, ax, class_idx: int = 0):
        """绘制指定类别的图像样本"""
        mask = (self.labels == class_idx)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, '无样本', ha='center', va='center', fontsize=12)
            return
        
        # 随机选择一个样本
        idx = np.where(mask)[0][np.random.randint(0, mask.sum())]
        img = self.images[idx]
        
        # 转换为 (H, W, C) 用于显示
        img_display = np.transpose(img, (1, 2, 0))
        
        ax.imshow(img_display)
        ax.set_title(f'{self.class_names[class_idx] if class_idx < len(self.class_names) else f"类别{class_idx}"}', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # ======================================================================
    # 10. 频域统计特征
    # ======================================================================
    
    def _plot_frequency_stats(self, ax):
        """绘制主要频率成分的统计特征"""
        # 计算每个样本的主频
        dominant_freqs = []
        class_labels = []
        
        for class_idx in range(self.n_classes):
            mask = (self.labels == class_idx)
            signals_class = self.signals[mask, :, :]
            
            for i in range(signals_class.shape[0]):
                # 计算平均频谱
                signal_avg = signals_class[i].mean(axis=0)
                fft_vals = np.abs(np.fft.rfft(signal_avg))
                
                # 找到主频（忽略直流分量）
                peak_idx = np.argmax(fft_vals[1:]) + 1
                dominant_freq = self.freqs[peak_idx]
                dominant_freqs.append(dominant_freq)
                class_labels.append(self.class_names[class_idx] if class_idx < len(self.class_names) else f'类别{class_idx}')
        
        # 绘制散点图
        df = pd.DataFrame({
            'Frequency': dominant_freqs,
            'Class': class_labels
        })
        
        sns.boxplot(data=df, x='Class', y='Frequency', ax=ax)
        ax.set_ylabel('主频 (Hz)', fontsize=11)
        ax.set_title('主频成分按类别分布', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    # ======================================================================
    # 11. 信噪比估计
    # ======================================================================
    
    def _plot_snr_estimation(self, ax):
        """绘制信噪比估计"""
        snr_values = []
        class_labels = []
        
        for class_idx in range(self.n_classes):
            mask = (self.labels == class_idx)
            signals_class = self.signals[mask, :, :]
            
            for i in range(signals_class.shape[0]):
                signal_avg = signals_class[i].mean(axis=1)
                signal_power = np.mean(signal_avg**2)
                noise_power = np.mean((signals_class[i] - signal_avg[:, np.newaxis])**2)
                
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_values.append(snr)
                class_labels.append(self.class_names[class_idx] if class_idx < len(self.class_names) else f'类别{class_idx}')
        
        # 绘制箱线图
        df = pd.DataFrame({
            'SNR_dB': snr_values,
            'Class': class_labels
        })
        
        sns.boxplot(data=df, x='Class', y='SNR_dB', ax=ax)
        ax.set_ylabel('SNR (dB)', fontsize=11)
        ax.set_title('信噪比估计', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        
        # 添加平均值线
        ax.axhline(y=np.mean(snr_values), color='red', linestyle='--', alpha=0.7, label=f'平均值: {np.mean(snr_values):.1f} dB')
        ax.legend(fontsize=8)
    
    # ======================================================================
    # 12. 异常值检测
    # ======================================================================
    
    def _plot_outlier_detection(self, ax):
        """绘制异常值检测结果"""
        # 计算每个样本的RMS值
        rms_values = np.sqrt(np.mean(self.signals**2, axis=(1, 2)))
        
        # 使用IQR方法检测异常值
        Q1 = np.percentile(rms_values, 25)
        Q3 = np.percentile(rms_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (rms_values < lower_bound) | (rms_values > upper_bound)
        
        # 绘制散点图
        colors = ['green' if not o else 'red' for o in outliers]
        ax.scatter(range(len(rms_values)), rms_values, c=colors, alpha=0.6, s=20)
        
        ax.axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7, label='下限')
        ax.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7, label='上限')
        
        ax.set_xlabel('样本索引', fontsize=11)
        ax.set_ylabel('RMS值', fontsize=11)
        ax.set_title(f'异常值检测 ({outliers.sum()} 个异常)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # ======================================================================
    # 13. 交互式探索模式
    # ======================================================================
    
    def interactive_explore(self, n_samples_per_class: int = 3):
        """
        交互式探索模式：展示每个类别的多个样本
        
        参数:
            n_samples_per_class: 每个类别展示的样本数量
        """
        n_cols = min(n_samples_per_class, 5)
        n_rows = self.n_classes * 2  # 2行：信号+图像
        
        fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25)
        
        sample_idx = 0
        for class_idx in range(self.n_classes):
            mask = (self.labels == class_idx)
            class_indices = np.where(mask)[0]
            
            # 随机选择样本
            if len(class_indices) > n_samples_per_class:
                selected_indices = np.random.choice(class_indices, n_samples_per_class, replace=False)
            else:
                selected_indices = class_indices
            
            for col, idx in enumerate(selected_indices):
                # 绘制信号（所有DOF叠加）
                ax_signal = fig.add_subplot(gs[2*class_idx, col])
                time_axis = np.arange(self.n_timepoints) / self.fs
                
                for dof in range(min(self.n_dof, 5)):  # 最多显示5个DOF
                    ax_signal.plot(time_axis, self.signals[idx, dof, :], 
                                  alpha=0.7, linewidth=0.8, 
                                  label=f'DOF{dof+1}' if col == 0 else "")
                
                if col == 0:
                    ax_signal.legend(fontsize=7)
                ax_signal.set_xlabel('时间 (s)', fontsize=9)
                ax_signal.set_ylabel('位移', fontsize=9)
                ax_signal.set_title(f'{self.class_names[class_idx] if class_idx < len(self.class_names) else f"类别{class_idx}"} - 样本{idx}',
                                   fontsize=10, fontweight='bold')
                ax_signal.grid(alpha=0.3)
                ax_signal.set_xlim(0, min(5, time_axis[-1]))
                
                # 绘制图像
                ax_img = fig.add_subplot(gs[2*class_idx+1, col])
                img = np.transpose(self.images[idx], (1, 2, 0))
                ax_img.imshow(img)
                ax_img.axis('off')
                ax_img.set_title('结构图像', fontsize=9)
        
        plt.suptitle('交互式数据探索', fontsize=16, fontweight='bold')
        plt.savefig('mdof_interactive_explore.png', dpi=150, bbox_inches='tight')
        print("✓ 交互式探索结果已保存: mdof_interactive_explore.png")
        plt.show()
    
    # ======================================================================
    # 14. 生成质量报告文本
    # ======================================================================
    
    def generate_text_report(self) -> str:
        """生成文本格式的质量报告"""
        report = []
        report.append("="*70)
        report.append("MDOF仿真数据质量检查报告")
        report.append("="*70)
        
        # 基本信息部分
        report.append("\n【1. 数据基本信息】")
        report.append(f"  样本数量: {self.n_samples}")
        report.append(f"  自由度数量: {self.n_dof}")
        report.append(f"  时间点数量: {self.n_timepoints}")
        report.append(f"  类别数量: {self.n_classes}")
        report.append(f"  采样频率: {self.fs} Hz")
        
        # 数据完整性检查
        report.append("\n【2. 数据完整性检查】")
        nan_count = np.isnan(self.signals).sum()
        inf_count = np.isinf(self.signals).sum()
        report.append(f"  信号中NaN值数量: {nan_count}")
        report.append(f"  信号中Inf值数量: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            report.append("  ✓ 数据完整，无缺失值")
        else:
            report.append("  ⚠ 警告: 数据中存在缺失值或异常值")
        
        # 信号统计特性
        report.append("\n【3. 信号统计特性】")
        report.append(f"  取值范围: [{self.signals.min():.6f}, {self.signals.max():.6f}]")
        report.append(f"  均值: {self.signals.mean():.6f}")
        report.append(f"  标准差: {self.signals.std():.6f}")
        report.append(f"  偏度: {skew(self.signals.flatten()):.6f}")
        report.append(f"  峰度: {kurtosis(self.signals.flatten()):.6f}")
        
        # 类别分布
        report.append("\n【4. 类别分布】")
        label_counts = np.bincount(self.labels)
        total = len(self.labels)
        for i, count in enumerate(label_counts):
            percentage = count / total * 100
            class_name = self.class_names[i] if i < len(self.class_names) else f'类别{i}'
            report.append(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # 类别平衡性评估
        min_count = label_counts.min()
        max_count = label_counts.max()
        balance_ratio = min_count / max_count
        if balance_ratio > 0.8:
            report.append(f"  ✓ 类别分布平衡 (比例: {balance_ratio:.2f})")
        elif balance_ratio > 0.5:
            report.append(f"  ⚠ 类别分布略有不均 (比例: {balance_ratio:.2f})")
        else:
            report.append(f"  ✗ 类别分布严重不平衡 (比例: {balance_ratio:.2f})")
        
        # 异常值检测
        report.append("\n【5. 异常值检测】")
        rms_values = np.sqrt(np.mean(self.signals**2, axis=(1, 2)))
        Q1, Q3 = np.percentile(rms_values, [25, 75])
        IQR = Q3 - Q1
        outliers = (rms_values < Q1 - 1.5*IQR) | (rms_values > Q3 + 1.5*IQR)
        outlier_ratio = outliers.sum() / len(rms_values) * 100
        
        report.append(f"  异常样本数量: {outliers.sum()}")
        report.append(f"  异常样本比例: {outlier_ratio:.2f}%")
        
        if outlier_ratio < 5:
            report.append("  ✓ 异常值比例在可接受范围内")
        else:
            report.append(f"  ⚠ 警告: 异常值比例较高 ({outlier_ratio:.2f}%)")
        
        # 信噪比分析
        report.append("\n【6. 信噪比分析】")
        snr_list = []
        for class_idx in range(self.n_classes):
            mask = (self.labels == class_idx)
            signals_class = self.signals[mask]
            signal_power = np.mean(signals_class**2)
            noise_power = np.mean((signals_class - signals_class.mean(axis=1, keepdims=True))**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_list.append(snr)
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'类别{class_idx}'
            report.append(f"  {class_name}: {snr:.2f} dB")
        
        avg_snr = np.mean(snr_list)
        if avg_snr > 20:
            report.append(f"  ✓ 信噪比良好 (平均: {avg_snr:.2f} dB)")
        elif avg_snr > 10:
            report.append(f"  ⚠ 信噪比一般 (平均: {avg_snr:.2f} dB)")
        else:
            report.append(f"  ✗ 信噪比较低 (平均: {avg_snr:.2f} dB)")
        
        # 频域特性
        report.append("\n【7. 频域特性】")
        report.append(f"  奈奎斯特频率: {self.nyquist:.2f} Hz")
        report.append(f"  频率分辨率: {self.freqs[1]:.4f} Hz")
        
        # 计算主频
        all_dominant_freqs = []
        for i in range(len(self.signals)):
            signal_avg = self.signals[i].mean(axis=0)
            fft_vals = np.abs(np.fft.rfft(signal_avg))
            peak_idx = np.argmax(fft_vals[1:]) + 1
            all_dominant_freqs.append(self.freqs[peak_idx])
        
        report.append(f"  主频范围: {min(all_dominant_freqs):.3f} - {max(all_dominant_freqs):.3f} Hz")
        report.append(f"  主频均值: {np.mean(all_dominant_freqs):.3f} Hz")
        
        # 图像质量
        report.append("\n【8. 图像质量】")
        report.append(f"  图像尺寸: {self.images.shape[2]} x {self.images.shape[3]}")
        report.append(f"  像素值范围: [{self.images.min():.4f}, {self.images.max():.4f}]")
        
        # 结论
        report.append("\n" + "="*70)
        report.append("【总体评估】")
        
        # 综合评分
        score = 0
        max_score = 5
        
        if nan_count == 0 and inf_count == 0:
            score += 1
        if balance_ratio > 0.8:
            score += 1
        if outlier_ratio < 5:
            score += 1
        if avg_snr > 15:
            score += 1
        if 0 <= self.images.min() <= 1 and 0 <= self.images.max() <= 1:
            score += 1
        
        percentage = score / max_score * 100
        if percentage >= 80:
            assessment = "✓ 数据质量优秀"
        elif percentage >= 60:
            assessment = "⚠ 数据质量良好，有改进空间"
        else:
            assessment = "✗ 数据质量需改进"
        
        report.append(f"  综合评分: {score}/{max_score} ({percentage:.0f}%)")
        report.append(f"  {assessment}")
        report.append("="*70 + "\n")
        
        return "\n".join(report)
    
    # ======================================================================
    # 15. 批量检查和报告生成
    # ======================================================================
    
    def run_full_inspection(self, output_prefix: str = 'mdof_quality_check'):
        """
        运行完整的检查流程并生成所有报告
        
        参数:
            output_prefix: 输出文件名前缀
        """
        print("\n" + "="*70)
        print("开始运行完整的数据质量检查流程")
        print("="*70)
        
        # 1. 打印基本统计信息
        self.print_basic_stats()
        
        # 2. 生成综合可视化报告
        print("\n生成综合可视化报告...")
        fig = self.generate_full_report(f'{output_prefix}_report.png')
        
        # 3. 生成文本报告
        print("生成文本质量报告...")
        text_report = self.generate_text_report()
        
        # 保存文本报告
        with open(f'{output_prefix}_report.txt', 'w', encoding='utf-8') as f:
            f.write(text_report)
        print(f"✓ 文本报告已保存: {output_prefix}_report.txt")
        
        # 打印文本报告
        print("\n" + text_report)
        
        # 4. 生成交互式探索图
        print("\n生成交互式探索图...")
        self.interactive_explore(n_samples_per_class=3)
        
        print("\n" + "="*70)
        print("数据质量检查完成！")
        print("="*70)
        
        return fig, text_report


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == "__main__":
    # 模拟生成一些测试数据
    print("生成测试数据...")
    np.random.seed(42)
    
    n_samples = 500
    n_dof = 10
    n_timepoints = 1000
    fs = 100.0
    
    # 生成模拟信号
    signals = np.zeros((n_samples, n_dof, n_timepoints))
    for i in range(n_samples):
        for j in range(n_dof):
            # 基础信号：多个频率成分
            t = np.arange(n_timepoints) / fs
            base_signal = (0.5 * np.sin(2 * np.pi * 0.5 * t) +
                          0.3 * np.sin(2 * np.pi * 1.2 * t) +
                          0.2 * np.sin(2 * np.pi * 2.5 * t))
            
            # 添加类别相关的变化
            label = i % 4
            if label == 0:  # 健康
                amplitude = 1.0
            elif label == 1:  # 轻微损伤
                amplitude = 0.9 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
            elif label == 2:  # 中等损伤
                amplitude = 0.8 + 0.2 * np.sin(2 * np.pi * 0.2 * t)
            else:  # 严重损伤
                amplitude = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
            
            # 添加噪声
            noise = np.random.randn(n_timepoints) * 0.1
            
            signals[i, j, :] = amplitude * base_signal + noise
    
    # 生成模拟图像
    images = np.random.rand(n_samples, 3, 224, 224).astype(np.float32)
    
    # 生成标签
    labels = np.random.randint(0, 4, n_samples)
    
    print(f"✓ 测试数据生成完成")
    print(f"  信号形状: {signals.shape}")
    print(f"  图像形状: {images.shape}")
    print(f"  标签形状: {labels.shape}")
    
    # 创建检查器并运行完整检查
    inspector = MDOFDataQualityInspector(signals, images, labels, fs=fs)
    fig, report = inspector.run_full_inspection(output_prefix='mdof_data_quality')
    
    plt.show()
