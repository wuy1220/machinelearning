import numpy as np
from PIL import Image, ImageDraw
import random
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


class DamageDataGenerator:
    """
    基于信号合成原理的损伤检测数据生成器
    模拟多传感器加速度信号和结构图像数据
    """
    
    def __init__(self, 
                 n_sensors: int = 5,
                 sampling_rate: int = 1000,
                 duration: float = 3.0,
                 n_classes: int = 4,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        参数:
            n_sensors: 传感器数量
            sampling_rate: 采样率 (Hz)
            duration: 每个样本的时间长度 (秒)
            n_classes: 损伤类别数 (0=健康, 1-3=不同损伤类型)
            image_size: 图像尺寸 (高, 宽)
        """
        self.n_sensors = n_sensors
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = int(duration * sampling_rate)
        self.n_classes = n_classes
        self.image_size = image_size
        
        # 基础频率成分 (Hz) - 模拟结构的固有频率
        self.base_frequencies = [2.0, 5.0, 10.0]
        
        # 每个传感器的基础信号参数
        self.sensor_params = self._initialize_sensor_parameters()
        
        # 损伤参数配置
        self.damage_config = self._initialize_damage_configuration()
        
    def _initialize_sensor_parameters(self) -> Dict[int, Dict]:
        """
        初始化每个传感器的基础信号参数
        """
        sensor_params = {}
        for sensor_id in range(self.n_sensors):
            # 为每个频率生成随机幅值和相位
            amplitudes = np.random.uniform(0.2, 0.8, len(self.base_frequencies))
            phases = np.random.uniform(0, 2*np.pi, len(self.base_frequencies))
            
            # 传感器位置相关的衰减因子 (距离损伤位置越远，影响越小)
            # 假设传感器按顺序排列，位置因子逐渐减小
            position_factor = 1.0 - (sensor_id * 0.15)
            position_factor = max(0.3, position_factor)  # 最小衰减到30%
            
            sensor_params[sensor_id] = {
                'amplitudes': amplitudes,
                'phases': phases,
                'position_factor': position_factor,
                'noise_level': np.random.uniform(0.05, 0.15)  # 噪声水平
            }
        
        return sensor_params
    
    def _initialize_damage_configuration(self) -> Dict[int, Dict]:
        """
        初始化损伤配置
        每种损伤类型对应特定的传感器和频率变化
        """
        damage_config = {
            0: {  # 健康状态
                'affected_sensors': [],
                'frequency_changes': [],
                'additional_frequency': None,
                'image_regions': []
            },
            1: {  # 损伤类型1 - 影响传感器1
                'affected_sensors': [0],  # 传感器1 (0-based)
                'frequency_changes': [(2, 1.1)],  # 第2个频率增加10%
                'additional_frequency': 15.0,  # 添加15Hz频率
                'image_regions': [0]  # 图像区域0
            },
            2: {  # 损伤类型2 - 影响传感器2和3
                'affected_sensors': [1, 2],  # 传感器2和3
                'frequency_changes': [(1, 1.15), (2, 1.2)],  # 频率变化
                'additional_frequency': 18.0,  # 添加18Hz频率
                'image_regions': [1, 2]  # 图像区域1和2
            },
            3: {  # 损伤类型3 - 多损伤，影响多个传感器
                'affected_sensors': [0, 2, 4],  # 传感器1,3,5
                'frequency_changes': [(0, 1.08), (2, 1.15)],  # 频率变化
                'additional_frequency': 20.0,  # 添加20Hz频率
                'image_regions': [0, 2, 4]  # 图像区域0,2,4
            }
        }
        
        return damage_config
    
    def generate_single_sample(self, label: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        生成单个样本
        
        参数:
            label: 损伤类别标签
            
        返回:
            acceleration_signals: 多传感器加速度信号 (n_sensors, n_samples)
            image: 结构图像 (H, W, 3)
            label: 损伤标签
        """
        # 生成时间轴
        t = np.linspace(0, self.duration, self.n_samples)
        
        # 生成多传感器信号
        acceleration_signals = np.zeros((self.n_sensors, self.n_samples))
        
        for sensor_id in range(self.n_sensors):
            # 获取该传感器的参数
            params = self.sensor_params[sensor_id]
            
            # 生成基础健康信号
            signal = np.zeros_like(t)
            for i, freq in enumerate(self.base_frequencies):
                amplitude = params['amplitudes'][i]
                phase = params['phases'][i]
                signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # 添加噪声
            noise = np.random.normal(0, params['noise_level'], len(t))
            signal += noise
            
            # 应用损伤影响
            damage_params = self.damage_config[label]
            
            if sensor_id in damage_params['affected_sensors']:
                # 计算影响因子 (考虑传感器位置)
                impact_factor = params['position_factor']
                
                # 频率变化
                for freq_idx, change_factor in damage_params['frequency_changes']:
                    original_freq = self.base_frequencies[freq_idx]
                    new_freq = original_freq * change_factor
                    amplitude = params['amplitudes'][freq_idx] * impact_factor
                    phase = params['phases'][freq_idx]
                    
                    # 添加频率变化的分量
                    signal += amplitude * 0.5 * np.sin(2 * np.pi * new_freq * t + phase)
                
                # 添加额外频率分量
                if damage_params['additional_frequency'] is not None:
                    extra_freq = damage_params['additional_frequency']
                    extra_amplitude = 0.3 * impact_factor
                    extra_phase = np.random.uniform(0, 2 * np.pi)
                    signal += extra_amplitude * np.sin(2 * np.pi * extra_freq * t + extra_phase)
                
                # 添加局部特征 (模拟损伤引起的局部振动)
                # 在信号中间添加短时高频振动
                start_idx = int(0.4 * self.n_samples)
                end_idx = int(0.6 * self.n_samples)
                envelope = np.zeros_like(t)
                envelope[start_idx:end_idx] = np.sin(np.pi * np.arange(end_idx-start_idx) / (end_idx-start_idx))**2
                signal += 0.2 * impact_factor * envelope * np.sin(2 * np.pi * 25 * t)
            
            # 归一化信号
            signal = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal
            
            acceleration_signals[sensor_id] = signal
        
        # 生成对应的结构图像
        image = self._generate_structure_image(label)
        
        return acceleration_signals, image, label
    
    def _generate_structure_image(self, label: int) -> np.ndarray:
        """
        生成结构图像
        
        参数:
            label: 损伤类别
            
        返回:
            image: RGB图像 (H, W, 3)
        """
        # 创建白色背景图像
        img = Image.new('RGB', self.image_size, (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # 绘制基础结构网格 (模拟结构)
        self._draw_structure_grid(draw)
        
        # 根据损伤类型绘制损伤区域
        damage_params = self.damage_config[label]
        
        # 定义图像区域
        region_colors = {
            0: (255, 0, 0),      # 红色 - 严重损伤
            1: (0, 255, 0),      # 绿色 - 中等损伤
            2: (0, 0, 255),      # 蓝色 - 轻微损伤
            3: (255, 165, 0),    # 橙色 - 多损伤
            4: (128, 0, 128)     # 紫色 - 其他
        }
        
        # 绘制损伤区域
        for region_id in damage_params['image_regions']:
            self._draw_damage_region(draw, region_id, region_colors[region_id % 5])
        
        # 转换为numpy数组并归一化到[0,1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    
    def _draw_structure_grid(self, draw):
        """绘制基础结构网格"""
        width, height = self.image_size
        
        # 绘制横向线条
        for y in range(height):
            if y % 20 == 0:
                draw.line([(0, y), (width, y)], fill=(200, 200, 200))
        
        # 绘制纵向线条
        for x in range(width):
            if x % 20 == 0:
                draw.line([(x, 0), (x, height)], fill=(200, 200, 200))
        
        # 绘制传感器位置 (用圆圈标记)
        for sensor_id in range(self.n_sensors):
            x = (sensor_id + 1) * width // (self.n_sensors + 1)
            y = height // 2
            radius = 15
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        outline=(100, 100, 100), width=2)
            draw.text((x-5, y-8), f"S{sensor_id+1}", fill=(0, 0, 0))
    
    def _draw_damage_region(self, draw, region_id: int, color: Tuple[int, int, int]):
        """绘制损伤区域"""
        width, height = self.image_size
        
        # 定义区域位置 (将图像分成5个区域)
        region_width = width // 5
        region_x = region_id * region_width
        
        # 修复: 减小 margin 以防止在小区域中坐标反转
        margin = 10  
        
        x1 = region_x + margin
        y1 = margin
        x2 = region_x + region_width - margin
        y2 = height - margin
        
        # 绘制半透明矩形 (通过绘制多个小矩形模拟半透明效果)
        for i in range(5):
            # 修复: 在循环内部计算当前坐标并检查有效性
            cur_x1 = x1 + i * 2
            cur_y1 = y1 + i * 2
            cur_x2 = x2 - i * 2
            cur_y2 = y2 - i * 2
            
            # 只有当左上角坐标小于右下角坐标时才绘制
            if cur_x1 < cur_x2 and cur_y1 < cur_y2:
                alpha_color = (
                    int(color[0] * (0.2 + i*0.15)),
                    int(color[1] * (0.2 + i*0.15)),
                    int(color[2] * (0.2 + i*0.15))
                )
                draw.rectangle([cur_x1, cur_y1, cur_x2, cur_y2], 
                              outline=alpha_color, width=2)
        
        # 绘制"损伤"标记
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        draw.text((center_x-15, center_y-8), "损伤", fill=color)
    
    def generate_dataset(self, n_samples_per_class: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成完整数据集
        
        参数:
            n_samples_per_class: 每个类别的样本数
            
        返回:
            acceleration_signals: 多传感器信号 (n_total_samples, n_sensors, n_samples)
            images: 图像数据 (n_total_samples, 3, H, W)
            labels: 损伤标签 (n_total_samples,)
        """
        total_samples = n_samples_per_class * self.n_classes
        acceleration_signals = np.zeros((total_samples, self.n_sensors, self.n_samples))
        images = np.zeros((total_samples, 3, *self.image_size))
        labels = np.zeros(total_samples, dtype=int)
        
        sample_idx = 0
        
        for label in range(self.n_classes):
            for _ in range(n_samples_per_class):
                # 生成单个样本
                acc_signal, image, _ = self.generate_single_sample(label)
                
                # 存储数据
                acceleration_signals[sample_idx] = acc_signal
                images[sample_idx] = np.transpose(image, (2, 0, 1))  # 转换为(C,H,W)格式
                labels[sample_idx] = label
                
                sample_idx += 1
        
        return acceleration_signals, images, labels
    
    def visualize_samples(self, n_samples: int = 3, save_path: str = None):
        """
        可视化生成的样本
        
        参数:
            n_samples: 要可视化的样本数
            save_path: 保存路径 (可选)
        """
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        labels_to_show = random.sample(range(self.n_classes), min(n_samples, self.n_classes))
        
        for idx, label in enumerate(labels_to_show):
            # 生成样本
            acc_signal, image, _ = self.generate_single_sample(label)
            
            # 绘制加速度信号
            t = np.linspace(0, self.duration, self.n_samples)
            
            for sensor_id in range(self.n_sensors):
                axes[idx, 0].plot(t, acc_signal[sensor_id], 
                                label=f'传感器 {sensor_id+1}', alpha=0.7)
            
            axes[idx, 0].set_xlabel('时间 (秒)', fontsize=12)
            axes[idx, 0].set_ylabel('加速度', fontsize=12)
            axes[idx, 0].set_title(f'类别 {label}: 多传感器加速度信号', fontsize=14, fontweight='bold')
            axes[idx, 0].legend(loc='upper right')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # 显示图像
            axes[idx, 1].imshow(image)
            axes[idx, 1].set_title(f'类别 {label}: 结构图像', fontsize=14, fontweight='bold')
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_signal_statistics(self, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        计算生成信号的统计特性
        
        参数:
            n_samples: 用于统计的样本数
            
        返回:
            statistics: 各类别的信号统计信息
        """
        stats = {}
        
        for label in range(self.n_classes):
            # 生成多个样本
            all_signals = []
            for _ in range(n_samples):
                acc_signal, _, _ = self.generate_single_sample(label)
                all_signals.append(acc_signal)
            
            all_signals = np.array(all_signals)
            
            # 计算统计量
            label_stats = {
                'mean': all_signals.mean(axis=(0, 2)),  # 每个传感器的均值
                'std': all_signals.std(axis=(0, 2)),   # 每个传感器的标准差
                'rms': np.sqrt(np.mean(all_signals**2, axis=(0, 2))),  # RMS值
                'peak_to_peak': all_signals.max(axis=(0, 2)) - all_signals.min(axis=(0, 2))  # 峰峰值
            }
            
            stats[f'class_{label}'] = label_stats
        
        return stats
    
# ============================================================================
# 使用示例
# ============================================================================

def main():
    """
    主函数：演示数据生成器的使用
    """
    print("=" * 70)
    print("损伤检测数据生成器")
    print("基于信号合成原理的多模态数据生成")
    print("=" * 70)
    
    # ==================== 参数设置 ====================
    """
    配置参数模块
    定义了系统使用的各种常量参数，包括传感器设置、采样参数、分类参数和图像参数等
    """
    N_SENSORS = 5   # 传感器数量 - 定义系统使用的传感器总数
    SAMPLING_RATE = 1000  # 采样率 - 每秒采样次数，单位为Hz
    DURATION = 3.0  # 采样时长 - 每次采样的持续时间，单位为秒
    N_CLASSES = 4   # 分类数量 - 系统需要区分的类别总数
    IMAGE_SIZE = (224, 224)  # 图像尺寸 - 处理图像的固定尺寸，格式为(宽度, 高度)
    SAMPLES_PER_CLASS = 200  # 每类样本数 - 每个类别包含的训练样本数量
    
    # ==================== 初始化生成器 ====================
    print("\n[1] 初始化数据生成器...")
    generator = DamageDataGenerator(
        n_sensors=N_SENSORS,
        sampling_rate=SAMPLING_RATE,
        duration=DURATION,
        n_classes=N_CLASSES,
        image_size=IMAGE_SIZE
    )
    print(f"  ✓ 传感器数量: {N_SENSORS}")
    print(f"  ✓ 采样率: {SAMPLING_RATE} Hz")
    print(f"  ✓ 每个样本时长: {DURATION} 秒")
    print(f"  ✓ 样本点数: {generator.n_samples}")
    print(f"  ✓ 损伤类别数: {N_CLASSES}")
    print(f"  ✓ 图像尺寸: {IMAGE_SIZE}")
    
    # ==================== 可视化样本 ====================
    print("\n[2] 可视化生成的样本...")
    generator.visualize_samples(n_samples=4, save_path='generated_samples.png')
    
    # ==================== 生成完整数据集 ====================
    print("\n[3] 生成完整数据集...")
    acceleration_signals, images, labels = generator.generate_dataset(
        n_samples_per_class=SAMPLES_PER_CLASS
    )
    
    print(f"  ✓ 加速度信号形状: {acceleration_signals.shape}")
    print(f"  ✓ 图像数据形状: {images.shape}")
    print(f"  ✓ 标签形状: {labels.shape}")
    print(f"  ✓ 各类别样本数: {np.bincount(labels)}")
    
    # ==================== 计算信号统计特性 ====================
    print("\n[4] 计算信号统计特性...")
    statistics = generator.get_signal_statistics(n_samples=50)
    
    for label_key, label_stats in statistics.items():
        print(f"\n{label_key} 统计特性:")
        print("-" * 40)
        for sensor_id in range(N_SENSORS):
            print(f"  传感器 {sensor_id+1}:")
            print(f"    均值: {label_stats['mean'][sensor_id]:.4f}")
            print(f"    标准差: {label_stats['std'][sensor_id]:.4f}")
            print(f"    RMS: {label_stats['rms'][sensor_id]:.4f}")
            print(f"    峰峰值: {label_stats['peak_to_peak'][sensor_id]:.4f}")
    
    # ==================== 绘制信号对比 ====================
    print("\n[5] 绘制不同类别信号对比...")
    fig, axes = plt.subplots(N_SENSORS, N_CLASSES, figsize=(4*N_CLASSES, 3*N_SENSORS))
    
    for sensor_id in range(N_SENSORS):
        for label in range(N_CLASSES):
            # 生成样本
            acc_signal, _, _ = generator.generate_single_sample(label)
            t = np.linspace(0, DURATION, generator.n_samples)
            
            axes[sensor_id, label].plot(t, acc_signal[sensor_id], linewidth=1)
            axes[sensor_id, label].set_title(f'传感器{sensor_id+1} - 类别{label}', fontsize=10)
            axes[sensor_id, label].grid(True, alpha=0.3)
            axes[sensor_id, label].set_xlabel('时间 (秒)', fontsize=8)
            axes[sensor_id, label].set_ylabel('加速度', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('signal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==================== 验证数据质量 ====================
    print("\n[6] 验证数据质量...")
    
    # 检查信号范围
    signal_range = (acceleration_signals.min(), acceleration_signals.max())
    print(f"  ✓ 信号值范围: [{signal_range[0]:.4f}, {signal_range[1]:.4f}]")
    
    # 检查图像范围
    image_range = (images.min(), images.max())
    print(f"  ✓ 图像值范围: [{image_range[0]:.4f}, {image_range[1]:.4f}]")
    
    # 检查类别分布
    class_distribution = np.bincount(labels) / len(labels) * 100
    print(f"  ✓ 类别分布:")
    for class_idx, percentage in enumerate(class_distribution):
        print(f"    类别 {class_idx}: {percentage:.1f}%")
    
    # ==================== 保存数据 ====================
    print("\n[7] 保存生成的数据...")
    np.savez_compressed('damage_detection_dataset.npz',
                        acceleration_signals=acceleration_signals,
                        images=images,
                        labels=labels)
    print("  ✓ 数据已保存到: damage_detection_dataset.npz")
    
    print("\n" + "=" * 70)
    print("数据生成完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
