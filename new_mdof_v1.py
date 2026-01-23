import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import savemat
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import json

class ImprovedJacketPlatformSimulator:
    """
    改进的海洋导管架平台多自由度仿真系统 (剪切型模型)
    旨在生成具有显著损伤特征的振动响应数据
    """
    
    def __init__(self, 
                 num_degrees: int = 30,
                 dt: float = 0.005,
                 duration: float = 60.0, 
                 damping_ratio: float = 0.05,
                 seed: Optional[int] = None):
        """初始化改进的导管架平台仿真系统"""
        self.num_degrees = num_degrees
        self.dt = dt
        self.duration = duration
        self.damping_ratio = damping_ratio
        self.num_steps = int(duration / dt)
        self.time = np.linspace(0, duration, self.num_steps)
        
        if seed is not None:
            np.random.seed(seed)
            
        self.M = None  
        self.C = None  
        self.K = None  
        self.K0 = None 
        self.layer_stiffness = None
        
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化基于剪切模型的系统矩阵"""
        # 调整质量分布，优化响应幅值
        base_mass = 50000.0 
        mass_distribution = np.linspace(1.2, 1.0, self.num_degrees)
        self.mass_values = base_mass * mass_distribution
        self.M = np.diag(self.mass_values)
        
        # 增加基础刚度，提高系统频率稳定性
        base_stiffness = 5.0e7 
        perturbation = np.random.uniform(0.95, 1.05, self.num_degrees)
        self.layer_stiffness = base_stiffness * perturbation
        
        self.K = self._build_stiffness_matrix_from_layers()
        self.K0 = self.K.copy()  # 保存健康状态刚度
        
        self._compute_modal_properties()
        self._build_damping_matrix()
        
        print(f"系统初始化完成。固有频率范围: {self.natural_frequencies[0]/(2*np.pi):.2f} - {self.natural_frequencies[-1]/(2*np.pi):.2f} Hz")

    def _build_stiffness_matrix_from_layers(self) -> np.ndarray:
        """根据层间刚度构建剪切型结构的总刚度矩阵"""
        n = self.num_degrees
        K = np.zeros((n, n))
        
        for i in range(n):
            k_curr = self.layer_stiffness[i]
            k_next = self.layer_stiffness[i+1] if i < n - 1 else 0
            
            if i == 0:
                # 底部节点，连接地基和上层
                K[0, 0] = k_curr + k_next
                if n > 1:
                    K[0, 1] = -k_next
                    K[1, 0] = -k_next
            elif i == n - 1:
                # 顶部节点，仅连接下层
                K[i, i] = k_curr
                K[i, i-1] = -k_curr
                K[i-1, i] = -k_curr
            else:
                # 中间节点
                K[i, i] = k_curr + k_next
                K[i, i+1] = -k_next
                K[i+1, i] = -k_next
                K[i, i-1] = -k_curr
                K[i-1, i] = -k_curr
                
        return K
    
    def _compute_modal_properties(self):
        """计算模态属性"""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(self.M) @ self.K)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
        except np.linalg.LinAlgError:
            print("警告：矩阵奇异，使用单位矩阵初始化")
            self.K = np.eye(self.num_degrees) * 1e6
            self.M = np.eye(self.num_degrees) * 1000
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(self.M) @ self.K)

        self.natural_frequencies = np.sqrt(eigenvalues)
        self.mode_shapes = eigenvectors
    
    def _build_damping_matrix(self):
        """构建Rayleigh阻尼矩阵"""
        if len(self.natural_frequencies) >= 2:
            omega1 = self.natural_frequencies[0]
            omega2 = self.natural_frequencies[1]
            
            alpha = 2 * self.damping_ratio * omega1 * omega2 / (omega1 + omega2)
            beta = 2 * self.damping_ratio / (omega1 + omega2)
            
            self.C = alpha * self.M + beta * self.K
        else:
            self.C = 2 * self.damping_ratio * np.sqrt(self.K[0,0] / self.M[0,0]) * self.M

    def apply_damage(self, 
                     damaged_dofs: List[int], 
                     severity_ratios: List[float]) -> np.ndarray:
        """
        施加损伤：降低指定自由度对应的层间刚度。
        修改为无状态操作，每次都基于 K0 计算，避免累积误差。
        """
        K_damaged = self.K0.copy()
        
        # 基于 K0 提取原始层刚度的副本
        current_layers = np.zeros(self.num_degrees)
        n = self.num_degrees
        
        # 从 K0 反推层刚度
        # 最后一行对角线即为最后一个弹簧
        current_layers[n-1] = K_damaged[n-1, n-1]
        
        for i in range(n-2, -1, -1):
            # k[i] = K[i,i] - k[i+1]
            current_layers[i] = K_damaged[i, i] - current_layers[i+1]
            
        # 应用损伤
        for dof, severity in zip(damaged_dofs, severity_ratios):
            if dof < self.num_degrees:
                # 修改层刚度
                reduction = current_layers[dof] * severity
                current_layers[dof] -= reduction
        
        # 重建 K_damaged
        for i in range(n):
            k_curr = current_layers[i]
            k_next = current_layers[i+1] if i < n - 1 else 0
            
            if i == 0:
                K_damaged[0, 0] = k_curr + k_next
            elif i == n - 1:
                K_damaged[i, i] = k_curr
            else:
                K_damaged[i, i] = k_curr + k_next
            
            # 重置非对角线
            if i < n - 1:
                K_damaged[i, i+1] = -k_next
                K_damaged[i+1, i] = -k_next
                
        return K_damaged

    def generate_excitation(self, 
                          excitation_type: str = 'filtered_noise',
                          amplitude: float = 50000.0,  # 优化幅值
                          **kwargs) -> np.ndarray:
        """生成激励力"""
        n = self.num_degrees
        num_steps = self.num_steps
        
        if excitation_type == 'filtered_noise':
            raw_noise = np.random.randn(num_steps, n)
            
            fs = 1.0 / self.dt
            nyquist = 0.5 * fs
            low = 0.2 / nyquist
            high = 3.0 / nyquist  # 专注于低频，避免高频噪声
            
            if low >= high:
                low = 0.01
                high = 0.1
            
            F = np.zeros_like(raw_noise)
            for i in range(n):
                # 使用 sos (second-order sections) 提高数值稳定性
                sos = signal.butter(4, [low, high], btype='band', output='sos')
                F[:, i] = signal.sosfiltfilt(sos, raw_noise[:, i]) * amplitude
                
        elif excitation_type == 'harmonic':
            freq_start = self.natural_frequencies[0] * 0.9
            freq_end = self.natural_frequencies[0] * 1.1
            
            F = np.zeros((num_steps, n))
            t = self.time
            instantaneous_freq = freq_start + (freq_end - freq_start) * t / self.duration
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) * self.dt
            
            F[:, 0] = amplitude * np.sin(phase)
            
        elif excitation_type == 'random':
            F = np.random.randn(num_steps, n) * amplitude * 0.1
            
        else:
            raise ValueError(f"Unknown excitation type: {excitation_type}")
        
        return F

    def simulate_response(self, 
                         K: np.ndarray, 
                         F: np.ndarray) -> np.ndarray:
        """使用Newmark-β方法求解结构响应"""
        gamma = 0.5
        beta = 0.25
        
        n = self.num_degrees
        num_steps = self.num_steps
        
        x = np.zeros((num_steps, n))
        v = np.zeros((num_steps, n))
        a = np.zeros((num_steps, n))
        
        a[0] = np.linalg.solve(self.M, F[0])
        
        K_eff = K + gamma/(beta*self.dt) * self.C + 1/(beta*self.dt**2) * self.M
        
        for i in range(num_steps - 1):
            x_tilde = x[i] + self.dt * v[i] + 0.5 * self.dt**2 * (1 - 2*beta) * a[i]
            v_tilde = v[i] + self.dt * (1 - gamma) * a[i]
            
            F_eff = F[i+1] + self.M @ (1/(beta*self.dt**2) * x_tilde + 1/(beta*self.dt) * v[i]) + \
                    self.C @ (gamma/(beta*self.dt) * x_tilde + (gamma/beta - 1) * v[i])
            
            x[i+1] = np.linalg.solve(K_eff, F_eff)
            
            a[i+1] = 1/(beta*self.dt**2) * (x[i+1] - x_tilde) - 1/(beta*self.dt) * v[i]
            v[i+1] = v[i] + self.dt * ((1-gamma) * a[i] + gamma * a[i+1])
        
        return a


class GVRFeatureExtractor:
    """改进的GVR特征提取器（修复零信号问题）"""
    
    def __init__(self, dt, window_length=3000, step_size=50, cutoff_freq=5.0, filter_order=4):
        """
        初始化GVR特征提取器
        
        Args:
            dt: 采样时间间隔（秒）
            window_length: 滑动窗口长度
            step_size: 滑动窗口步长
            cutoff_freq: 低通滤波截止频率
            filter_order: 滤波器阶数
        """
        self.dt = dt
        self.window_length = window_length
        self.step_size = step_size
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        
        # 设计Butterworth低通滤波器
        nyquist = 0.5 / self.dt
        self.b, self.a = signal.butter(filter_order, cutoff_freq / nyquist, btype='low')
    
    def butterworth_filter(self, data):
        """Butterworth低通滤波器"""
        return signal.filtfilt(self.b, self.a, data, axis=0)
    
    def compute_damage_index(self, damaged_signal, healthy_signal):
        """
        计算损伤指标DI (Damage Index)
        
        公式：DI_j = √[Σ(xd_ij - xh_ij)²] / √[Σ(xh_ij)² + ε]
        
        Args:
            damaged_signal: 损伤后的信号 (N, C)
            healthy_signal: 健康信号 (N, C)
            
        Returns:
            DI: 损伤指标 (C,)
        """
        num_channels = damaged_signal.shape[1]
        DI = np.zeros(num_channels)
        
        for ch in range(num_channels):
            numerator = np.sum((damaged_signal[:, ch] - healthy_signal[:, ch]) ** 2)
            denominator = np.sum(healthy_signal[:, ch] ** 2)
            
            epsilon = 1e-10
            if denominator > epsilon:
                DI[ch] = np.sqrt(numerator) / np.sqrt(denominator)
            else:
                DI[ch] = 0.0
                
        return DI
    
    def compute_gvr(self, DI_series):
        """
        计算梯度变化率 (GVR)
        
        Args:
            DI_series: 损伤指标序列 (num_windows, num_channels)
            
        Returns:
            DI_prime: 一阶差分
            DI_double_prime: 二阶差分（绝对值）
        """
        DI_prime = np.zeros_like(DI_series)
        if DI_series.shape[0] > 1:
            DI_prime[1:] = DI_series[1:] - DI_series[:-1]
        
        DI_double_prime = np.zeros_like(DI_series)
        if DI_prime.shape[0] > 1:
            DI_double_prime[1:] = np.abs(DI_prime[1:] - DI_prime[:-1])
        
        return DI_prime, DI_double_prime
    
    def extract_gvr_features(self, damaged_signal, healthy_signal):
        """
        提取GVR特征（修复版）
        
        关键修复：对健康样本添加微小噪声
        """
        filtered_damaged = self.butterworth_filter(damaged_signal)
        filtered_healthy = self.butterworth_filter(healthy_signal)
        
        # === 修复1：对健康样本添加微小噪声 ===
        # 检测是否为健康样本（damaged和healthy信号几乎相同）
        if np.allclose(filtered_damaged, filtered_healthy, atol=1e-10):
            noise_level = 1e-6
            signal_std = np.std(filtered_healthy) if np.std(filtered_healthy) > 0 else 1.0
            filtered_healthy_noisy = filtered_healthy + \
                np.random.randn(*filtered_healthy.shape) * noise_level * signal_std
        else:
            filtered_healthy_noisy = filtered_healthy
        
        # 滑动窗口提取DI
        num_steps = filtered_damaged.shape[0]
        num_windows = (num_steps - self.window_length) // self.step_size + 1
        
        DI_series = []
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_length
            
            window_damaged = filtered_damaged[start_idx:end_idx]
            window_healthy = filtered_healthy_noisy[start_idx:end_idx] if end_idx <= filtered_healthy_noisy.shape[0] else window_damaged
            
            # 确保窗口长度一致
            if window_damaged.shape[0] < self.window_length:
                window_damaged = np.pad(window_damaged, 
                                       ((0, self.window_length - window_damaged.shape[0]), (0, 0)), 
                                       'edge')
                window_healthy = np.pad(window_healthy,
                                       ((0, self.window_length - window_healthy.shape[0]), (0, 0)),
                                       'edge')
            
            DI_window = self.compute_damage_index(window_damaged, window_healthy)
            DI_series.append(DI_window)
        
        DI_series = np.array(DI_series)
        DI_prime, DI_double_prime = self.compute_gvr(DI_series)
        
        return {
            'DI': DI_series,
            'GVR_prime': DI_prime,
            'GVR_double_prime': DI_double_prime,
        }
    
    def generate_gvr_feature_map(self, gvr_features, image_size=(224, 224)):
        """
        生成改进的GVR特征图
        
        关键修复：更好的归一化和图像生成
        """
        DI_double_prime = gvr_features['GVR_double_prime'] 
        num_samples = DI_double_prime.shape[0]
        num_channels = DI_double_prime.shape[1]
        
        feature_maps = np.zeros((num_samples, image_size[0], image_size[1], 3))
        
        for i in range(num_samples):
            data = DI_double_prime[i]
            
            data_min = data.min()
            data_max = data.max()
            data_range = data_max - data_min
            
            # 改进的归一化：处理零信号情况
            if data_range < 1e-10:
                if np.any(data != 0):
                    abs_max = np.abs(data).max()
                    if abs_max > 0:
                        normalized = (data + abs_max) / (2 * abs_max)
                    else:
                        normalized = np.linspace(0, 1, len(data))
                else:
                    normalized = np.linspace(0, 1, len(data))
            else:
                normalized = (data - data_min) / data_range
            
            # 插值到图像宽度
            x_original = np.arange(num_channels)
            x_new = np.linspace(0, num_channels - 1, image_size[1])
            
            try:
                z_row = np.interp(x_new, x_original, normalized, 
                                 left=normalized[0], right=normalized[-1])
            except:
                z_row = normalized
            
            # 改进的2D图像生成
            y_gradient = np.linspace(0, 1, image_size[0]).reshape(-1, 1)
            
            img_r = np.tile(z_row, (image_size[0], 1))
            img_g = img_r * y_gradient
            img_b = np.tile(z_row.reshape(1, -1), (image_size[0], 1))
            
            img_rgb = np.stack([img_r, img_g, img_b], axis=2)
            feature_maps[i] = img_rgb
        
        return feature_maps


class ImprovedDamageDataGenerator:
    """
    改进的损伤数据生成器
    """
    
    def __init__(self, simulator, gvr_extractor, output_dir='./jacket_damage_data_improved'):
        self.simulator = simulator
        self.gvr_extractor = gvr_extractor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = []
    
    def generate_single_damage_scenario(self,
                                       damaged_dofs,
                                       severity_ratios,
                                       scenario_id,
                                       save_data=True):
        """生成单个场景"""
        # 1. 生成激励
        F = self.simulator.generate_excitation(excitation_type='filtered_noise')
        
        # 2. 计算健康响应
        healthy_response = self.simulator.simulate_response(self.simulator.K0, F)
        
        # 3. 施加损伤并计算损伤响应
        K_damaged = self.simulator.apply_damage(damaged_dofs, severity_ratios)
        damaged_response = self.simulator.simulate_response(K_damaged, F)
        
        # 4. 提取特征
        gvr_features = self.gvr_extractor.extract_gvr_features(damaged_response, healthy_response)
        feature_maps = self.gvr_extractor.generate_gvr_feature_map(gvr_features)
        
        # 5. 生成标签
        labels = np.zeros(self.simulator.num_degrees, dtype=int)
        if damaged_dofs:
            labels[np.array(damaged_dofs)] = 1
            
        # 6. 确定类别
        damage_class = 0
        if len(damaged_dofs) == 0:
            damage_class = 0
        elif len(damaged_dofs) == 1:
            s = severity_ratios[0]
            if s < 0.3: damage_class = 1
            elif s < 0.6: damage_class = 2
            else: damage_class = 3
        else:
            damage_class = 4
            
        # 7. 保存
        if save_data:
            self._save_scenario_data(damaged_response, feature_maps, labels, damage_class, 
                                     damaged_dofs, severity_ratios, scenario_id)
            
        self.metadata.append({
            'scenario_id': scenario_id,
            'damaged_dofs': damaged_dofs,
            'severity_ratios': severity_ratios,
            'damage_class': damage_class,
            'num_samples': feature_maps.shape[0]
        })
        
        return {
            'acceleration': damaged_response,
            'healthy': healthy_response,
            'feature_maps': feature_maps
        }
    
    def _save_scenario_data(self, acc, feat_maps, labels, damage_class, dofs, sevs, sid):
        filename = os.path.join(self.output_dir, f'scenario_{sid:04d}.h5')
        
        # === 核心优化 1: 数据类型转换 (float64 -> float32) ===
        # 特征图和加速度信号转换为 float32，体积减半且不影响深度学习精度
        acc = acc.astype(np.float32)
        feat_maps = feat_maps.astype(np.float32)
        
        # 将 labels 转为最小的整型以节省空间
        labels = labels.astype(np.uint8)
        
        with h5py.File(filename, 'w') as hf:
            # === 核心优化 2: 启用 Gzip 压缩 ===
            # compression='gzip', compression_opts=4 提供了压缩率和速度的平衡
            # chunks=True 让 HDF5 自动选择分块，配合压缩效果最好
            
            # 保存加速度数据
            hf.create_dataset('acceleration', data=acc, 
                             compression='gzip', compression_opts=4)
            
            # 保存特征图 (这是最大的数据源，压缩效果最明显)
            hf.create_dataset('feature_maps', data=feat_maps, 
                             compression='gzip', compression_opts=4)
            
            # 保存标签
            hf.create_dataset('labels', data=labels)
            
            # 保存损伤类别
            hf.create_dataset('damage_class', data=np.array([damage_class], dtype=np.uint8))
            
            hf.attrs['damaged_dofs'] = np.array(dofs)
            hf.attrs['severity_ratios'] = np.array(sevs)
            
            # === 新增：保存滑动窗口参数，供加载数据时使用 ===
            hf.attrs['window_length'] = self.gvr_extractor.window_length
            hf.attrs['step_size'] = self.gvr_extractor.step_size

    
    def generate_comprehensive_dataset(self,
                                      num_scenarios=100,
                                      min_damage_dofs=1,
                                      max_damage_dofs=3,
                                      min_severity=0.15,
                                      max_severity=0.8,
                                      healthy_ratio=0.3):
        print(f"开始生成改进版数据集，总场景数: {num_scenarios}...")
        
        num_healthy = int(num_scenarios * healthy_ratio)
        num_damaged = num_scenarios - num_healthy
        
        print(f"生成健康样本: {num_healthy} ...")
        for i in range(num_healthy):
            self.generate_single_damage_scenario([], [], i, save_data=True)
            
        print(f"生成损伤样本: {num_damaged} ...")
        for i in range(num_damaged):
            sid = num_healthy + i
            num_damage = np.random.randint(min_damage_dofs, max_damage_dofs + 1)
            dofs = np.random.choice(range(self.simulator.num_degrees), num_damage, replace=False).tolist()
            sevs = np.random.uniform(min_severity, max_severity, num_damage).tolist()
            
            self.generate_single_damage_scenario(dofs, sevs, sid, save_data=True)
            
        self._save_metadata()
        print("数据集生成完成。")
        
    def _save_metadata(self):
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)


if __name__ == "__main__":
    dt = 0.005
    duration = 60.0
    
    simulator = ImprovedJacketPlatformSimulator(
        num_degrees=30,
        dt=dt,
        duration=duration,
        damping_ratio=0.05,
        seed=42
    )
    
    gvr_extractor = GVRFeatureExtractor(
        dt=dt,
        window_length=2000,
        step_size=50,
        cutoff_freq=2.0
    )
    
    generator = ImprovedDamageDataGenerator(
        simulator=simulator,
        gvr_extractor=gvr_extractor,
        output_dir='./jacket_damage_data'
    )
    
    generator.generate_comprehensive_dataset(
        num_scenarios=100,
        healthy_ratio=0.3,
        min_severity=0.2,
        max_severity=0.8
    )
    
    print("数据生成完毕，请运行 check_data.py 验证质量。")
