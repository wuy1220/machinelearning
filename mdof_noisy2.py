import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
import h5py
import os
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import json
import random

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
                          amplitude: float = 50000.0,
                          **kwargs) -> np.ndarray:
        """生成激励力"""
        n = self.num_degrees
        num_steps = self.num_steps
        
        if excitation_type == 'filtered_noise':
            raw_noise = np.random.randn(num_steps, n)
            
            fs = 1.0 / self.dt
            nyquist = 0.5 * fs
            low = 0.15 / nyquist
            high = 3.5 / nyquist  # 专注于低频，避免高频噪声
            
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

    def add_realistic_noise(self, signal, snr_db=40, add_powerline=True, 
                        powerline_freq=50, amplitude_ratio=0.005, seed=None):
        """
        添加真实工况噪声组合
        
        Args:
            signal: 原始加速度响应 (num_steps, num_channels)
            snr_db: 测量噪声信噪比(dB)
            add_powerline: 是否添加工频干扰
            powerline_freq: 工频(Hz)
            amplitude_ratio: 工频干扰相对幅值
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 1. 高斯测量噪声
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        signal = signal + np.random.randn(*signal.shape) * np.sqrt(noise_power)
        
        # 2. 工频干扰
        if add_powerline:
            fs = 1.0 / self.dt
            t = np.arange(signal.shape[0]) * self.dt
            interference = amplitude_ratio * np.sin(2 * np.pi * powerline_freq * t)
            signal = signal + interference[:, np.newaxis]
        
        return signal

class TimeStackedGVRFeatureExtractor:
    """
    时序堆叠GVR特征提取器
    生成时序-空间融合的特征图，充分利用CNN的二维卷积能力
    """
    
    def __init__(self, dt, window_length=3000, step_size=50, 
                 num_stack_windows=100, cutoff_freq=5.0):
        """
        Args:
            dt: 采样时间间隔（秒）
            window_length: 滑动窗口长度
            step_size: 滑动窗口步长
            num_stack_windows: 用于堆叠的时间窗口数量（建议224以匹配图像高度）
            cutoff_freq: 低通滤波截止频率
        """
        self.dt = dt
        self.window_length = window_length
        self.step_size = step_size
        self.num_stack_windows = num_stack_windows
        self.cutoff_freq = cutoff_freq
        
        # 设计Butterworth低通滤波器
        nyquist = 0.5 / self.dt
        self.b, self.a = signal.butter(4, cutoff_freq / nyquist, btype='low')
    
    def butterworth_filter(self, data):
        """Butterworth低通滤波器"""
        return signal.filtfilt(self.b, self.a, data, axis=0)
    
    def compute_damage_index(self, damaged_signal, healthy_signal):
        """
        计算损伤指标DI (Damage Index)
        
        公式：DI_j = √[Σ(xd_ij - xh_ij)²] / √[Σ(xh_ij)² + ε]
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
        提取GVR特征（基础版本，生成DI序列）
        """
        filtered_damaged = self.butterworth_filter(damaged_signal)
        filtered_healthy = self.butterworth_filter(healthy_signal)
        
        # 对健康样本添加微小噪声（避免零信号问题）
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
    
    def extract_stacked_gvr_features(self, damaged_signal, healthy_signal):
        """
        提取连续时间窗口的GVR特征，形成时序序列
        
        Returns:
            stacked_features: 时序特征列表
            num_samples: 生成的样本数
        """
        # 基础特征提取
        base_features = self.extract_gvr_features(damaged_signal, healthy_signal)
        
        # 获取所有窗口的DI、DI'、DI''
        DI_series = base_features['DI']
        DI_prime_series = base_features['GVR_prime']
        DI_double_prime_series = base_features['GVR_double_prime']
        
        num_windows = DI_series.shape[0]
        
        # 窗口不足时进行填充
        if num_windows < self.num_stack_windows:
            padding = self.num_stack_windows - num_windows
            DI_series = np.pad(DI_series, ((0, padding), (0, 0)), mode='edge')
            DI_prime_series = np.pad(DI_prime_series, ((0, padding), (0, 0)), mode='edge')
            DI_double_prime_series = np.pad(DI_double_prime_series, ((0, padding), (0, 0)), mode='edge')
        
        # 形成时序序列：滑动窗口采样
        stacked_features = []
        num_samples = max(1, num_windows - self.num_stack_windows + 1)
        step_size_for_image = self.num_stack_windows // 4  # 重叠采样，增加样本数

        for start_idx in range(0, num_samples, step_size_for_image):
            end_idx = start_idx + self.num_stack_windows
            
            DI_seq = DI_series[start_idx:end_idx]
            DI_prime_seq = DI_prime_series[start_idx:end_idx]
            DI_double_prime_seq = DI_double_prime_series[start_idx:end_idx]
            
            stacked_features.append({
                'DI_seq': DI_seq,
                'DI_prime_seq': DI_prime_seq,
                'DI_double_prime_seq': DI_double_prime_seq
            })
        
        return stacked_features, num_samples
    
    def generate_time_space_feature_map(self, stacked_features, image_size=(224, 224)):
        """
        生成时序-空间融合的特征图
        
        图像结构：
        - X轴（224）：传感器空间位置（插值后）
        - Y轴（224）：时间演进（连续的224个时间窗口）
        - R通道：DI'(变化趋势)的时序演化
        - G通道：DI''(突变特征)的时序演化
        - B通道：DI(损伤强度)的时序演化
        """
        feature_maps = []
        
        for features in stacked_features:
            DI_seq = features['DI_seq']  # (num_stack_windows, num_channels)
            DI_prime_seq = features['DI_prime_seq']
            DI_double_prime_seq = features['DI_double_prime_seq']
            
            num_windows, num_channels = DI_seq.shape
            
            # 每个通道独立归一化（归一化整个时序）
            def normalize_sequence(seq):
                seq_min = seq.min()
                seq_max = seq.max()
                seq_range = seq_max - seq_min
                if seq_range < 1e-10:
                    return np.zeros_like(seq)
                return (seq - seq_min) / seq_range
            
            DI_seq_norm = normalize_sequence(DI_seq)
            DI_prime_seq_norm = normalize_sequence(DI_prime_seq)
            DI_double_prime_seq_norm = normalize_sequence(DI_double_prime_seq)
            
            # 插值到图像宽度（X轴）
            x_original = np.arange(num_channels)
            x_new = np.linspace(0, num_channels - 1, image_size[1])
            
            # 对每个时间窗口进行插值
            DI_interp = np.zeros((num_windows, image_size[1]))
            DI_prime_interp = np.zeros((num_windows, image_size[1]))
            DI_double_prime_interp = np.zeros((num_windows, image_size[1]))
            
            for t in range(num_windows):
                DI_interp[t] = np.interp(x_new, x_original, DI_seq_norm[t])
                DI_prime_interp[t] = np.interp(x_new, x_original, DI_prime_seq_norm[t])
                DI_double_prime_interp[t] = np.interp(x_new, x_original, DI_double_prime_seq_norm[t])
            
            # 处理Y轴尺寸匹配
            if num_windows < image_size[0]:
                # 在Y轴插值
                y_original = np.arange(num_windows)
                y_new = np.linspace(0, num_windows - 1, image_size[0])
                
                img_r = np.zeros((image_size[0], image_size[1]))
                img_g = np.zeros((image_size[0], image_size[1]))
                img_b = np.zeros((image_size[0], image_size[1]))
                
                for x in range(image_size[1]):
                    img_r[:, x] = np.interp(y_new, y_original, DI_prime_interp[:, x])
                    img_g[:, x] = np.interp(y_new, y_original, DI_double_prime_interp[:, x])
                    img_b[:, x] = np.interp(y_new, y_original, DI_interp[:, x])
            elif num_windows > image_size[0]:
                # 下采样
                indices = np.linspace(0, num_windows - 1, image_size[0], dtype=int)
                img_r = DI_prime_interp[indices]
                img_g = DI_double_prime_interp[indices]
                img_b = DI_interp[indices]
            else:
                img_r = DI_prime_interp
                img_g = DI_double_prime_interp
                img_b = DI_interp
            
            # 添加传感器位置标记（竖线）
            sensor_spacing = image_size[1] // num_channels if num_channels > 0 else 1
            for x in range(0, image_size[1], sensor_spacing):
                img_r[:, x] = np.minimum(img_r[:, x] * 0.9, 1.0)
            
            # 添加时间维度标记（底部更亮）
            time_gradient = np.linspace(0, 1, image_size[0]).reshape(-1, 1)
            img_b = img_b * (0.8 + 0.2 * time_gradient)
            
            img_rgb = np.stack([img_r, img_g, img_b], axis=2)
            feature_maps.append(img_rgb)
        
        return np.array(feature_maps, dtype=np.float32)


class ImprovedDamageDataGenerator:
    """
    改进的损伤数据生成器
    集成时序堆叠GVR特征提取器
    """
    
    def __init__(self, simulator, gvr_extractor, output_dir='./jacket_damage_data_improved'):
        """
        Args:
            simulator: 仿真器实例
            gvr_extractor: 必须是TimeStackedGVRFeatureExtractor实例
            output_dir: 输出目录
        """
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
        """
        生成单个损伤场景
        
        Args:
            damaged_dofs: 损伤自由度列表
            severity_ratios: 损伤严重程度列表
            scenario_id: 场景ID
            save_data: 是否保存数据
        """
        # 采用动态且可复现的种子，防止相同噪声模式导致数据泄漏
        current_seed_healthy = 42000 + scenario_id * 2
        current_seed_damaged = 24 + scenario_id

        # 1. 生成激励和响应
        F1 = self.simulator.generate_excitation(excitation_type='filtered_noise', seed=current_seed_healthy)
        F2 = self.simulator.generate_excitation(excitation_type='filtered_noise', seed=current_seed_damaged)
        healthy_response = self.simulator.simulate_response(self.simulator.K0, F1)
        K_damaged = self.simulator.apply_damage(damaged_dofs, severity_ratios)
        damaged_response = self.simulator.simulate_response(K_damaged, F2)
        
        # 1.5. 添加噪声
        # mems加速度计的snr在40-80db之间，顶级光纤加速度计可达110db+
        snr_db = random.randint(75, 85)
        healthy_response = self.simulator.add_realistic_noise(healthy_response, snr_db=snr_db, seed=current_seed_healthy, amplitude_ratio=0.0003) 
        damaged_response = self.simulator.add_realistic_noise(damaged_response, snr_db=snr_db, seed=current_seed_damaged, amplitude_ratio=0.0003) 
        
        # 2. 使用时序堆叠提取特征
        stacked_features, num_samples = self.gvr_extractor.extract_stacked_gvr_features(
            damaged_response, healthy_response
        )
        
        # 3. 生成时序-空间特征图
        feature_maps = self.gvr_extractor.generate_time_space_feature_map(stacked_features)
        
        # 4. 生成标签
        labels = np.zeros(self.simulator.num_degrees, dtype=int)
        if damaged_dofs:
            labels[np.array(damaged_dofs)] = 1
        
        # 扩展标签以匹配样本数
        labels_array = np.tile(labels, (num_samples, 1))
        
        # 5. 确定损伤类别
        damage_class = self._determine_damage_class(damaged_dofs, severity_ratios)

        # 6. 保存数据
        if save_data:
            # 计算需要保存的加速度数据长度
            total_steps_needed = (num_samples - 1) * self.gvr_extractor.step_size + \
                               self.gvr_extractor.num_stack_windows * self.gvr_extractor.window_length
            acc_to_save = damaged_response[:total_steps_needed]
            
            self._save_scenario_data(
                acc_to_save,
                feature_maps,
                labels_array[0],
                damage_class,
                damaged_dofs,
                severity_ratios,
                scenario_id
            )
        
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
            'feature_maps': feature_maps,
            'labels': labels_array
        }
    
    def _determine_damage_class(self, damaged_dofs, severity_ratios):
        """确定损伤类别"""
        if len(damaged_dofs) == 0:
            return 0  # 健康
        else:
            return 1
        '''
        elif len(damaged_dofs) == 1:
            s = severity_ratios[0]
            if s < 0.3:
                return 1  # 轻度损伤
            elif s < 0.6:
                return 2  # 中度损伤
            else:
                return 3  # 重度损伤
        else:
            return 4  # 多处损伤
        '''

    
    def _save_scenario_data(self, acc, feat_maps, labels, damage_class, dofs, sevs, sid):
        """保存场景数据到HDF5文件"""
        filename = os.path.join(self.output_dir, f'scenario_{sid:04d}.h5')
        
        # 数据类型转换（节省空间）
        acc = acc.astype(np.float32)
        feat_maps = feat_maps.astype(np.float32)
        labels = labels.astype(np.uint8)
        
        with h5py.File(filename, 'w') as hf:
            # 保存加速度数据（带压缩）
            hf.create_dataset('acceleration', data=acc, 
                             compression='gzip', compression_opts=4)
            
            # 保存特征图（带压缩）
            hf.create_dataset('feature_maps', data=feat_maps, 
                             compression='gzip', compression_opts=4)
            
            # 保存标签
            hf.create_dataset('labels', data=labels)
            
            # 保存损伤类别
            hf.create_dataset('damage_class', data=np.array([damage_class], dtype=np.uint8))
            
            # 保存属性
            hf.attrs['damaged_dofs'] = np.array(dofs)
            hf.attrs['severity_ratios'] = np.array(sevs)
            hf.attrs['window_length'] = self.gvr_extractor.window_length
            hf.attrs['step_size'] = self.gvr_extractor.step_size
            hf.attrs['num_stack_windows'] = self.gvr_extractor.num_stack_windows

    def generate_comprehensive_dataset(self,
                                      num_scenarios=100,
                                      min_damage_dofs=1,
                                      max_damage_dofs=3,
                                      min_severity=0.15,
                                      max_severity=0.8,
                                      healthy_ratio=0.3):
        """
        生成综合数据集
        
        Args:
            num_scenarios: 总场景数
            min_damage_dofs: 最小损伤位置数
            max_damage_dofs: 最大损伤位置数
            min_severity: 最小损伤严重程度
            max_severity: 最大损伤严重程度
            healthy_ratio: 健康样本比例
        """
        print(f"开始生成改进版数据集（时序-空间融合），总场景数: {num_scenarios}...")
        print(f"堆叠窗口数: {self.gvr_extractor.num_stack_windows}")
        print(f"特征图尺寸: (224, 224, 3) - X轴=空间，Y轴=时序")
        
        num_healthy = int(num_scenarios * healthy_ratio)
        num_damaged = num_scenarios - num_healthy
        
        # 生成健康样本
        print(f"\n生成健康样本: {num_healthy} ...")
        for i in tqdm(range(num_healthy), desc="健康样本"):
            self.generate_single_damage_scenario([], [], i, save_data=True)
        
        # 生成损伤样本
        print(f"\n生成损伤样本: {num_damaged} ...")
        for i in tqdm(range(num_damaged), desc="损伤样本"):
            sid = num_healthy + i
            num_damage = np.random.randint(min_damage_dofs, max_damage_dofs + 1)
            dofs = np.random.choice(range(self.simulator.num_degrees), 
                                   num_damage, replace=False).tolist()
            sevs = np.random.uniform(min_severity, max_severity, num_damage).tolist()
            
            self.generate_single_damage_scenario(dofs, sevs, sid, save_data=True)
        
        self._save_metadata()
        print("\n数据集生成完成！")
        print(f"输出目录: {self.output_dir}")
        print(f"总场景数: {num_scenarios}")
        
    def _save_metadata(self):
        """保存元数据到JSON文件"""
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)


if __name__ == "__main__":
    # ===== 参数配置 =====
    dt = 0.005
    duration = 60.0
    num_degrees = 10  # 自由度数量（传感器数量）
    
    # 初始化仿真器
    print("=" * 60)
    print("初始化导管架平台仿真系统")
    print("=" * 60)
    simulator = ImprovedJacketPlatformSimulator(
        num_degrees=num_degrees,
        dt=dt,
        duration=duration,
        damping_ratio=0.05,  # 阻尼比
        seed=42
    )
    
    # 初始化时序堆叠GVR特征提取器
    print("\n初始化时序堆叠GVR特征提取器")
    print("=" * 60)
    gvr_extractor = TimeStackedGVRFeatureExtractor(
        dt=dt,
        window_length=2000,          # 单窗口长度
        step_size=50,                # 滑动步长
        num_stack_windows=112,       # 堆叠窗口数（=图像高度）
        cutoff_freq=8.0             # 滤波截止频率
    )
    
    print(f"配置参数:")
    print(f"  - 单窗口长度: {gvr_extractor.window_length} 点")
    print(f"  - 滑动步长: {gvr_extractor.step_size} 点")
    print(f"  - 堆叠窗口数: {gvr_extractor.num_stack_windows} (图像高度)")
    print(f"  - 滤波截止频率: {gvr_extractor.cutoff_freq} Hz")
    
    # 初始化数据生成器
    print("\n初始化数据生成器")
    print("=" * 60)
    generator = ImprovedDamageDataGenerator(
        simulator=simulator,
        gvr_extractor=gvr_extractor,
        output_dir='./jacket_damage_data_timespace'
    )
    
    # 生成数据集
    print("\n开始生成数据集")
    print("=" * 60)
    generator.generate_comprehensive_dataset(
        num_scenarios=2500,
        healthy_ratio=0.4,
        min_severity=0.1,  # 测试模型能力，如果跑通再调小
        max_severity=0.5
    )
    
    print("\n" + "=" * 60)
    print("数据生成完毕！")
    print("=" * 60)
    print("\n特征图说明:")
    print("  - X轴 (224px): 传感器空间位置 (1-30)")
    print("  - Y轴 (224px): 时间演进 (连续224个时间窗口)")
    print("  - R通道: DI' (变化趋势) 的时序演化")
    print("  - G通道: DI'' (突变特征) 的时序演化")
    print("  - B通道: DI (损伤强度) 的时序演化")
