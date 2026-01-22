import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互模式，避免 tkagg 警告
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


class JacketPlatformSimulator:
    """
    海洋导管架平台多自由度仿真系统
    用于生成振动响应数据和GVR特征图
    """
    
    def __init__(self, 
                 num_degrees: int = 30,
                 dt: float = 0.001,
                 duration: float = 30.0,
                 damping_ratio: float = 0.05,
                 seed: Optional[int] = None):
        """
        初始化导管架平台仿真系统
        
        Args:
            num_degrees: 自由度数量
            dt: 时间步长
            duration: 仿真持续时间
            damping_ratio: 阻尼比
            seed: 随机种子
        """
        self.num_degrees = num_degrees
        self.dt = dt
        self.duration = duration
        self.damping_ratio = damping_ratio
        self.num_steps = int(duration / dt)
        self.time = np.linspace(0, duration, self.num_steps)
        
        if seed is not None:
            np.random.seed(seed)
            
        # 初始化系统矩阵
        self.M = None  # 质量矩阵
        self.C = None  # 阻尼矩阵
        self.K = None  # 刚度矩阵
        self.K0 = None # 初始刚度矩阵（用于健康状态）
        
        # 存储健康状态响应
        self.healthy_response = None
        
        # 初始化系统
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化导管架平台的物理参数和系统矩阵"""
        # 构建简化的导管架平台模型
        # 采用多层多自由度集中质量模型
        
        # 假设导管架平台分为多个层级
        self.num_layers = 5  # 导管架层数
        self.dofs_per_layer = self.num_degrees // self.num_layers
        
        # 构建对角质量矩阵（简化模型）
        # 假设每层质量不同，模拟实际结构
        mass_per_layer = np.linspace(1000, 5000, self.num_layers)  # kg
        self.mass_values = np.repeat(mass_per_layer, self.dofs_per_layer)
        self.M = np.diag(self.mass_values)
        
        # 构建刚度矩阵
        self.K = self._build_stiffness_matrix()
        self.K0 = self.K.copy()  # 保存初始刚度矩阵
        
        # 计算固有频率和振型
        self._compute_modal_properties()
        
        # 构建阻尼矩阵（Rayleigh阻尼）
        self._build_damping_matrix()
    
    def _build_stiffness_matrix(self) -> np.ndarray:
        """构建导管架平台的刚度矩阵"""
        n = self.num_degrees
        K = np.zeros((n, n))
        
        # 构建带状刚度矩阵，模拟结构连接
        for i in range(n):
            # 主对角线元素
            K[i, i] = 1e7  # 基础刚度 (N/m)
            
            # 考虑层级差异，刚度随高度变化
            layer_idx = i // self.dofs_per_layer
            K[i, i] *= (1 + 0.2 * layer_idx)  # 上部刚度更大
            
            # 耦合项（模拟结构连接）
            for j in range(n):
                if abs(i - j) <= 2:  # 近邻耦合
                    coupling_strength = 0.1 * np.exp(-abs(i - j))
                    K[i, j] = K[i, i] * coupling_strength
        
        return K
    
    def _compute_modal_properties(self):
        """计算模态属性"""
        # 求解广义特征值问题 (K - λM)φ = 0
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(self.M) @ self.K)
        
        # 固有频率
        self.natural_frequencies = np.sqrt(eigenvalues)
        self.mode_shapes = eigenvectors
        
        print(f"系统固有频率 (前5阶): {self.natural_frequencies[:5] / (2*np.pi)} Hz")
    
    def _build_damping_matrix(self):
        """构建Rayleigh阻尼矩阵 C = αM + βK"""
        # 根据前两阶模态频率计算Rayleigh阻尼系数
        omega1 = self.natural_frequencies[0]
        omega2 = self.natural_frequencies[1]
        
        # Rayleigh阻尼参数
        alpha = 2 * self.damping_ratio * omega1 * omega2 / (omega1 + omega2)
        beta = 2 * self.damping_ratio / (omega1 + omega2)
        
        self.C = alpha * self.M + beta * self.K
        self.rayleigh_params = {'alpha': alpha, 'beta': beta}
    
    def apply_damage(self, 
                     damaged_dofs: List[int], 
                     severity_ratios: List[float]) -> np.ndarray:
        """
        在指定自由度施加损伤
        
        Args:
            damaged_dofs: 受损自由度索引列表
            severity_ratios: 每个受损自由度的刚度降低比例 (0-1)
        
        Returns:
            修改后的刚度矩阵
        """
        K_damaged = self.K.copy()
        
        for dof, severity in zip(damaged_dofs, severity_ratios):
            # 降低刚度矩阵中对应自由度的元素
            K_damaged[dof, dof] *= (1 - severity)
            
            # 同时影响耦合项
            for i in range(self.num_degrees):
                if i != dof:
                    K_damaged[dof, i] *= (1 - severity * 0.5)
                    K_damaged[i, dof] *= (1 - severity * 0.5)
        
        return K_damaged
    
    def generate_excitation(self, 
                          excitation_type: str = 'impact',
                          amplitude_range: Tuple[float, float] = (0, 2000),
                          frequencies: Optional[List[float]] = None) -> np.ndarray:
        """
        生成激励力
        
        Args:
            excitation_type: 激励类型 ('random', 'harmonic', 'impact')
            amplitude_range: 激励幅值范围
            frequencies: 谐波激励频率列表
        
        Returns:
            激励力矩阵 (num_steps × num_degrees)
        """
        n = self.num_degrees
        num_steps = self.num_steps
        
        if excitation_type == 'random':
            # 随机激励
            F = np.random.uniform(amplitude_range[0], amplitude_range[1], (num_steps, n))
            
        elif excitation_type == 'harmonic':
            # 谐波激励
            if frequencies is None:
                # 使用固有频率附近的频率
                frequencies = [self.natural_frequencies[0] * 0.8, 
                             self.natural_frequencies[1] * 0.9]
            
            F = np.zeros((num_steps, n))
            for freq in frequencies:
                omega = 2 * np.pi * freq
                phase = np.random.uniform(0, 2*np.pi, n)
                amplitude = np.random.uniform(amplitude_range[0]*0.5, amplitude_range[1])
                F += amplitude * np.sin(omega * self.time[:, None] + phase[None, :])
                
        elif excitation_type == 'impact':
            # 冲击激励
            F = np.zeros((num_steps, n))
            # 在随机位置施加冲击
            impact_location = np.random.randint(0, n)
            impact_time = np.random.uniform(0.1, 0.3) * self.duration
            impact_idx = int(impact_time / self.dt)
            impact_duration = 50  # 数据点
            
            # 半正弦波冲击
            t_impact = np.linspace(0, np.pi, impact_duration)
            impact_force = np.sin(t_impact) * amplitude_range[1]
            
            end_idx = min(impact_idx + impact_duration, num_steps)
            F[impact_idx:end_idx, impact_location] = impact_force[:end_idx-impact_idx]
            
        else:
            raise ValueError(f"Unknown excitation type: {excitation_type}")
        
        return F
    
    def simulate_response(self, 
                         K: np.ndarray, 
                         F: np.ndarray,
                         initial_conditions: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        使用Newmark-β方法求解结构响应
        
        Args:
            K: 刚度矩阵
            F: 激励力矩阵
            initial_conditions: 初始位移和速度
        
        Returns:
            加速度响应矩阵
        """
        # Newmark-β参数
        gamma = 0.5
        beta = 0.25
        
        n = self.num_degrees
        num_steps = self.num_steps
        
        # 初始化数组
        x = np.zeros((num_steps, n))  # 位移
        v = np.zeros((num_steps, n))  # 速度
        a = np.zeros((num_steps, n))  # 加速度
        
        # 设置初始条件
        if initial_conditions is not None:
            x[0] = initial_conditions[0]
            v[0] = initial_conditions[1]
        else:
            # 计算初始加速度
            a[0] = np.linalg.solve(self.M, F[0] - self.C @ v[0] - K @ x[0])
        
        # 时间积分
        for i in range(num_steps - 1):
            # 预测步骤
            x_tilde = x[i] + self.dt * v[i] + 0.5 * self.dt**2 * (1 - 2*beta) * a[i]
            v_tilde = v[i] + self.dt * (1 - gamma) * a[i]
            
            # 计算有效刚度矩阵
            K_eff = K + gamma/(beta*self.dt) * self.C + 1/(beta*self.dt**2) * self.M
            
            # 计算有效载荷
            F_eff = F[i+1] + self.M @ (1/(beta*self.dt**2) * x_tilde + 1/(beta*self.dt) * v[i]) + \
                    self.C @ (gamma/(beta*self.dt) * x_tilde + (gamma/beta - 1) * v[i])
            
            # 求解位移
            x[i+1] = np.linalg.solve(K_eff, F_eff)
            
            # 计算速度和加速度
            a[i+1] = 1/(beta*self.dt**2) * (x[i+1] - x_tilde) - 1/(beta*self.dt) * v[i]
            v[i+1] = v[i] + self.dt * ((1-gamma) * a[i] + gamma * a[i+1])
        
        return a  # 返回加速度响应
    
    def get_healthy_response(self, 
                           excitation_type: str = 'random',
                           num_simulations: int = 5,
                           save_to_file: bool = False,
                           filename: Optional[str] = None) -> np.ndarray:
        """
        生成健康状态响应并保存
        """
        healthy_responses = []
        
        for _ in range(num_simulations):
            F = self.generate_excitation(excitation_type=excitation_type)
            a = self.simulate_response(self.K0, F)
            healthy_responses.append(a)
        
        self.healthy_response = np.mean(healthy_responses, axis=0)
        
        if save_to_file:
            if filename is None:
                filename = 'healthy_response.mat'
            savemat(filename, {'healthy_response': self.healthy_response})
            print(f"健康状态响应已保存至 {filename}")
        
        return self.healthy_response


class GVRFeatureExtractor:
    """
    梯度变化率（GVR）特征提取器
    基于论文中提出的方法
    """
    
    def __init__(self, 
                 dt: float,  # 必须传入dt参数
                 window_length: int = 3000,
                 step_size: int = 50,
                 cutoff_freq: float = 50.0,
                 filter_order: int = 4):
        """
        初始化GVR特征提取器
        
        Args:
            dt: 时间步长 - 必需参数
            window_length: 滑动窗口长度
            step_size: 滑动窗口步长
            cutoff_freq: 低通滤波截止频率
            filter_order: 滤波器阶数
        """
        self.dt = dt  # 保存dt
        self.window_length = window_length
        self.step_size = step_size
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        
        # 设计Butterworth低通滤波器
        nyquist = 0.5 / self.dt  # 奈奎斯特频率
        self.b, self.a = signal.butter(filter_order, cutoff_freq / nyquist, btype='low')
    
    def butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """应用Butterworth低通滤波器"""
        filtered_data = signal.filtfilt(self.b, self.a, data, axis=0)
        return filtered_data
    
    def compute_damage_index(self, 
                           damaged_signal: np.ndarray, 
                           healthy_signal: np.ndarray) -> np.ndarray:
        """
        计算损伤指数
        
        Args:
            damaged_signal: 损伤状态信号
            healthy_signal: 健康状态信号
        
        Returns:
            损伤指数数组
        """
        num_channels = damaged_signal.shape[1]
        DI = np.zeros(num_channels)
        
        for ch in range(num_channels):
            numerator = np.sum(np.abs(damaged_signal[:, ch] - healthy_signal[:, ch]))
            denominator = np.sum(healthy_signal[:, ch] ** 2) + 1e-10  # 避免除零
            DI[ch] = numerator / denominator
        
        return DI
    
    def compute_gvr(self, DI: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算梯度变化率
        
        Args:
            DI: 损伤指数序列
        
        Returns:
            GVR一阶导数和二阶导数
        """
        # 一阶导数
        DI_prime = np.zeros_like(DI)
        DI_prime[1:] = DI[1:] - DI[:-1]
        
        # 二阶导数绝对值
        DI_double_prime = np.zeros_like(DI)
        DI_double_prime[1:] = np.abs(DI_prime[1:] - DI_prime[:-1])
        
        return DI_prime, DI_double_prime
    
    def extract_gvr_features(self, 
                           damaged_signal: np.ndarray, 
                           healthy_signal: np.ndarray,
                           return_time_series: bool = False) -> Dict:
        """
        提取完整的GVR特征
        
        Args:
            damaged_signal: 损伤状态信号
            healthy_signal: 健康状态信号
            return_time_series: 是否返回时间序列
        
        Returns:
            包含GVR特征的字典
        """
        # 应用低通滤波
        filtered_damaged = self.butterworth_filter(damaged_signal)
        
        # 使用滑动窗口计算DI
        num_steps = damaged_signal.shape[0]
        num_windows = (num_steps - self.window_length) // self.step_size + 1
        
        DI_series = []
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_length
            
            window_damaged = filtered_damaged[start_idx:end_idx]
            # 如果健康数据不够长，使用滤波后的损伤数据作为近似
            if healthy_signal.shape[0] >= end_idx:
                window_healthy = healthy_signal[start_idx:end_idx]
            else:
                window_healthy = filtered_damaged[start_idx:end_idx]
            
            DI_window = self.compute_damage_index(window_damaged, window_healthy)
            DI_series.append(DI_window)
        
        DI_series = np.array(DI_series)
        
        # 计算GVR
        DI_prime, DI_double_prime = self.compute_gvr(DI_series)
        
        # 检测局部极大值
        gvr_peaks = []
        for ch in range(DI_double_prime.shape[1]):
            peaks, _ = signal.find_peaks(DI_double_prime[:, ch], distance=5)
            gvr_peaks.append(peaks)
        
        features = {
            'DI': DI_series,
            'GVR_prime': DI_prime,
            'GVR_double_prime': DI_double_prime,
            'GVR_peaks': gvr_peaks
        }
        
        if return_time_series:
            features['time_series'] = np.arange(num_windows) * self.step_size * self.dt
        
        return features
    
    def generate_gvr_feature_map(self, 
                                gvr_features: Dict,
                                image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        生成GVR特征图（用于CNN输入）
        使用numpy插值方法，兼容所有SciPy版本
        
        Args:
            gvr_features: GVR特征字典
            image_size: 输出图像尺寸
        
        Returns:
            GVR特征图数组
        """
        DI_double_prime = gvr_features['GVR_double_prime']  # (num_windows, num_channels)
        
        num_samples = DI_double_prime.shape[0]
        num_channels = DI_double_prime.shape[1]
        
        # 为每个时间窗口生成特征图
        feature_maps = np.zeros((num_samples, image_size[0], image_size[1], 3))
        
        # 定义目标网格的X轴坐标（图像宽度）
        # 原始数据对应 num_channels 个点，范围是 [0, num_channels-1]
        x_original = np.arange(num_channels)
        x_new = np.linspace(0, num_channels - 1, image_size[1])
        
        for i in range(num_samples):
            # 归一化到[0, 1]
            data_min = DI_double_prime[i].min()
            data_max = DI_double_prime[i].max()
            
            if data_max > data_min:
                normalized = (DI_double_prime[i] - data_min) / (data_max - data_min)
            else:
                normalized = DI_double_prime[i].copy()
            
            # 使用numpy的interp函数进行插值（更稳定且兼容所有版本）
            # np.interp函数：np.interp(x, xp, fp, left=None, right=None, period=None)
            # 其中left和right参数控制外推行为
            try:
                # 使用np.interp进行插值，left/right参数处理外推
                z_row = np.interp(x_new, x_original, normalized, 
                                   left=normalized[0], right=normalized[-1])
            except Exception as e:
                # 如果插值失败（例如点太少），使用最近邻插值
                print(f"Warning: Interpolation failed for sample {i}, using nearest neighbor. Error: {e}")
                # 使用最近邻方法
                indices = np.clip(np.searchsorted(x_original, x_new), 0, len(x_original) - 1)
                z_row = normalized[indices]
            
            # 将一维数组垂直平铺，形成2D图像
            # (image_size[1],) -> (image_size[0], image_size[1])
            img_2d = np.tile(z_row, (image_size[0], 1))
            
            # 创建RGB图像（复制到三个通道）
            img_rgb = np.stack([img_2d, img_2d, img_2d], axis=2)
            feature_maps[i] = img_rgb
        
        return feature_maps


class DamageDataGenerator:
    """
    损伤数据生成器
    用于生成包含多种损仿场景的训练数据
    """
    
    def __init__(self, 
                 simulator: JacketPlatformSimulator,
                 gvr_extractor: GVRFeatureExtractor,
                 output_dir: str = './damage_data'):
        """
        初始化数据生成器
        
        Args:
            simulator: 导管架平台仿真器
            gvr_extractor: GVR特征提取器
            output_dir: 输出目录
        """
        self.simulator = simulator
        self.gvr_extractor = gvr_extractor
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储生成数据的元数据
        self.metadata = []
    
    def generate_single_damage_scenario(self,
                                       damaged_dofs: List[int],
                                       severity_ratios: List[float],
                                       scenario_id: int,
                                       save_data: bool = True) -> Dict:
        """
        生成单个损仿场景的数据
        
        Args:
            damaged_dofs: 受损自由度列表
            severity_ratios: 严重程度列表
            scenario_id: 场景ID
            save_data: 是否保存数据
        
        Returns:
            生成的数据字典
        """
        # 应用损仿
        K_damaged = self.simulator.apply_damage(damaged_dofs, severity_ratios)
        
        # 生成激励
        F = self.simulator.generate_excitation(excitation_type='random')
        
        # 计算响应
        damaged_response = self.simulator.simulate_response(K_damaged, F)
        
        # 提取GVR特征
        if self.simulator.healthy_response is None:
            # 如果没有健康状态数据，生成一个
            self.simulator.get_healthy_response()
        
        gvr_features = self.gvr_extractor.extract_gvr_features(
            damaged_response, 
            self.simulator.healthy_response
        )
        
        # 生成特征图
        feature_maps = self.gvr_extractor.generate_gvr_feature_map(gvr_features)
        
        # 创建标签（二进制指示哪些通道受损）
        labels = np.zeros(self.simulator.num_degrees)
        labels[damaged_dofs] = 1
        
        # 创建损仿类别标签（用于分类）
        damage_class = self._create_damage_class_label(damaged_dofs, severity_ratios)
        
        # 组织数据
        data = {
            'acceleration_data': damaged_response,
            'gvr_features': gvr_features,
            'feature_maps': feature_maps,
            'labels': labels,
            'damage_class': damage_class,
            'damaged_dofs': damaged_dofs,
            'severity_ratios': severity_ratios,
            'scenario_id': scenario_id
        }
        
        # 保存数据
        if save_data:
            self._save_scenario_data(data, scenario_id)
        
        # 记录元数据
        self.metadata.append({
            'scenario_id': scenario_id,
            'damaged_dofs': damaged_dofs,
            'severity_ratios': severity_ratios,
            'damage_class': damage_class,
            'num_samples': feature_maps.shape[0]
        })
        
        return data
    
    def _create_damage_class_label(self, 
                                  damaged_dofs: List[int],
                                  severity_ratios: List[float]) -> int:
        """
        创建损仿类别标签
        用于多分类任务
        """
        # 根据损仿位置和严重程度创建类别
        # 这是一个简化的分类方案
        
        if len(damaged_dofs) == 0:
            return 0  # 健康
        elif len(damaged_dofs) == 1:
            # 单点损仿，根据位置分类
            dof = damaged_dofs[0]
            severity = severity_ratios[0]
            
            if severity < 0.3:
                return 1  # 轻微损仿
            elif severity < 0.6:
                return 2  # 中等损仿
            else:
                return 3  # 严重损仿
        else:
            # 多点损仿
            return 4  # 多点损仿
    
    def _save_scenario_data(self, data: Dict, scenario_id: int):
        """保存单个场景的数据"""
        # 保存为HDF5格式
        filename = os.path.join(self.output_dir, f'scenario_{scenario_id:04d}.h5')
        
        with h5py.File(filename, 'w') as hf:
            # 保存加速度数据
            hf.create_dataset('acceleration', data=data['acceleration_data'])
            
            # 保存特征图
            hf.create_dataset('feature_maps', data=data['feature_maps'])
            
            # 保存标签
            hf.create_dataset('labels', data=data['labels'])
            hf.create_dataset('damage_class', data=np.array([data['damage_class']]))
            
            # 保存元数据
            hf.attrs['damaged_dofs'] = np.array(data['damaged_dofs'])
            hf.attrs['severity_ratios'] = np.array(data['severity_ratios'])
            hf.attrs['scenario_id'] = scenario_id
    
    def generate_comprehensive_dataset(self,
                                      num_scenarios: int = 100,
                                      min_damage_dofs: int = 1,
                                      max_damage_dofs: int = 3,
                                      min_severity: float = 0.2,
                                      max_severity: float = 0.8,
                                      healthy_ratio: float = 0.1) -> Dict:
        """
        生成综合损仿数据集
        
        Args:
            num_scenarios: 总场景数
            min_damage_dofs: 最小损仿自由度数
            max_damage_dofs: 最大损仿自由度数
            min_severity: 最小严重程度
            max_severity: 最大严重程度
            healthy_ratio: 健康样本比例
        
        Returns:
            数据集统计信息
        """
        print(f"开始生成综合数据集，共{num_scenarios}个场景...")
        
        # 计算健康样本数量
        num_healthy = int(num_scenarios * healthy_ratio)
        num_damaged = num_scenarios - num_healthy
        
        # 生成健康样本
        print(f"生成 {num_healthy} 个健康样本...")
        for i in range(num_healthy):
            self.generate_single_damage_scenario(
                damaged_dofs=[],
                severity_ratios=[],
                scenario_id=i,
                save_data=True
            )
        
        # 生成损仿样本
        print(f"生成 {num_damaged} 个损仿样本...")
        for i in range(num_damaged):
            scenario_id = num_healthy + i
            
            # 随机选择损仿位置和严重程度
            num_damage_dofs = np.random.randint(min_damage_dofs, max_damage_dofs + 1)
            damaged_dofs = np.random.choice(
                range(self.simulator.num_degrees), 
                num_damage_dofs, 
                replace=False
            ).tolist()
            
            severity_ratios = np.random.uniform(
                min_severity, 
                max_severity, 
                num_damage_dofs
            ).tolist()
            
            self.generate_single_damage_scenario(
                damaged_dofs=damaged_dofs,
                severity_ratios=severity_ratios,
                scenario_id=scenario_id,
                save_data=True
            )
        
        # 保存元数据
        self._save_metadata()
        
        # 计算统计信息
        stats = self._compute_statistics()
        
        print(f"数据集生成完成！")
        print(f"总场景数: {num_scenarios}")
        print(f"健康样本: {num_healthy}")
        print(f"损仿样本: {num_damaged}")
        print(f"总样本数: {stats['total_samples']}")
        
        return stats
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"元数据已保存至 {metadata_file}")
    
    def _compute_statistics(self) -> Dict:
        """计算数据集统计信息"""
        total_samples = sum(item['num_samples'] for item in self.metadata)
        
        # 统计各类别数量
        class_counts = {}
        for item in self.metadata:
            damage_class = item['damage_class']
            class_counts[damage_class] = class_counts.get(damage_class, 0) + 1
        
        return {
            'total_samples': total_samples,
            'total_scenarios': len(self.metadata),
            'class_distribution': class_counts
        }


class JacketDamageDataset(Dataset):
    """
    导管架平台损仿数据集类
    用于PyTorch训练
    """
    
    def __init__(self, 
                 data_dir: str, 
                 transform=None,
                 use_gvr_maps: bool = True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            transform: 数据变换
            use_gvr_maps: 是否使用GVR特征图
        """
        self.data_dir = data_dir
        self.transform = transform
        self.use_gvr_maps = use_gvr_maps
        
        # 加载元数据
        metadata_file = os.path.join(data_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # 创建样本索引
        self.samples = []
        for item in self.metadata:
            scenario_id = item['scenario_id']
            num_samples = item['num_samples']
            for sample_idx in range(num_samples):
                self.samples.append({
                    'scenario_id': scenario_id,
                    'sample_idx': sample_idx,
                    'damage_class': item['damage_class']
                })
        
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        scenario_id = sample_info['scenario_id']
        sample_idx = sample_info['sample_idx']
        
        # 加载数据
        filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
        
        with h5py.File(filename, 'r') as hf:
            # 加载特征图
            feature_maps = hf['feature_maps'][sample_idx]
            
            # 加载标签
            damage_class = hf['damage_class'][0]
            labels = hf['labels'][:]
        
        # 应用变换
        if self.transform:
            feature_maps = self.transform(feature_maps)
        
        # 转换为PyTorch张量
        feature_maps = torch.from_numpy(feature_maps).float()
        feature_maps = feature_maps.permute(2, 0, 1)  # HWC -> CHW
        
        damage_class = torch.tensor(damage_class, dtype=torch.long)
        labels = torch.from_numpy(labels).float()
        
        return feature_maps, labels, damage_class


def visualize_generated_data(data_generator: DamageDataGenerator, 
                            num_scenarios: int = 4):
    """
    可视化生成的数据
    
    Args:
        data_generator: 数据生成器
        num_scenarios: 要可视化的场景数量
    """
    fig, axes = plt.subplots(num_scenarios, 4, figsize=(20, 5*num_scenarios))
    
    for i in range(num_scenarios):
        if i >= len(data_generator.metadata):
            break
            
        scenario_id = data_generator.metadata[i]['scenario_id']
        
        # 加载数据
        filename = os.path.join(data_generator.output_dir, f'scenario_{scenario_id:04d}.h5')
        with h5py.File(filename, 'r') as hf:
            feature_maps = hf['feature_maps'][:]
            labels = hf['labels'][:]
            damage_class = hf['damage_class'][0]
        
        # 可视化第一个样本的特征图
        feature_map = feature_maps[0]
        
        # 原始特征图
        axes[i, 0].imshow(feature_map)
        axes[i, 0].set_title(f'Scenario {scenario_id}: Feature Map')
        axes[i, 0].axis('off')
        
        # RGB通道
        for c in range(3):
            axes[i, c+1].imshow(feature_map[:, :, c], cmap='hot')
            axes[i, c+1].set_title(f'Channel {c}')
            axes[i, c+1].axis('off')
        
        # 显示标签信息
        damaged_dofs = np.where(labels > 0)[0]
        axes[i, 0].text(0.02, 0.98, 
                       f'Damaged DOFs: {damaged_dofs}\nClass: {damage_class}',
                       transform=axes[i, 0].transAxes,
                       color='white', fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_generator.output_dir, 'data_visualization.png'), dpi=300)
    plt.close()
    
    print(f"数据可视化图已保存")


# 示例使用
if __name__ == "__main__":
    # 设置参数
    dt = 0.001  # 时间步长
    duration = 30.0  # 持续时间
    
    # 1. 创建导管架平台仿真器
    print("初始化导管架平台仿真系统...")
    simulator = JacketPlatformSimulator(
        num_degrees=30,  # 30自由度系统
        dt=dt,
        duration=duration,
        damping_ratio=0.05,
        seed=42
    )
    
    # 2. 创建GVR特征提取器
    print("初始化GVR特征提取器...")
    gvr_extractor = GVRFeatureExtractor(
        dt=dt,  # 传入dt参数
        window_length=3000,
        step_size=50,
        cutoff_freq=50.0,
        filter_order=4
    )
    
    # 3. 生成健康状态响应
    print("生成健康状态响应...")
    simulator.get_healthy_response(
        excitation_type='random',
        num_simulations=5,
        save_to_file=True,
        filename='healthy_response.mat'
    )
    
    # 4. 创建数据生成器
    print("创建数据生成器...")
    data_generator = DamageDataGenerator(
        simulator=simulator,
        gvr_extractor=gvr_extractor,
        output_dir='./jacket_damage_data'
    )
    
    # 5. 生成单个损仿场景（测试）
    print("生成测试损仿场景...")
    try:
        test_scenario = data_generator.generate_single_damage_scenario(
            damaged_dofs=[5, 10, 15],
            severity_ratios=[0.6, 0.4, 0.5],
            scenario_id=0,
            save_data=True
        )
        print("测试场景生成成功！")
    except Exception as e:
        print(f"测试场景生成出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 生成综合数据集
    print("生成综合数据集...")
    try:
        dataset_stats = data_generator.generate_comprehensive_dataset(
            num_scenarios=10,  # 为了快速测试，先生成10个
            min_damage_dofs=1,
            max_damage_dofs=3,
            min_severity=0.2,
            max_severity=0.8,
            healthy_ratio=0.1  # 10%健康样本
        )
    except Exception as e:
        print(f"数据集生成出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 可视化生成的数据
    if len(data_generator.metadata) > 0:
        print("可视化生成的数据...")
        visualize_generated_data(data_generator, num_scenarios=min(4, len(data_generator.metadata)))
        
        # 8. 创建PyTorch数据集
        print("创建PyTorch数据集...")
        dataset = JacketDamageDataset(
            data_dir='./jacket_damage_data',
            use_gvr_maps=True
        )
        
        # 9. 创建数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=4, # 小批次用于测试
            shuffle=True,
            num_workers=0  # Windows下可能需要设置为0避免多进程错误
        )
        
        # 测试数据加载
        print("测试数据加载...")
        for batch_idx, (feature_maps, labels, damage_classes) in enumerate(train_loader):
            print(f"批次 {batch_idx + 1}:")
            print(f"  特征图形状: {feature_maps.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  损仿类别形状: {damage_classes.shape}")
            print(f"  损仿类别示例: {damage_classes}")
            
            if batch_idx >= 0:  # 只打印第一个批次
                break
        
        print("数据生成器创建完成！")
    else:
        print("没有生成数据，跳过可视化。")
