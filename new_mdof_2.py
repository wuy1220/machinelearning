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

class ImprovedJacketPlatformSimulator:
    """
    改进的海洋导管架平台多自由度仿真系统 (剪切型模型)
    旨在生成具有显著损伤特征的振动响应数据
    """
    
    def __init__(self, 
                 num_degrees: int = 30,
                 dt: float = 0.005,  # 增大时间步长以提高生成速度，同时保持精度
                 duration: float = 60.0, # 增加时长以获得更稳定的统计特征
                 damping_ratio: float = 0.05,
                 seed: Optional[int] = None):
        """
        初始化改进的导管架平台仿真系统
        
        Args:
            num_degrees: 自由度数量（串联剪切型）
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
            
        # 系统矩阵
        self.M = None  
        self.C = None  
        self.K = None  
        self.K0 = None 
        
        # 层间刚度参数 (用于剪切模型)
        self.layer_stiffness = None
        
        # 初始化系统
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化基于剪切模型的系统矩阵"""
        # 1. 构建质量矩阵 (对角矩阵)
        # 假设每层质量递减，模拟真实海洋平台顶部质量较小
        # 典型海洋平台单层质量约为 50,000 - 100,000 kg
        base_mass = 80000.0 
        mass_distribution = np.linspace(1.2, 1.0, self.num_degrees)
        self.mass_values = base_mass * mass_distribution
        self.M = np.diag(self.mass_values)
        
        # 2. 构建层间刚度 (用于剪切模型)
        # 典型层间刚度约为 1.0e7 - 1.0e8 N/m
        # 我们设置刚度使得固有频率集中在 0.5 - 3.0 Hz (真实海洋平台范围)
        base_stiffness = 2.0e7 
        # 添加微小的随机扰动，使结构不完全均匀
        perturbation = np.random.uniform(0.9, 1.1, self.num_degrees + 1)
        # 注意：num_degrees个节点，需要 num_degrees + 1 个弹簧（包含底部地基和顶部连接）
        # 或者简化为每层之间一个弹簧，共 num_degrees 个自由度，num_degrees 个弹簧
        self.layer_stiffness = base_stiffness * perturbation[:self.num_degrees]
        
        # 3. 构建总体刚度矩阵 (三对角矩阵)
        self.K = self._build_stiffness_matrix_from_layers()
        self.K0 = self.K.copy()  # 保存健康状态刚度
        
        # 4. 计算模态属性
        self._compute_modal_properties()
        
        # 5. 构建阻尼矩阵
        self._build_damping_matrix()
        
        print(f"系统初始化完成。固有频率范围: {self.natural_frequencies[0]/(2*np.pi):.2f} - {self.natural_frequencies[-1]/(2*np.pi):.2f} Hz")

    def _build_stiffness_matrix_from_layers(self) -> np.ndarray:
        """
        根据层间刚度构建剪切型结构的总刚度矩阵
        K_ij 物理意义: 
        - K_ii = k_i + k_{i+1} (节点连接的上下弹簧刚度之和)
        - K_i(i+1) = -k_{i+1} (耦合项)
        """
        n = self.num_degrees
        K = np.zeros((n, n))
        
        # 边界条件处理：假设底部固定 (自由度0连接地基k0)
        # 这里简化模型：节点i由弹簧 i (下方) 和 i+1 (上方) 连接
        # 节点 0: 连接地基(假设无穷大) 和 节点1
        
        for i in range(n):
            # 对角线元素
            # k_down = self.layer_stiffness[i] if i > 0 else self.layer_stiffness[i] # 简化：每个节点i对应下方弹簧i
            # 修正：标准的层模型。
            # 节点 i 的刚度由 k_i (与i-1连接) 和 k_{i+1} (与i+1连接) 组成
            k_down = self.layer_stiffness[i] if i < n else 0
            k_up = self.layer_stiffness[i+1] if i < n - 1 else 0
            
            # 简化串联模型：每个自由度i由k_i支撑，连接i-1和i
            if i == 0:
                 K[i, i] = self.layer_stiffness[i] # 节点0连接地基k0
            elif i == n - 1:
                 K[i, i] = self.layer_stiffness[i] # 顶部节点只受下方支撑
            else:
                 K[i, i] = self.layer_stiffness[i] + self.layer_stiffness[i+1]
                 
            # 非对角线元素
            if i > 0:
                K[i, i-1] = -self.layer_stiffness[i]
                K[i-1, i] = -self.layer_stiffness[i]
                
        return K
    
    def _compute_modal_properties(self):
        """计算模态属性"""
        # 求解广义特征值问题
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(self.M) @ self.K)
            # 确保特征值为正 (数值稳定性)
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
        # 只对前两阶模态设置目标阻尼比
        if len(self.natural_frequencies) >= 2:
            omega1 = self.natural_frequencies[0]
            omega2 = self.natural_frequencies[1]
            
            # Rayleigh阻尼参数
            alpha = 2 * self.damping_ratio * omega1 * omega2 / (omega1 + omega2)
            beta = 2 * self.damping_ratio / (omega1 + omega2)
            
            self.C = alpha * self.M + beta * self.K
        else:
            # 如果只有一个模态，使用比例阻尼
            self.C = 2 * self.damping_ratio * np.sqrt(self.K[0,0] / self.M[0,0]) * self.M

    def apply_damage(self, 
                     damaged_dofs: List[int], 
                     severity_ratios: List[float]) -> np.ndarray:
        """
        施加损伤：降低指定自由度对应的层间刚度
        
        Args:
            damaged_dofs: 受损自由度索引列表
            severity_ratios: 每个受损位置的刚度降低比例 (0-1)
        
        Returns:
            修改后的刚度矩阵
        """
        K_damaged = self.K0.copy()
        
        for dof, severity in zip(damaged_dofs, severity_ratios):
            if dof < self.num_degrees:
                # 修改层间刚度
                # 注意：降低刚度会导致固有频率下降，这是物理上的关键变化
                original_k = self.layer_stiffness[dof]
                new_k = original_k * (1.0 - severity)
                self.layer_stiffness[dof] = new_k
        
        # 根据修改后的层刚度重新构建总刚度矩阵
        # 这保证了矩阵的对称性和物理一致性
        K_damaged = self._build_stiffness_matrix_from_layers()
        
        return K_damaged
    
    def generate_excitation(self, 
                          excitation_type: str = 'filtered_noise',
                          amplitude: float = 200000.0,  # 力幅值 (N)
                          **kwargs) -> np.ndarray:
        """
        生成激励力 (带限噪声，避免过大共振)
        
        Args:
            excitation_type: 激励类型 ('filtered_noise', 'harmonic', 'random')
            amplitude: 激励力幅值 (牛顿)
        
        Returns:
            激励力矩阵 (num_steps × num_degrees)
        """
        n = self.num_degrees
        num_steps = self.num_steps
        
        if excitation_type == 'filtered_noise':
            # 生成白噪声
            raw_noise = np.random.randn(num_steps, n)
            
            # 设计带通滤波器 (0.1 Hz - 5.0 Hz)
            # 覆盖结构的基频范围，但避开极高频
            fs = 1.0 / self.dt
            nyquist = 0.5 * fs
            low = 0.1 / nyquist
            high = min(5.0 / nyquist, 0.99)
            b, a = signal.butter(4, [low, high], btype='band')
            
            # 对每一列进行滤波
            F = np.zeros_like(raw_noise)
            for i in range(n):
                F[:, i] = signal.filtfilt(b, a, raw_noise[:, i]) * amplitude
                
        elif excitation_type == 'harmonic':
            # 谐波激励 (避开固有频率以观察瞬态，或接近固有频率以观察稳态)
            # 这里选择在基频附近扫频
            freq_start = self.natural_frequencies[0] * 0.9
            freq_end = self.natural_frequencies[0] * 1.1
            
            F = np.zeros((num_steps, n))
            t = self.time
            
            # 线性调频信号
            instantaneous_freq = freq_start + (freq_end - freq_start) * t / self.duration
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) * self.dt
            
            # 只在底部施加激励
            F[:, 0] = amplitude * np.sin(phase)
            
        elif excitation_type == 'random':
            # 简单随机激励 (保留兼容性，但幅值受控)
            F = np.random.randn(num_steps, n) * amplitude * 0.1
            
        else:
            raise ValueError(f"Unknown excitation type: {excitation_type}")
        
        return F

    def simulate_response(self, 
                         K: np.ndarray, 
                         F: np.ndarray) -> np.ndarray:
        """
        使用Newmark-β方法求解结构响应
        
        Args:
            K: 刚度矩阵
            F: 激励力矩阵
        
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
        
        # 初始计算
        a[0] = np.linalg.solve(self.M, F[0] - self.C @ v[0] - K @ x[0])
        
        # 预计算有效刚度矩阵 (假设K不变，若K随时间变化需每步更新)
        K_eff = K + gamma/(beta*self.dt) * self.C + 1/(beta*self.dt**2) * self.M
        
        # 时间积分
        for i in range(num_steps - 1):
            # 预测
            x_tilde = x[i] + self.dt * v[i] + 0.5 * self.dt**2 * (1 - 2*beta) * a[i]
            v_tilde = v[i] + self.dt * (1 - gamma) * a[i]
            
            # 有效载荷
            F_eff = F[i+1] + self.M @ (1/(beta*self.dt**2) * x_tilde + 1/(beta*self.dt) * v[i]) + \
                    self.C @ (gamma/(beta*self.dt) * x_tilde + (gamma/beta - 1) * v[i])
            
            # 求解
            x[i+1] = np.linalg.solve(K_eff, F_eff)
            
            # 更新
            a[i+1] = 1/(beta*self.dt**2) * (x[i+1] - x_tilde) - 1/(beta*self.dt) * v[i]
            v[i+1] = v[i] + self.dt * ((1-gamma) * a[i] + gamma * a[i+1])
        
        return a


class GVRFeatureExtractor:
    """
    GVR 特征提取器 (保持不变，但输入数据质量提升)
    """
    def __init__(self, dt, window_length=3000, step_size=50, cutoff_freq=5.0, filter_order=4):
        self.dt = dt
        self.window_length = window_length
        self.step_size = step_size
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        
        nyquist = 0.5 / self.dt
        self.b, self.a = signal.butter(filter_order, cutoff_freq / nyquist, btype='low')
    
    def butterworth_filter(self, data):
        return signal.filtfilt(self.b, self.a, data, axis=0)
    
    def compute_damage_index(self, damaged_signal, healthy_signal):
        num_channels = damaged_signal.shape[1]
        DI = np.zeros(num_channels)
        for ch in range(num_channels):
            # 使用欧氏距离作为损伤指标
            numerator = np.sum((damaged_signal[:, ch] - healthy_signal[:, ch])**2)
            denominator = np.sum(healthy_signal[:, ch]**2) + 1e-10
            DI[ch] = np.sqrt(numerator) / (np.sqrt(denominator) + 1e-10)
        return DI
    
    def compute_gvr(self, DI):
        DI_prime = np.zeros_like(DI)
        DI_prime[1:] = DI[1:] - DI[:-1]
        DI_double_prime = np.zeros_like(DI)
        DI_double_prime[1:] = np.abs(DI_prime[1:] - DI_prime[:-1])
        return DI_prime, DI_double_prime
    
    def extract_gvr_features(self, damaged_signal, healthy_signal):
        # 优化：如果信号太长，先进行下采样处理以提高速度
        # 这里保持原样，因为物理模型改进后特征应该更明显
        
        filtered_damaged = self.butterworth_filter(damaged_signal)
        # 注意：健康信号也应该使用同样的滤波器
        filtered_healthy = self.butterworth_filter(healthy_signal)
        
        num_steps = damaged_signal.shape[0]
        num_windows = (num_steps - self.window_length) // self.step_size + 1
        
        DI_series = []
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_length
            
            window_damaged = filtered_damaged[start_idx:end_idx]
            window_healthy = filtered_healthy[start_idx:end_idx] if end_idx <= filtered_healthy.shape[0] else window_damaged
            
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
        DI_double_prime = gvr_features['GVR_double_prime'] 
        num_samples = DI_double_prime.shape[0]
        num_channels = DI_double_prime.shape[1]
        feature_maps = np.zeros((num_samples, image_size[0], image_size[1], 3))
        
        # 使用更稳健的归一化方式
        # 针对每个样本独立归一化，增强对比度
        for i in range(num_samples):
            data = DI_double_prime[i]
            
            # 1. 全局归一化
            data_min = data.min()
            data_max = data.max()
            
            if data_max - data_min > 1e-10:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(data) # 如果数据全为常数
            
            # 2. 插值到图像宽度
            x_original = np.arange(num_channels)
            x_new = np.linspace(0, num_channels - 1, image_size[1])
            
            try:
                z_row = np.interp(x_new, x_original, normalized, left=normalized[0], right=normalized[-1])
            except:
                z_row = normalized # Fallback
            
            # 3. 垂直平铺
            img_2d = np.tile(z_row, (image_size[0], 1))
            
            # 4. RGB 通道 (这里使用热图着色的模拟，或者直接复制)
            # 为了让特征图更丰富，可以将 GVR 二阶导数映射到不同通道
            # 这里简单复制三通道保持灰度，但您可以尝试伪彩色映射
            img_rgb = np.stack([img_2d, img_2d, img_2d], axis=2)
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
        """
        生成单个场景
        关键修正：确保健康基准和损伤响应使用完全相同的激励
        """
        # 1. 生成激励 (带限噪声，更适合激起模态且不爆炸)
        F = self.simulator.generate_excitation(excitation_type='filtered_noise')
        
        # 2. 计算健康响应 (使用当前激励)
        healthy_response = self.simulator.simulate_response(self.simulator.K0, F)
        
        # 3. 施加损伤并计算损伤响应 (使用同一个激励)
        K_damaged = self.simulator.apply_damage(damaged_dofs, severity_ratios)
        damaged_response = self.simulator.simulate_response(K_damaged, F)
        
        # 4. 提取特征
        gvr_features = self.gvr_extractor.extract_gvr_features(damaged_response, healthy_response)
        feature_maps = self.gvr_extractor.generate_gvr_feature_map(gvr_features)
        
        # 5. 生成标签 (修正：确保数据类型正确)
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
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('acceleration', data=acc)
            hf.create_dataset('feature_maps', data=feat_maps)
            hf.create_dataset('labels', data=labels)
            hf.create_dataset('damage_class', data=np.array([damage_class]))
            hf.attrs['damaged_dofs'] = np.array(dofs)
            hf.attrs['severity_ratios'] = np.array(sevs)
    
    def generate_comprehensive_dataset(self,
                                      num_scenarios=100,
                                      min_damage_dofs=1,
                                      max_damage_dofs=3,
                                      min_severity=0.3,
                                      max_severity=0.8,
                                      healthy_ratio=0.3): # 增加健康样本比例
        print(f"开始生成改进版数据集，总场景数: {num_scenarios}...")
        
        num_healthy = int(num_scenarios * healthy_ratio)
        num_damaged = num_scenarios - num_healthy
        
        # 生成健康样本
        print(f"生成健康样本: {num_healthy} ...")
        for i in range(num_healthy):
            self.generate_single_damage_scenario([], [], i, save_data=True)
            
        # 生成损伤样本
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


# 主程序执行
if __name__ == "__main__":
    # 1. 初始化改进的仿真器
    dt = 0.005
    duration = 60.0
    
    simulator = ImprovedJacketPlatformSimulator(
        num_degrees=30,
        dt=dt,
        duration=duration,
        damping_ratio=0.05,
        seed=42
    )
    
    # 2. 初始化特征提取器
    gvr_extractor = GVRFeatureExtractor(
        dt=dt,
        window_length=2000, # 稍微缩短窗口以适应更长的信号
        step_size=50,
        cutoff_freq=2.0  # 降低截止频率，专注于低频结构模态
    )
    
    # 3. 生成数据
    generator = ImprovedDamageDataGenerator(
        simulator=simulator,
        gvr_extractor=gvr_extractor,
        output_dir='./jacket_damage_data'
    )
    
    # 4. 运行生成 (建议先少量测试)
    generator.generate_comprehensive_dataset(
        num_scenarios=20,
        healthy_ratio=0.3, # 增加健康样本到30%
        min_severity=0.3,   # 确保损伤不是太轻微
        max_severity=0.8
    )
    
    print("数据生成完毕，请运行 check_data.py 验证质量。")
