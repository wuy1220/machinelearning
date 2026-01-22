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


class JacketPlatformSimulator:
    """ 海洋导管架平台多自由度仿真系统 """
    
    def __init__(self, 
                 num_degrees: int = 30,
                 dt: float = 0.001,
                 duration: float = 30.0,
                 damping_ratio: float = 0.05,
                 seed: Optional[int] = None):
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
        self.healthy_response = None
        
        self._initialize_system()
    
    def _initialize_system(self):
        self.num_layers = 5
        self.dofs_per_layer = self.num_degrees // self.num_layers
        mass_per_layer = np.linspace(1000, 5000, self.num_layers)
        self.mass_values = np.repeat(mass_per_layer, self.dofs_per_layer)
        self.M = np.diag(self.mass_values)
        self.K = self._build_stiffness_matrix()
        self.K0 = self.K.copy()
        self._compute_modal_properties()
        self._build_damping_matrix()
    
    def _build_stiffness_matrix(self) -> np.ndarray:
        n = self.num_degrees
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 1e7
            layer_idx = i // self.dofs_per_layer
            K[i, i] *= (1 + 0.2 * layer_idx)
            for j in range(n):
                if abs(i - j) <= 2:
                    coupling_strength = 0.1 * np.exp(-abs(i - j))
                    K[i, j] = K[i, i] * coupling_strength
        return K
    
    def _compute_modal_properties(self):
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(self.M) @ self.K)
        self.natural_frequencies = np.sqrt(eigenvalues)
        self.mode_shapes = eigenvectors
        print(f"[DEBUG SIM] 固有频率 (前5阶): {self.natural_frequencies[:5] / (2*np.pi)} Hz")
    
    def _build_damping_matrix(self):
        omega1 = self.natural_frequencies[0]
        omega2 = self.natural_frequencies[1]
        alpha = 2 * self.damping_ratio * omega1 * omega2 / (omega1 + omega2)
        beta = 2 * self.damping_ratio / (omega1 + omega2)
        self.C = alpha * self.M + beta * self.K
        self.rayleigh_params = {'alpha': alpha, 'beta': beta}
    
    def apply_damage(self, damaged_dofs: List[int], severity_ratios: List[float]) -> np.ndarray:
        K_damaged = self.K.copy()
        for dof, severity in zip(damaged_dofs, severity_ratios):
            K_damaged[dof, dof] *= (1 - severity)
            for i in range(self.num_degrees):
                if i != dof:
                    K_damaged[dof, i] *= (1 - severity * 0.5)
                    K_damaged[i, dof] *= (1 - severity * 0.5)
        return K_damaged
    
    def generate_excitation(self, 
                          excitation_type: str = 'random',
                          amplitude_range: Tuple[float, float] = (0, 2000),
                          frequencies: Optional[List[float]] = None) -> np.ndarray:
        n = self.num_degrees
        num_steps = self.num_steps
        
        if excitation_type == 'random':
            F = np.random.uniform(amplitude_range[0], amplitude_range[1], (num_steps, n))
            print(f"[DEBUG EXC] 生成随机激励, Max: {np.max(F)}")
            
        elif excitation_type == 'harmonic':
            if frequencies is None:
                frequencies = [self.natural_frequencies[0] * 0.8, self.natural_frequencies[1] * 0.9]
            F = np.zeros((num_steps, n))
            for freq in frequencies:
                omega = 2 * np.pi * freq
                phase = np.random.uniform(0, 2*np.pi, n)
                amplitude = np.random.uniform(amplitude_range[0]*0.5, amplitude_range[1])
                F += amplitude * np.sin(omega * self.time[:, None] + phase[None, :])
            print(f"[DEBUG EXC] 生成谐波激励, 频率: {frequencies}")
                
        elif excitation_type == 'impact':
            F = np.zeros((num_steps, n))
            impact_location = np.random.randint(0, n)
            impact_time = np.random.uniform(0.1, 0.3) * self.duration
            impact_idx = int(impact_time / self.dt)
            impact_duration = 50
            
            t_impact = np.linspace(0, np.pi, impact_duration)
            impact_force = np.sin(t_impact) * amplitude_range[1]
            
            end_idx = min(impact_idx + impact_duration, num_steps)
            F[impact_idx:end_idx, impact_location] = impact_force[:end_idx-impact_idx]
            
            print(f"[DEBUG EXC] 生成冲击激励. 位置: {impact_location}, 时间: {impact_time:.2f}s, 峰值: {np.max(impact_force):.2f}")
        else:
            raise ValueError(f"Unknown excitation type: {excitation_type}")
        
        return F
    
    def simulate_response(self, K: np.ndarray, F: np.ndarray, initial_conditions: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        gamma = 0.5
        beta = 0.25
        n = self.num_degrees
        num_steps = self.num_steps
        
        x = np.zeros((num_steps, n))
        v = np.zeros((num_steps, n))
        a = np.zeros((num_steps, n))
        
        if initial_conditions is not None:
            x[0] = initial_conditions[0]
            v[0] = initial_conditions[1]
        else:
            a[0] = np.linalg.solve(self.M, F[0] - self.C @ v[0] - K @ x[0])
        
        for i in range(num_steps - 1):
            x_tilde = x[i] + self.dt * v[i] + 0.5 * self.dt**2 * (1 - 2*beta) * a[i]
            v_tilde = v[i] + self.dt * (1 - gamma) * a[i]
            
            K_eff = K + gamma/(beta*self.dt) * self.C + 1/(beta*self.dt**2) * self.M
            F_eff = F[i+1] + self.M @ (1/(beta*self.dt**2) * x_tilde + 1/(beta*self.dt) * v[i]) + \
                    self.C @ (gamma/(beta*self.dt) * x_tilde + (gamma/beta - 1) * v[i])
            
            x[i+1] = np.linalg.solve(K_eff, F_eff)
            a[i+1] = 1/(beta*self.dt**2) * (x[i+1] - x_tilde) - 1/(beta*self.dt) * v[i]
            v[i+1] = v[i] + self.dt * ((1-gamma) * a[i] + gamma * a[i+1])
        
        print(f"[DEBUG SIM] 仿真完成. Max: {np.max(a):.2e}, Min: {np.min(a):.2e}")
        return a


class GVRFeatureExtractor:
    """ 梯度变化率（GVR）特征提取器 """
    
    def __init__(self, dt: float, window_length: int = 3000, step_size: int = 50, cutoff_freq: float = 50.0, filter_order: int = 4):
        self.dt = dt
        self.window_length = window_length
        self.step_size = step_size
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        nyquist = 0.5 / self.dt
        self.b, self.a = signal.butter(filter_order, cutoff_freq / nyquist, btype='low')
    
    def butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        filtered_data = signal.filtfilt(self.b, self.a, data, axis=0)
        return filtered_data
    
    def compute_damage_index(self, damaged_signal: np.ndarray, healthy_signal: np.ndarray) -> np.ndarray:
        num_channels = damaged_signal.shape[1]
        DI = np.zeros(num_channels)
        for ch in range(num_channels):
            numerator = np.sum(np.abs(damaged_signal[:, ch] - healthy_signal[:, ch]))
            denominator = np.sum(healthy_signal[:, ch] ** 2) + 1e-10
            DI[ch] = numerator / denominator
        return DI
    
    def compute_gvr(self, DI: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        DI_prime = np.zeros_like(DI)
        DI_prime[1:] = DI[1:] - DI[:-1]
        DI_double_prime = np.zeros_like(DI)
        DI_double_prime[1:] = np.abs(DI_prime[1:] - DI_prime[:-1])
        return DI_prime, DI_double_prime
    
    def extract_gvr_features(self, damaged_signal: np.ndarray, healthy_signal: np.ndarray, return_time_series: bool = False) -> Dict:
        filtered_damaged = self.butterworth_filter(damaged_signal)
        
        num_steps = damaged_signal.shape[0]
        num_windows = (num_steps - self.window_length) // self.step_size + 1
        
        DI_series = []
        # 减少 debug 输出以免刷屏
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_length
            
            window_damaged = filtered_damaged[start_idx:end_idx]
            
            if healthy_signal.shape[0] >= end_idx:
                window_healthy = healthy_signal[start_idx:end_idx]
            else:
                window_healthy = filtered_damaged[start_idx:end_idx]
            
            DI_window = self.compute_damage_index(window_damaged, window_healthy)
            DI_series.append(DI_window)

        DI_series = np.array(DI_series)
        print(f"[DEBUG GVR] DI Shape: {DI_series.shape}, Max: {np.max(DI_series):.2e}, Mean: {np.mean(DI_series):.2e}")
        
        DI_prime, DI_double_prime = self.compute_gvr(DI_series)
        print(f"[DEBUG GVR] GVR Double Prime Max: {np.max(DI_double_prime):.2e}")
        
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
    
    def generate_gvr_feature_map(self, gvr_features: Dict, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        DI_double_prime = gvr_features['GVR_double_prime']
        num_samples = DI_double_prime.shape[0]
        num_channels = DI_double_prime.shape[1]
        
        feature_maps = np.zeros((num_samples, image_size[0], image_size[1], 3))
        x_original = np.arange(num_channels)
        x_new = np.linspace(0, num_channels - 1, image_size[1])
        
        for i in range(num_samples):
            data_min = DI_double_prime[i].min()
            data_max = DI_double_prime[i].max()
            
            if data_max > data_min:
                normalized = (DI_double_prime[i] - data_min) / (data_max - data_min)
            else:
                normalized = DI_double_prime[i].copy()
                # 只有当数据全为0时才打印警告，避免刷屏
                if i == 0: 
                    print(f"[DEBUG MAP] Window 0 数据全为0 (Max<=Min)，图像将为全黑。这在冲击激励的前几个窗口是正常的。")
            
            try:
                z_row = np.interp(x_new, x_original, normalized, left=normalized[0], right=normalized[-1])
            except Exception as e:
                indices = np.clip(np.searchsorted(x_original, x_new), 0, len(x_original) - 1)
                z_row = normalized[indices]
            
            img_2d = np.tile(z_row, (image_size[0], 1))
            img_rgb = np.stack([img_2d, img_2d, img_2d], axis=2)
            feature_maps[i] = img_rgb
        
        return feature_maps


class DamageDataGenerator:
    """ 损伤数据生成器 """
    
    def __init__(self, simulator: JacketPlatformSimulator, gvr_extractor: GVRFeatureExtractor, output_dir: str = './damage_data'):
        self.simulator = simulator
        self.gvr_extractor = gvr_extractor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = []
    
    def generate_single_damage_scenario(self, damaged_dofs: List[int], severity_ratios: List[float], scenario_id: int, save_data: bool = True) -> Dict:
        print(f"\n=== 场景 {scenario_id} ===")
        F = self.simulator.generate_excitation(excitation_type='impact')
        healthy_response_current = self.simulator.simulate_response(self.simulator.K0, F)
        K_damaged = self.simulator.apply_damage(damaged_dofs, severity_ratios)
        damaged_response = self.simulator.simulate_response(K_damaged, F)
        
        gvr_features = self.gvr_extractor.extract_gvr_features(
            damaged_response, 
            healthy_response_current
        )
        
        feature_maps = self.gvr_extractor.generate_gvr_feature_map(gvr_features)
        
        labels = np.zeros(self.simulator.num_degrees)
        labels[damaged_dofs] = 1
        damage_class = self._create_damage_class_label(damaged_dofs, severity_ratios)
        
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
        
        if save_data:
            self._save_scenario_data(data, scenario_id)
        
        self.metadata.append({
            'scenario_id': scenario_id,
            'damaged_dofs': damaged_dofs,
            'severity_ratios': severity_ratios,
            'damage_class': damage_class,
            'num_samples': feature_maps.shape[0]
        })
        return data
    
    def _create_damage_class_label(self, damaged_dofs: List[int], severity_ratios: List[float]) -> int:
        if len(damaged_dofs) == 0: return 0
        elif len(damaged_dofs) == 1:
            severity = severity_ratios[0]
            if severity < 0.3: return 1
            elif severity < 0.6: return 2
            else: return 3
        else: return 4
    
    def _save_scenario_data(self, data: Dict, scenario_id: int):
        filename = os.path.join(self.output_dir, f'scenario_{scenario_id:04d}.h5')
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('acceleration', data=data['acceleration_data'])
            hf.create_dataset('feature_maps', data=data['feature_maps'])
            hf.create_dataset('labels', data=data['labels'])
            hf.create_dataset('damage_class', data=np.array([data['damage_class']]))
            hf.attrs['damaged_dofs'] = np.array(data['damaged_dofs'])
            hf.attrs['severity_ratios'] = np.array(data['severity_ratios'])
            hf.attrs['scenario_id'] = scenario_id
    
    def generate_comprehensive_dataset(self, num_scenarios: int = 100, min_damage_dofs: int = 1, max_damage_dofs: int = 3, min_severity: float = 0.2, max_severity: float = 0.8, healthy_ratio: float = 0.1) -> Dict:
        print(f"开始生成综合数据集...")
        num_healthy = int(num_scenarios * healthy_ratio)
        num_damaged = num_scenarios - num_healthy
        
        for i in range(num_healthy):
            self.generate_single_damage_scenario([], [], scenario_id=i, save_data=True)
        
        for i in range(num_damaged):
            scenario_id = num_healthy + i
            num_damage_dofs = np.random.randint(min_damage_dofs, max_damage_dofs + 1)
            damaged_dofs = np.random.choice(range(self.simulator.num_degrees), num_damage_dofs, replace=False).tolist()
            severity_ratios = np.random.uniform(min_severity, max_severity, num_damage_dofs).tolist()
            self.generate_single_damage_scenario(damaged_dofs, severity_ratios, scenario_id=scenario_id, save_data=True)
        
        stats = {'total_samples': len(self.metadata)}
        print(f"数据集生成完成！共 {len(self.metadata)} 个场景")
        return stats


# ================= 修正后的可视化函数 =================

def visualize_generated_data(data_generator: DamageDataGenerator, num_scenarios: int = 4):
    """
    修正版可视化：
    1. 自动选择包含冲击（能量最大）的窗口进行显示，避免显示静默的黑窗口。
    2. 修复 axes 索引错误。
    """
    # 确保至少有 num_scenarios 个场景
    actual_num = min(num_scenarios, len(data_generator.metadata))
    if actual_num == 0:
        print("没有数据可供可视化")
        return

    print(f"正在可视化前 {actual_num} 个场景...")
    
    fig, axes = plt.subplots(actual_num, 4, figsize=(20, 5*actual_num))
    
    # 修复 IndexError：当只有1行时，axes 可能是 (1, 4) 或 (4,)，强制 reshape 为 (actual_num, 4)
    if actual_num == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(actual_num):
        scenario_id = data_generator.metadata[i]['scenario_id']
        filename = os.path.join(data_generator.output_dir, f'scenario_{scenario_id:04d}.h5')
        
        with h5py.File(filename, 'r') as hf:
            feature_maps = hf['feature_maps'][:] # Shape: (num_windows, H, W, 3)
            labels = hf['labels'][:]
            damage_class = hf['damage_class'][0]
        
        # --- 核心修改：计算每个窗口的能量，选取能量最大的窗口 ---
        # 能量计算：对每个窗口的所有像素求绝对值之和
        energies = np.sum(np.abs(feature_maps), axis=(1, 2, 3))
        best_window_idx = np.argmax(energies)
        
        print(f"场景 {i}: 选中窗口 {best_window_idx}/{len(feature_maps)-1} 进行显示 (能量: {energies[best_window_idx]:.2f})")
        
        feature_map = feature_maps[best_window_idx]
        
        # 显示特征图
        axes[i, 0].imshow(feature_map)
        axes[i, 0].set_title(f'Scenario {scenario_id} (Window {best_window_idx})')
        axes[i, 0].axis('off')
        
        # 显示 RGB 通道
        for c in range(3):
            axes[i, c+1].imshow(feature_map[:, :, c], cmap='hot')
            axes[i, c+1].set_title(f'Channel {c}')
            axes[i, c+1].axis('off')
        
        damaged_dofs = np.where(labels > 0)[0]
        info_text = f'Damaged DOFs: {damaged_dofs}\nClass: {damage_class}'
        axes[i, 0].text(0.02, 0.98, info_text, transform=axes[i, 0].transAxes,
                       color='white', fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(data_generator.output_dir, 'data_visualization.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"可视化已保存至 {save_path}")


if __name__ == "__main__":
    dt = 0.001
    duration = 30.0
    
    print("初始化导管架平台仿真系统...")
    simulator = JacketPlatformSimulator(num_degrees=30, dt=dt, duration=duration, damping_ratio=0.05, seed=42)
    
    print("初始化GVR特征提取器...")
    gvr_extractor = GVRFeatureExtractor(dt=dt, window_length=3000, step_size=50, cutoff_freq=50.0, filter_order=4)
    
    print("创建数据生成器...")
    data_generator = DamageDataGenerator(simulator=simulator, gvr_extractor=gvr_extractor, output_dir='./jacket_damage_data')
    
    print("生成测试损伤场景...")
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
    
    # 可视化
    if len(data_generator.metadata) > 0:
        visualize_generated_data(data_generator, num_scenarios=1)
