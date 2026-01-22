"""
=============================================================================
修改后的 MDOF 海洋结构损伤仿真数据生成器 (GVR 特征图版)
Modified MDOF Offshore Structure Damage Simulation Data Generator (GVR Feature Map)

集成特性:
1. Numba JIT 加速 (10-50x speedup)
2. 流式数据生成 (内存优化)
3. 多线程并行计算
4. 智能缓存管理
5. 物理激励信号模拟 (带通滤波 + 随机游走)
6. 自动生成多模态数据 (振动信号 + GVR 特征分布图)

适用场景: 验证GVR自动标注与多模态损伤检测算法
=============================================================================
"""

import numpy as np
from numba import jit
from scipy import signal, interpolate
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import hashlib
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Numba JIT 加速的核心函数 (高性能数值计算)
# ==============================================================================

@jit(nopython=True, cache=True)
def _derivatives_numba_mdo(y, M_inv, K, C, f_t):
    """
    通用MDOF系统状态导数计算 (Numba加速)
    方程: M*a + C*v + K*x = F
    状态: y = [x, v]
    导数: dy/dt = [v, M^(-1)*(F - C*v - K*x)]
    """
    n = len(y) // 2
    dydt = np.zeros_like(y)
    
    # 提取位移和速度
    x = y[:n]
    v = y[n:]
    
    # 前半部分：dxdt = v
    dydt[:n] = v
    
    # 后半部分：dvdt = a = M^(-1) * (f - C*v - K*x)
    force = f_t - C @ v - K @ x
    a = M_inv @ force
    dydt[n:] = a
    
    return dydt

@jit(nopython=True, cache=True)
def _rk4_step_mdo(y, dt, M_inv, K, C, f_current, f_next):
    """
    单步RK4积分 (Numba加速)
    """
    # k1
    k1 = _derivatives_numba_mdo(y, M_inv, K, C, f_current)
    
    # k2
    y2 = y + k1 * dt * 0.5
    k2 = _derivatives_numba_mdo(y2, M_inv, K, C, f_current)
    
    # k3
    y3 = y + k2 * dt * 0.5
    k3 = _derivatives_numba_mdo(y3, M_inv, K, C, f_next)
    
    # k4
    y4 = y + k3 * dt
    k4 = _derivatives_numba_mdo(y4, M_inv, K, C, f_next)
    
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@jit(nopython=True, cache=True)
def _integrate_segment_mdo(y0, excitation, dt, M_inv, K, C, record_every):
    """
    积分一个段，并按间隔记录响应 (内存优化版)
    
    参数:
        y0: 初始状态向量 [x1...xn, v1...vn]
        excitation: 激励时程 (n_steps, n_dof)
        dt: 积分时间步长
        M_inv, K, C: 系统矩阵
        record_every: 记录间隔 (下采样因子)
    
    返回:
        recorded_response: 记录的位移响应 (n_records, n_dof)
        y_final: 最终状态向量
    """
    n_steps = len(excitation)
    n_states = len(y0)
    n_dof = n_states // 2
    
    # 计算需要记录的点数
    n_records = n_steps // record_every + 1
    recorded_response = np.zeros((n_records, n_dof))
    
    y = y0.copy()
    record_idx = 0
    
    # 记录初始状态
    recorded_response[record_idx] = y[:n_dof]
    record_idx += 1
    
    for i in range(n_steps - 1):
        if i % record_every == 0 and record_idx < n_records:
            recorded_response[record_idx] = y[:n_dof]
            record_idx += 1
        
        f_curr = excitation[i]
        f_next = excitation[i + 1]
        y = _rk4_step_mdo(y, dt, M_inv, K, C, f_curr, f_next)
    
    # 记录最后一个点
    if record_idx < n_records:
        recorded_response[-1] = y[:n_dof]
    
    return recorded_response, y

# ==============================================================================
# 2. 优化后的 MDOF 仿真器类
# ==============================================================================

class OptimizedMDOFSimulator:
    """
    优化后的 MDOF 仿真器
    专注于物理仿真，不再包含图像生成逻辑
    """
    
    def __init__(self, n_dof=10, mass=100.0, k_base=5e6, 
                 damping_ratio=0.02, fs=100.0, dt=0.01, 
                 downsample_factor=10):
        self.n_dof = n_dof
        self.mass = mass
        self.k_base = k_base
        self.damping_ratio = damping_ratio
        self.fs = fs
        self.dt = dt
        self.downsample_factor = int(downsample_factor)
        self.record_every = int(self.downsample_factor)
        
        self.M = np.eye(n_dof) * mass
        self.K_healthy = self._build_stiffness_matrix(k_base)
        self.C_healthy = self._build_damping_matrix(self.M, self.K_healthy, damping_ratio)
        self.M_inv = np.linalg.inv(self.M)
        self.K_current = self.K_healthy.copy()
        
    def _build_stiffness_matrix(self, k_val):
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_dof):
            if i > 0: K[i, i-1] -= k_val
            K[i, i] += k_val
            if i < self.n_dof - 1: K[i, i+1] -= k_val
        K[0, 0] = k_val
        K[self.n_dof-1, self.n_dof-1] = k_val
        return K
    
    def _build_damping_matrix(self, M, K, zeta):
        omega_approx = np.sqrt(np.mean(np.diag(K) / np.diag(M)))
        alpha = 2 * zeta * omega_approx * 0.01
        beta = 2 * zeta / omega_approx * 0.99
        return alpha * M + beta * K
    
    def apply_damage(self, damaged_dofs, severity=0.3):
        self.K_current = self.K_healthy.copy()
        for dof in damaged_dofs:
            self.K_current[dof, dof] *= (1 - severity)
            if dof < self.n_dof - 1:
                self.K_current[dof, dof+1] *= (1 - severity)
                self.K_current[dof+1, dof] *= (1 - severity)
            if dof > 0:
                self.K_current[dof, dof-1] *= (1 - severity)
                self.K_current[dof-1, dof] *= (1 - severity)
    
    def generate_excitation(self, duration, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_points = int(duration * self.fs)
        white_noise = np.random.randn(n_points)
        baseline_points_per_segment = max(2000, n_points // 20)
        num_segments = int(np.ceil(n_points / baseline_points_per_segment))
        segment_length = n_points // num_segments
        excitation = np.zeros((n_points, self.n_dof))
        amplitudes = np.zeros(num_segments)
        amplitudes[0] = np.random.uniform(0.5, 1.0)
        for i in range(1, num_segments):
            delta = np.random.uniform(-0.2, 0.2)
            amplitudes[i] = np.clip(amplitudes[i-1] + delta, 0.5, 1.0)
        nyquist = self.fs / 2
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, n_points)
            segment = white_noise[start_idx:end_idx] * amplitudes[i]
            f_low = np.random.uniform(0.01 * self.fs, 0.05 * self.fs)
            f_high = f_low + 0.05 * self.fs
            low = max(0.0001, min(0.9999, f_low / nyquist))
            high = max(low + 0.001, min(0.9999, f_high / nyquist))
            try:
                sos = signal.butter(4, [low, high], btype='band', output='sos')
                segment_filtered = signal.sosfiltfilt(sos, segment)
            except:
                segment_filtered = segment
            max_val = np.max(np.abs(segment_filtered))
            if max_val > 1e-6:
                segment_filtered = segment_filtered / max_val * 0.5
            else:
                segment_filtered = np.random.uniform(-0.5, 0.5, size=segment_filtered.shape)
            excitation[start_idx:end_idx, -1] = segment_filtered
        return excitation
    
    def integrate_segment(self, y0, excitation):
        return _integrate_segment_mdo(
            y0, excitation, self.dt, 
            self.M_inv, self.K_current, self.C_healthy,
            self.record_every
        )

# ==============================================================================
# 3. 智能缓存管理器
# ==============================================================================

class MDOFDataCache:
    """智能缓存管理器"""
    
    def __init__(self, cache_dir='./mdof_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, n_samples, n_dof, segment_time, seed, noise_level, damage_subtlety):
        """生成唯一的缓存键 (已移除 add_texture)"""
        params_str = f"{n_samples}_{n_dof}_{segment_time}_{seed}_{noise_level}_{damage_subtlety}"
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def load_or_generate(self, n_samples, n_dof=10, segment_time=10.0, 
                         force_regen=False, generator_func=None, seed=42, 
                         noise_level=0.05, damage_subtlety=0.7):
        """加载缓存或生成新数据 (已移除 add_texture)"""
        cache_key = self._get_cache_key(n_samples, n_dof, segment_time, seed, noise_level, damage_subtlety)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
        
        # 尝试加载缓存
        if not force_regen and os.path.exists(cache_file):
            print(f"[Cache] ✓ 从缓存加载: {cache_file}")
            data = np.load(cache_file)
            return data['signals'], data['images'], data['labels']
        
        # 生成新数据
        print(f"[Cache] ⚙️  生成新数据 (n_samples={n_samples}, n_dof={n_dof})...")
        if generator_func:
            signals, images, labels = generator_func(n_samples, n_dof, segment_time, noise_level, damage_subtlety)
        else:
            raise ValueError("必须提供生成函数")
        
        # 保存缓存
        np.savez_compressed(
            cache_file, 
            signals=signals, 
            images=images, 
            labels=labels
        )
        print(f"[Cache] ✓ 缓存已保存: {cache_file}")
        
        return signals, images, labels

# ==============================================================================
# 4. GVR 特征计算与图像生成逻辑
# ==============================================================================

def calculate_gvr_spatial_profile(resp_healthy, resp_damage):
    """
    计算沿传感器通道的空间 GVR (Gradient Variation Rate)
    
    逻辑参考论文 Eq. (8), (9), (10):
    1. 计算 DI (Damage Index): 比较健康与损伤信号的差异
    2. 计算 DI' (一阶空间差分)
    3. 计算 GVR = abs(DI'' (二阶空间差分))
    
    参数:
        resp_healthy: (n_dof, n_time) 健康响应
        resp_damage: (n_dof, n_time) 损伤响应
    
    返回:
        gvr_profile: (n_dof,) 沿结构长度归一化的 GVR 分布
    """
    # 1. 计算 Damage Index (DI)
    # 论文公式: DI_j = sum(xd - xh) / sum(xh^2 + eps)
    # 这里我们计算时程的累积量
    diff = resp_damage - resp_healthy
    
    # 对时间轴求和，得到每个 DOF 的总差异
    diff_sum = diff_sum = np.sum(np.abs(diff), axis=1)  
    # 分母：健康信号的平方和 (防止除零)
    denom = np.sum(resp_healthy**2, axis=1) + 1e-6
    
    di_profile = diff_sum / denom
    
    # 2. 计算空间 GVR
    # 一阶差分 (相邻传感器之间的变化率)
    di_diff_1 = np.diff(di_profile)
    # 二阶差分绝对值 (GVR)
    gvr_profile = np.abs(np.diff(di_diff_1))
    
    # 由于差分两次，长度变短了 (len - 2)，需要插值回 n_dof
    # 使用线性插值补齐
    if len(gvr_profile) > 0:
        x_old = np.arange(len(gvr_profile))
        x_new = np.arange(len(di_profile))
        interp_func = interpolate.interp1d(x_old, gvr_profile, kind='linear', fill_value="extrapolate")
        gvr_full = interp_func(x_new)
    else:
        gvr_full = np.zeros_like(di_profile)
        
    # 对 GVR 进行平滑处理 (模拟滑动窗口统计效果)
    gvr_smooth = np.convolve(gvr_full, np.ones(3)/3, mode='same')
    
    return gvr_smooth

def generate_gvr_feature_image(gvr_profile, target_size=224):
    """
    将 1D GVR 分布生成为 2D 特征分布图 (热图)
    
    参数:
        gvr_profile: (n_dof,) 归一化后的 GVR 向量
        target_size: 图像尺寸 (默认 224)
        
    返回:
        img_array: (3, target_size, target_size) RGB 图像数组
    """
    # 1. 插值: 将 n_dof 个点扩展到 target_size 个点
    x_old = np.linspace(0, target_size, len(gvr_profile))
    x_new = np.linspace(0, target_size, target_size)
    interp_func = interpolate.interp1d(x_old, gvr_profile, kind='cubic', fill_value="extrapolate")
    gvr_interp = interp_func(x_new)
    
    # 2. 扩展为 2D: 沿 Y 轴镜像复制，形成对称分布
    img_2d = np.tile(gvr_interp[np.newaxis, :], (target_size, 1))
    
    # 3. 归一化到 [0, 1] 范围 (用于颜色映射)
    gvr_min, gvr_max = np.min(gvr_interp), np.max(gvr_interp)
    if gvr_max - gvr_min > 1e-6:
        img_norm = (img_2d - gvr_min) / (gvr_max - gvr_min)
    else:
        img_norm = np.zeros_like(img_2d)
    
    # 4. 应用热力图颜色映射 (Blue -> Green -> Red)
    # R 通道: 随值增加而增加
    r_channel = img_norm 
    # G 通道: 在中间值(0.5)达到最大 (突出峰值)
    g_channel = 1.0 - 2.0 * np.abs(img_norm - 0.5)
    g_channel = np.clip(g_channel, 0, 1)
    # B 通道: 随值增加而减少
    b_channel = 1.0 - img_norm
    
    # 增加一点高斯模糊效果，使特征图更平滑 (可选)
    from scipy.ndimage import gaussian_filter
    sigma = target_size / 40.0
    r_channel = gaussian_filter(r_channel, sigma=sigma)
    g_channel = gaussian_filter(g_channel, sigma=sigma)
    b_channel = gaussian_filter(b_channel, sigma=sigma)
    
    # 组合为 RGB
    img_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1) # (H, W, C)
    
    # 转换为 PyTorch/CV 常用格式 (C, H, W)
    img_rgb = np.clip(img_rgb, 0, 1)
    return np.transpose(img_rgb, (2, 0, 1))

# ==============================================================================
# 5. 并行数据生成逻辑
# ==============================================================================

def generate_single_sample(seed, label, sim_params, noise_level=0.05, damage_subtlety=0.7):
    """
    生成单个样本 (使用 GVR 特征图)
    """
    # 1. 创建仿真器实例 (健康态)
    sim_healthy = OptimizedMDOFSimulator(**sim_params)
    np.random.seed(seed)
    
    # 2. 生成激励 (复用)
    duration = 10.0
    excitation = sim_healthy.generate_excitation(duration, seed=seed)
    
    # 3. 积分健康态响应
    y0 = np.zeros(sim_healthy.n_dof * 2)
    resp_healthy, _ = sim_healthy.integrate_segment(y0, excitation)
    resp_healthy = resp_healthy.T # (n_dof, n_time)
    
    # 4. 应用损伤并积分损伤态响应
    # 创建一个新的仿真器用于损伤模拟
    sim_damage = OptimizedMDOFSimulator(**sim_params)
    
    # 映射标签到损伤位置
    damage_scenarios = {
        0: [],           
        1: [sim_damage.n_dof//2],       
        2: [sim_damage.n_dof//3],       
        3: [sim_damage.n_dof//2, sim_damage.n_dof-1],  
    }
    damaged_dofs = damage_scenarios.get(label, [])
    if damaged_dofs:
        severity = np.random.uniform(0.2, 0.5)
        sim_damage.apply_damage(damaged_dofs, severity=severity)
    
    resp_damage, _ = sim_damage.integrate_segment(y0, excitation)
    resp_damage = resp_damage.T # (n_dof, n_time)
    
    # 5. 计算 GVR 特征
    gvr_profile = calculate_gvr_spatial_profile(resp_healthy, resp_damage)
    
    # 6. 生成 GVR 特征分布图
    img = generate_gvr_feature_image(gvr_profile, target_size=224)
    
    return resp_damage, img, label

def generate_mdof_dataset_parallel(n_samples=100, n_dof=10, 
                                   segment_time=10.0, n_jobs=-1,
                                   force_regen=False, seed=42,
                                   noise_level=0.05,
                                   damage_subtlety=0.7):
    """
    并行生成 MDOF 数据集 (GVR 版)
    返回:
        signals: (n_samples, n_dof, n_timepoints)
        images: (n_samples, 3, 224, 224)
        labels: (n_samples,)
    """
    sim_params = {
        'n_dof': n_dof,
        'mass': 100.0,
        'k_base': 5e6,
        'damping_ratio': 0.02,
        'fs': 100.0,
        'dt': 0.01,
        'downsample_factor': 10
    }
    
    cache = MDOFDataCache()
    
    def generator_func(n_samples, n_dof, segment_time, noise_level, damage_subtlety):
        np.random.seed(seed)
        tasks = []
        for i in range(n_samples):
            sample_seed = np.random.randint(0, 2**31-1)
            label = np.random.randint(0, 4)
            tasks.append((sample_seed, label, sim_params, noise_level, damage_subtlety))
        
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(generate_single_sample)(s, l, p, nl, ds)
            for s, l, p, nl, ds in tqdm(tasks, desc="并行生成 (GVR)", leave=False)
        )
        
        signals = np.array([r[0] for r in results])
        images = np.array([r[1] for r in results])
        labels = np.array([r[2] for r in results])
        
        return signals, images, labels
    
    signals, images, labels = cache.load_or_generate(
        n_samples, n_dof, segment_time, force_regen, 
        generator_func, seed, noise_level, damage_subtlety
    )
    
    return signals, images, labels

# ==============================================================================
# 6. 主程序与测试代码
# ==============================================================================

def visualize_results_gvr(signals, images, labels, save_path='mdof_gvr_simulation_test.png'):
    """可视化 GVR 特征图结果"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    healthy_idx = np.where(labels == 0)[0][0]
    damage_idx = np.where(labels == 1)[0][0]
    
    # 健康样本信号
    axes[0, 0].plot(signals[healthy_idx, 0, :])
    axes[0, 0].set_title('健康样本 - 传感器1加速度')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('加速度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 损伤样本信号
    axes[0, 1].plot(signals[damage_idx, 0, :])
    axes[0, 1].set_title('损伤样本 - 传感器1加速度')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('加速度')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 健康样本 GVR 特征图 (应为深色/蓝色，接近0)
    axes[1, 0].imshow(np.transpose(images[healthy_idx], (1, 2, 0)))
    axes[1, 0].set_title('健康样本 - GVR 特征图 (平坦)')
    axes[1, 0].axis('off')
    
    # 损伤样本 GVR 特征图 (应有明显峰值/热点)
    axes[1, 1].imshow(np.transpose(images[damage_idx], (1, 2, 0)))
    axes[1, 1].set_title('损伤样本 - GVR 特征图 (有峰值)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ GVR 可视化结果已保存: {save_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("修改后的 MDOF 仿真系统 (GVR 特征图版)")
    print("=" * 70)
    
    n_samples = 100
    n_dof = 10
    
    start_time = time.time()
    signals, images, labels = generate_mdof_dataset_parallel(
        n_samples=n_samples,
        n_dof=n_dof,
        segment_time=10.0,
        n_jobs=-1,
        force_regen=True,
        seed=42,
        noise_level=0.05,
        damage_subtlety=0.5
    )
    end_time = time.time()
    
    print(f"\n{'='*70}")
    print("数据生成完成！")
    print(f"{'='*70}")
    print(f"样本数: {n_samples}")
    print(f"信号形状: {signals.shape}")
    print(f"图像形状: {images.shape} (GVR Feature Maps)")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"{'='*70}")
    
    visualize_results_gvr(signals, images, labels)
