"""
=============================================================================
带调试信息的 MDOF 仿真系统 (GVR 特征图版)
Debug-enabled MDOF Offshore Structure Simulation

新增特性:
- 在 GVR 计算各阶段输出数值统计
- 在图像生成阶段输出归一化状态
- 支持指定特定样本进行详细调试
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
# 1. Numba JIT 加速的核心函数 (未变动)
# ==============================================================================

@jit(nopython=True, cache=True)
def _derivatives_numba_mdo(y, M_inv, K, C, f_t):
    n = len(y) // 2
    dydt = np.zeros_like(y)
    x = y[:n]
    v = y[n:]
    dydt[:n] = v
    force = f_t - C @ v - K @ x
    a = M_inv @ force
    dydt[n:] = a
    return dydt

@jit(nopython=True, cache=True)
def _rk4_step_mdo(y, dt, M_inv, K, C, f_current, f_next):
    k1 = _derivatives_numba_mdo(y, M_inv, K, C, f_current)
    y2 = y + k1 * dt * 0.5
    k2 = _derivatives_numba_mdo(y2, M_inv, K, C, f_current)
    y3 = y + k2 * dt * 0.5
    k3 = _derivatives_numba_mdo(y3, M_inv, K, C, f_next)
    y4 = y + k3 * dt
    k4 = _derivatives_numba_mdo(y4, M_inv, K, C, f_next)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@jit(nopython=True, cache=True)
def _integrate_segment_mdo(y0, excitation, dt, M_inv, K, C, record_every):
    n_steps = len(excitation)
    n_states = len(y0)
    n_dof = n_states // 2
    n_records = n_steps // record_every + 1
    recorded_response = np.zeros((n_records, n_dof))
    y = y0.copy()
    record_idx = 0
    recorded_response[record_idx] = y[:n_dof]
    record_idx += 1
    for i in range(n_steps - 1):
        if i % record_every == 0 and record_idx < n_records:
            recorded_response[record_idx] = y[:n_dof]
            record_idx += 1
        f_curr = excitation[i]
        f_next = excitation[i + 1]
        y = _rk4_step_mdo(y, dt, M_inv, K, C, f_curr, f_next)
    if record_idx < n_records:
        recorded_response[-1] = y[:n_dof]
    return recorded_response, y

# ==============================================================================
# 2. 优化后的 MDOF 仿真器类 (未变动)
# ==============================================================================

class OptimizedMDOFSimulator:
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
# 3. 智能缓存管理器 (未变动)
# ==============================================================================

class MDOFDataCache:
    def __init__(self, cache_dir='./mdof_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, n_samples, n_dof, segment_time, seed, noise_level, damage_subtlety):
        params_str = f"{n_samples}_{n_dof}_{segment_time}_{seed}_{noise_level}_{damage_subtlety}"
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def load_or_generate(self, n_samples, n_dof=10, segment_time=10.0, 
                         force_regen=False, generator_func=None, seed=42, 
                         noise_level=0.05, damage_subtlety=0.7):
        cache_key = self._get_cache_key(n_samples, n_dof, segment_time, seed, noise_level, damage_subtlety)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
        
        if not force_regen and os.path.exists(cache_file):
            print(f"[Cache] ✓ 从缓存加载: {cache_file}")
            data = np.load(cache_file)
            return data['signals'], data['images'], data['labels']
        
        print(f"[Cache] ⚙️  生成新数据 (n_samples={n_samples}, n_dof={n_dof})...")
        if generator_func:
            signals, images, labels = generator_func(n_samples, n_dof, segment_time, noise_level, damage_subtlety)
        else:
            raise ValueError("必须提供生成函数")
        
        np.savez_compressed(
            cache_file, 
            signals=signals, 
            images=images, 
            labels=labels
        )
        print(f"[Cache] ✓ 缓存已保存: {cache_file}")
        
        return signals, images, labels

# ==============================================================================
# 4. 带调试信息的 GVR 特征计算与图像生成逻辑
# ==============================================================================

def calculate_gvr_spatial_profile(resp_healthy, resp_damage, debug_id=None):
    """
    计算沿传感器通道的空间 GVR (带调试输出)
    """
    # 1. 计算 Damage Index (DI)
    diff = resp_damage - resp_healthy
    
    # --- 修复后 ---
    diff_sum = np.sum(np.abs(diff), axis=1) 
    
    denom = np.sum(resp_healthy**2, axis=1) + 1e-6
    di_profile = diff_sum / denom
    
    # --- 调试信息 1: 基础差异 ---
    if debug_id is not None:
        print(f"  [DEBUG {debug_id}] DI Stat: diff_sum_max={np.max(diff_sum):.2e}, denom_min={np.min(denom):.2e}, di_max={np.max(di_profile):.2e}")
    
    # 2. 计算空间 GVR
    di_diff_1 = np.diff(di_profile)
    gvr_profile = np.abs(np.diff(di_diff_1))
    
    # 插值回原长度
    if len(gvr_profile) > 0:
        x_old = np.arange(len(gvr_profile))
        x_new = np.arange(len(di_profile))
        interp_func = interpolate.interp1d(x_old, gvr_profile, kind='linear', fill_value="extrapolate")
        gvr_full = interp_func(x_new)
    else:
        gvr_full = np.zeros_like(di_profile)
        
    gvr_smooth = np.convolve(gvr_full, np.ones(3)/3, mode='same')
    
    # --- 调试信息 2: GVR 最终值 ---
    if debug_id is not None:
        print(f"  [DEBUG {debug_id}] GVR Stat: gvr_smooth_max={np.max(gvr_smooth):.2e}, gvr_smooth_mean={np.mean(gvr_smooth):.2e}")
    
    return gvr_smooth

def generate_gvr_feature_image(gvr_profile, target_size=224, debug_id=None):
    """
    将 1D GVR 分布生成为 2D 特征分布图 (带调试输出)
    """
    # 1. 插值
    x_old = np.linspace(0, target_size, len(gvr_profile))
    x_new = np.linspace(0, target_size, target_size)
    interp_func = interpolate.interp1d(x_old, gvr_profile, kind='cubic', fill_value="extrapolate")
    gvr_interp = interp_func(x_new)
    
    # 2. 扩展为 2D
    img_2d = np.tile(gvr_interp[np.newaxis, :], (target_size, 1))
    
    # 3. 归一化
    gvr_min, gvr_max = np.min(gvr_interp), np.max(gvr_interp)
    range_val = gvr_max - gvr_min
    
    # --- 调试信息 3: 归一化前 ---
    if debug_id is not None:
        print(f"  [DEBUG {debug_id}] Image Pre-norm: min={gvr_min:.2e}, max={gvr_max:.2e}, range={range_val:.2e}")
    
    if range_val > 1e-6:
        img_norm = (img_2d - gvr_min) / range_val
        if debug_id is not None:
            print(f"  [DEBUG {debug_id}] Image Post-norm: min={np.min(img_norm):.4f}, max={np.max(img_norm):.4f} -> Normalized")
    else:
        img_norm = np.zeros_like(img_2d)
        if debug_id is not None:
            print(f"  [DEBUG {debug_id}] Image: Range too small (<1e-6), set to ZEROS (Will be Blue!)")
    
    # 4. 颜色映射
    r_channel = img_norm 
    g_channel = 1.0 - 2.0 * np.abs(img_norm - 0.5)
    g_channel = np.clip(g_channel, 0, 1)
    b_channel = 1.0 - img_norm
    
    from scipy.ndimage import gaussian_filter
    sigma = target_size / 40.0
    r_channel = gaussian_filter(r_channel, sigma=sigma)
    g_channel = gaussian_filter(g_channel, sigma=sigma)
    b_channel = gaussian_filter(b_channel, sigma=sigma)
    
    img_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1) # (H, W, C)
    img_rgb = np.clip(img_rgb, 0, 1)
    return np.transpose(img_rgb, (2, 0, 1))

# ==============================================================================
# 5. 并行数据生成逻辑 (增加调试开关)
# ==============================================================================

def generate_single_sample(sample_idx, seed, label, sim_params, noise_level=0.05, damage_subtlety=0.7, debug_mode=False):
    """
    生成单个样本 (增加 sample_idx 和 debug_mode)
    """
    # 决定是否打印调试信息 (例如: 只打印第一个损伤样本)
    should_debug = debug_mode and (sample_idx == 0) and (label != 0) 
    debug_id = f"Idx{sample_idx}_L{label}" if should_debug else None

    if should_debug:
        print(f"\n{'='*60}")
        print(f"开始生成样本: SampleIdx={sample_idx}, Label={label}, Seed={seed}")
        print(f"{'='*60}")
    
    # 1. 创建健康态仿真器
    sim_healthy = OptimizedMDOFSimulator(**sim_params)
    np.random.seed(seed)
    
    duration = 10.0
    excitation = sim_healthy.generate_excitation(duration, seed=seed)
    
    # 2. 积分健康态
    y0 = np.zeros(sim_healthy.n_dof * 2)
    resp_healthy, _ = sim_healthy.integrate_segment(y0, excitation)
    resp_healthy = resp_healthy.T 
    
    # 3. 应用损伤
    sim_damage = OptimizedMDOFSimulator(**sim_params)
    damage_scenarios = {
        0: [],           
        1: [sim_damage.n_dof//2],       
        2: [sim_damage.n_dof//3],       
        3: [sim_damage.n_dof//2, sim_damage.n_dof-1],  
    }
    damaged_dofs = damage_scenarios.get(label, [])
    
    if should_debug:
        print(f"  [DEBUG {debug_id}] Damage Config: DOFs={damaged_dofs}")
    
    if damaged_dofs:
        severity = np.random.uniform(0.2, 0.5)
        sim_damage.apply_damage(damaged_dofs, severity=severity)
        if should_debug:
            print(f"  [DEBUG {debug_id}] Applied Severity: {severity:.4f}")
    else:
        if should_debug:
            print(f"  [DEBUG {debug_id}] No Damage Applied (Healthy)")
    
    # 4. 积分损伤态
    resp_damage, _ = sim_damage.integrate_segment(y0, excitation)
    resp_damage = resp_damage.T 
    
    # 5. 计算 GVR (传入 debug_id)
    gvr_profile = calculate_gvr_spatial_profile(resp_healthy, resp_damage, debug_id=debug_id)
    
    # 6. 生成图像 (传入 debug_id)
    img = generate_gvr_feature_image(gvr_profile, target_size=224, debug_id=debug_id)
    
    return resp_damage, img, label

def generate_mdof_dataset_parallel(n_samples=100, n_dof=10, 
                                   segment_time=10.0, n_jobs=-1,
                                   force_regen=False, seed=42,
                                   noise_level=0.05,
                                   damage_subtlety=0.7,
                                   debug_mode=True):  # 默认开启调试
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
            # 将 sample_idx 传入
            tasks.append((i, sample_seed, label, sim_params, noise_level, damage_subtlety))
        
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(generate_single_sample)(idx, s, l, p, nl, ds, debug_mode)
            for idx, s, l, p, nl, ds in tqdm(tasks, desc="并行生成", leave=False)
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
# 6. 主程序与测试代码 (未变动)
# ==============================================================================

def visualize_results_gvr(signals, images, labels, save_path='mdof_gvr_debug_test.png'):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    healthy_idx = np.where(labels == 0)[0][0]
    damage_idx = np.where(labels == 1)[0][0]
    
    axes[0, 0].plot(signals[healthy_idx, 0, :])
    axes[0, 0].set_title('健康样本 - 传感器1位移')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('位移')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(signals[damage_idx, 0, :])
    axes[0, 1].set_title('损伤样本 - 传感器1位移')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('位移')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].imshow(np.transpose(images[healthy_idx], (1, 2, 0)))
    axes[1, 0].set_title('健康样本 - GVR 特征图')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.transpose(images[damage_idx], (1, 2, 0)))
    axes[1, 1].set_title('损伤样本 - GVR 特征图')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存: {save_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("带调试信息的 MDOF 仿真系统")
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
        damage_subtlety=0.5,
        debug_mode=True  # 开启调试模式
    )
    end_time = time.time()
    
    print(f"\n{'='*70}")
    print("数据生成完成！")
    print(f"{'='*70}")
    print(f"样本数: {n_samples}")
    print(f"信号形状: {signals.shape}")
    print(f"图像形状: {images.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"{'='*70}")
    
    visualize_results_gvr(signals, images, labels)
