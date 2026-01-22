"""
=============================================================================
回退稳健配置版 MDOF 仿真系统
Robust MDOF Simulation (Reverted dt and Logic)

修复内容:
1. 恢复 dt = 0.01 (原始工作配置)
2. 恢复简单的激励索引逻辑 (excitation[i])
3. 保持纯 Python 和单线程运行
=============================================================================
"""

import numpy as np
from scipy import signal, interpolate
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import hashlib
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 移除 Numba，使用纯 Python
# ==============================================================================

def _derivatives_numba_mdo(y, M_inv, K, C, f_t):
    """计算导数"""
    n = len(y) // 2
    dydt = np.zeros_like(y)
    x = y[:n]
    v = y[n:]
    dydt[:n] = v
    force = f_t - C @ v - K @ x
    a = M_inv @ force
    dydt[n:] = a
    return dydt

def _rk4_step_mdo(y, dt, M_inv, K, C, f_current, f_next):
    """RK4 积分步"""
    k1 = _derivatives_numba_mdo(y, M_inv, K, C, f_current)
    y2 = y + k1 * dt * 0.5
    k2 = _derivatives_numba_mdo(y2, M_inv, K, C, f_current)
    y3 = y + k2 * dt * 0.5
    k3 = _derivatives_numba_mdo(y3, M_inv, K, C, f_next)
    y4 = y + k3 * dt
    k4 = _derivatives_numba_mdo(y4, M_inv, K, C, f_next)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _integrate_segment_mdo(y0, excitation, dt, M_inv, K, C, record_every):
    """
    积分循环 (恢复原始简单逻辑)
    """
    n_steps = len(excitation)
    n_states = len(y0)
    n_dof = n_states // 2
    
    # n_records 计算逻辑
    n_records = n_steps // record_every + 1
    recorded_response = np.zeros((n_records, n_dof))
    
    y = y0.copy()
    record_idx = 0
    recorded_response[record_idx] = y[:n_dof]
    record_idx += 1
    
    # 恢复原始循环：range(n_steps - 1)
    for i in range(n_steps - 1):
        if i % record_every == 0 and record_idx < n_records:
            recorded_response[record_idx] = y[:n_dof]
            record_idx += 1
        
        # 恢复原始逻辑：直接使用 excitation 索引，不计算 current_time
        # 注意：这要求 dt * n_steps = duration，且 fs * duration = n_steps
        # 这里 dt=0.01, fs=100, 所以每一步对应一个激励点
        f_curr = excitation[i]
        f_next = excitation[i + 1]
        
        y = _rk4_step_mdo(y, dt, M_inv, K, C, f_curr, f_next)
    
    if record_idx < n_records:
        recorded_response[-1] = y[:n_dof]
    
    return recorded_response, y

# ==============================================================================
# 2. 优化后的 MDOF 仿真器类 (简化初始化)
# ==============================================================================

class OptimizedMDOFSimulator:
    def __init__(self, n_dof=10, mass=100.0, k_base=5e6, 
                 damping_ratio=0.02, fs=100.0, dt=0.01, 
                 downsample_factor=10, rng_seed=None):
        self.n_dof = n_dof
        self.mass = mass
        self.k_base = k_base
        self.damping_ratio = damping_ratio
        self.fs = fs
        self.dt = dt
        
        # 恢复原始 record_every 逻辑
        self.record_every = int(downsample_factor)
        
        self.rng = np.random.RandomState(rng_seed)
        
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
            self.rng.seed(seed)
        n_points = int(duration * self.fs)
        white_noise = self.rng.randn(n_points)
        baseline_points_per_segment = max(2000, n_points // 20)
        num_segments = int(np.ceil(n_points / baseline_points_per_segment))
        segment_length = n_points // num_segments
        excitation = np.zeros((n_points, self.n_dof))
        amplitudes = np.zeros(num_segments)
        amplitudes[0] = self.rng.uniform(0.5, 1.0)
        for i in range(1, num_segments):
            delta = self.rng.uniform(-0.2, 0.2)
            amplitudes[i] = np.clip(amplitudes[i-1] + delta, 0.5, 1.0)
        nyquist = self.fs / 2
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, n_points)
            segment = white_noise[start_idx:end_idx] * amplitudes[i]
            f_low = self.rng.uniform(0.01 * self.fs, 0.05 * self.fs)
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
                segment_filtered = self.rng.uniform(-0.5, 0.5, size=segment_filtered.shape)
            excitation[start_idx:end_idx, -1] = segment_filtered
        return excitation
    
    def integrate_segment(self, y0, excitation, debug_id=None):
        if debug_id is not None:
            print(f"  [DEBUG {debug_id}] Integrate Input Check (Robust Mode):")
            print(f"    dt: {self.dt}, record_every: {self.record_every}")

        # 移除 fs 参数，使用简化逻辑
        response, y_final = _integrate_segment_mdo(
            y0, excitation, self.dt, 
            self.M_inv, self.K_current, self.C_healthy,
            self.record_every
        )
        
        if debug_id is not None:
            print(f"  [DEBUG {debug_id}] Response max: {np.max(np.abs(response)):.2e}, Has NaN: {np.any(np.isnan(response))}")

        return response, y_final

# ==============================================================================
# 3. 智能缓存管理器
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
# 4. GVR 特征计算与图像生成逻辑
# ==============================================================================

def calculate_gvr_spatial_profile(resp_healthy, resp_damage, debug_id=None):
    if np.any(np.isnan(resp_healthy)) or np.any(np.isnan(resp_damage)):
        if debug_id is not None:
            is_nan_healthy = np.any(np.isnan(resp_healthy))
            is_nan_damage = np.any(np.isnan(resp_damage))
            print(f"  [DEBUG {debug_id}] ERROR in GVR Input! Healthy_NaN: {is_nan_healthy}, Damage_NaN: {is_nan_damage}")
        return np.zeros(resp_healthy.shape[0])

    diff = resp_damage - resp_healthy
    diff_sum = np.sum(np.abs(diff), axis=1) 
    denom = np.sum(resp_healthy**2, axis=1) + 1e-6
    di_profile = diff_sum / denom
    
    di_diff_1 = np.diff(di_profile)
    gvr_profile = np.abs(np.diff(di_diff_1))
    
    if len(gvr_profile) > 0:
        x_old = np.arange(len(gvr_profile))
        x_new = np.arange(len(di_profile))
        interp_func = interpolate.interp1d(x_old, gvr_profile, kind='linear', fill_value="extrapolate")
        gvr_full = interp_func(x_new)
    else:
        gvr_full = np.zeros_like(di_profile)
        
    gvr_smooth = np.convolve(gvr_full, np.ones(3)/3, mode='same')
    
    if debug_id is not None:
        print(f"  [DEBUG {debug_id}] GVR Stat: gvr_smooth_max={np.max(gvr_smooth):.2e}")
    
    return gvr_smooth

def generate_gvr_feature_image(gvr_profile, target_size=224, debug_id=None):
    x_old = np.linspace(0, target_size, len(gvr_profile))
    x_new = np.linspace(0, target_size, target_size)
    try:
        interp_func = interpolate.interp1d(x_old, gvr_profile, kind='cubic', fill_value="extrapolate")
        gvr_interp = interp_func(x_new)
    except:
        interp_func = interpolate.interp1d(x_old, gvr_profile, kind='linear', fill_value="extrapolate")
        gvr_interp = interp_func(x_new)
    
    img_2d = np.tile(gvr_interp[np.newaxis, :], (target_size, 1))
    
    gvr_min, gvr_max = np.min(gvr_interp), np.max(gvr_interp)
    range_val = gvr_max - gvr_min
    
    if debug_id is not None:
        print(f"  [DEBUG {debug_id}] Image Pre-norm: min={gvr_min:.2e}, max={gvr_max:.2e}")
    
    if range_val > 1e-6:
        img_norm = (img_2d - gvr_min) / range_val
    else:
        img_norm = np.zeros_like(img_2d)
        if debug_id is not None:
            print(f"  [DEBUG {debug_id}] Image: Range too small, set to ZEROS")
    
    r_channel = img_norm 
    g_channel = 1.0 - 2.0 * np.abs(img_norm - 0.5)
    g_channel = np.clip(g_channel, 0, 1)
    b_channel = 1.0 - img_norm
    
    from scipy.ndimage import gaussian_filter
    sigma = target_size / 40.0
    r_channel = gaussian_filter(r_channel, sigma=sigma)
    g_channel = gaussian_filter(g_channel, sigma=sigma)
    b_channel = gaussian_filter(b_channel, sigma=sigma)
    
    img_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
    img_rgb = np.clip(img_rgb, 0, 1)
    return np.transpose(img_rgb, (2, 0, 1))

# ==============================================================================
# 5. 并行数据生成逻辑 (恢复原始参数)
# ==============================================================================

def generate_single_sample(sample_idx, seed, label, sim_params, 
                         noise_level=0.05, damage_subtlety=0.7, debug_mode=False):
    should_debug = debug_mode and (sample_idx == 0) and (label != 0) 
    debug_id = f"Idx{sample_idx}_L{label}" if should_debug else None

    if should_debug:
        print(f"\n{'='*60}")
        print(f"Start Sample: Idx={sample_idx}, Label={label}, Seed={seed}")
        print(f"{'='*60}")

    # 恢复原始 sim_params
    sim_healthy = OptimizedMDOFSimulator(**sim_params, rng_seed=seed)
    
    duration = 10.0
    excitation = sim_healthy.generate_excitation(duration, seed=seed)
    
    y0 = np.zeros(sim_healthy.n_dof * 2)
    resp_healthy, _ = sim_healthy.integrate_segment(y0, excitation, debug_id=debug_id)
    
    if np.any(np.isnan(resp_healthy)):
        if should_debug:
            print(f"  [DEBUG {debug_id}] FAILED! Healthy response is NaN.")
        return np.zeros((10, 101)), np.zeros((3, 224, 224)), label

    sim_damage = OptimizedMDOFSimulator(**sim_params, rng_seed=seed + 1000000)
    
    damage_scenarios = {
        0: [],           
        1: [sim_damage.n_dof//2],       
        2: [sim_damage.n_dof//3],       
        3: [sim_damage.n_dof//2, sim_damage.n_dof-1],  
    }
    damaged_dofs = damage_scenarios.get(label, [])
    
    if should_debug:
        print(f"  [DEBUG {debug_id}] Damage: DOFs={damaged_dofs}")

    if damaged_dofs:
        severity = np.random.uniform(0.2, 0.5)
        sim_damage.apply_damage(damaged_dofs, severity=severity)
        if should_debug:
            print(f"  [DEBUG {debug_id}] Severity: {severity:.4f}")
    
    resp_damage, _ = sim_damage.integrate_segment(y0, excitation, debug_id=debug_id)
    
    if np.any(np.isnan(resp_damage)):
         if should_debug:
            print(f"  [DEBUG {debug_id}] FAILED! Damage response is NaN.")
         return np.zeros((10, 101)), np.zeros((3, 224, 224)), label
    
    gvr_profile = calculate_gvr_spatial_profile(resp_healthy, resp_damage, debug_id=debug_id)
    img = generate_gvr_feature_image(gvr_profile, target_size=224, debug_id=debug_id)
    
    return resp_damage, img, label

def generate_mdof_dataset_parallel(n_samples=10, n_dof=10, 
                                   segment_time=10.0, n_jobs=1,
                                   force_regen=False, seed=42,
                                   noise_level=0.05,
                                   damage_subtlety=0.7,
                                   debug_mode=True):
    # 恢复原始参数
    sim_params = {
        'n_dof': n_dof,
        'mass': 100.0,
        'k_base': 5e6,
        'damping_ratio': 0.02,
        'fs': 100.0,
        'dt': 0.01,       # 恢复为 0.01
        'downsample_factor': 10
    }
    
    cache = MDOFDataCache()
    
    def generator_func(n_samples, n_dof, segment_time, noise_level, damage_subtlety):
        rng = np.random.RandomState(seed)
        tasks = []
        for i in range(n_samples):
            sample_seed = rng.randint(0, 2**31-1)
            label = rng.randint(0, 4)
            tasks.append((i, sample_seed, label, sim_params, noise_level, damage_subtlety))
        
        results = []
        for idx, s, l, p, nl, ds in tqdm(tasks, desc="单线程生成 (Robust)", leave=False):
            res = generate_single_sample(idx, s, l, p, nl, ds, debug_mode)
            results.append(res)
        
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
# 6. 主程序
# ==============================================================================

def visualize_results_gvr(signals, images, labels, save_path='mdof_gvr_robust_test.png'):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    healthy_idx = np.where(labels == 0)[0][0]
    damage_idx = np.where(labels == 1)[0][0]
    
    axes[0, 0].plot(signals[healthy_idx, 0, :])
    axes[0, 0].set_title('健康样本 - 传感器1')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(signals[damage_idx, 0, :])
    axes[0, 1].set_title('损伤样本 - 传感器1')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].imshow(np.transpose(images[healthy_idx], (1, 2, 0)))
    axes[1, 0].set_title('健康样本 - GVR')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.transpose(images[damage_idx], (1, 2, 0)))
    axes[1, 1].set_title('损伤样本 - GVR')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存: {save_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("回退稳健配置版 MDOF 仿真系统")
    print("=" * 70)
    
    n_samples = 10
    n_dof = 10
    
    start_time = time.time()
    signals, images, labels = generate_mdof_dataset_parallel(
        n_samples=n_samples,
        n_dof=n_dof,
        segment_time=10.0,
        n_jobs=1,  # 单线程
        force_regen=True,
        seed=42,
        noise_level=0.05,
        damage_subtlety=0.5,
        debug_mode=True
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
