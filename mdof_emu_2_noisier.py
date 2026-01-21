"""
=============================================================================
优化后的MDOF海洋结构损伤仿真数据生成器
Optimized MDOF Offshore Structure Damage Simulation Data Generator

集成特性:
1. Numba JIT 加速 (10-50x speedup)
2. 流式数据生成 (内存优化)
3. 多线程并行计算
4. 智能缓存管理
5. 物理激励信号模拟 (带通滤波 + 随机游走)
6. 自动生成多模态数据 (振动信号 + 结构损伤图像)

适用场景: 验证GVR自动标注与多模态损伤检测算法
=============================================================================
"""

import numpy as np
from numba import jit
from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import pickle
import hashlib
import time
import warnings
from skimage.draw import line
from PIL import Image, ImageDraw

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
# 2. 优化后的MDOF仿真器类
# ==============================================================================

class OptimizedMDOFSimulator:
    """
    优化后的MDOF仿真器
    结合了data_provider4.py中的多项优化技术
    """
    
    def __init__(self, n_dof=10, mass=100.0, k_base=5e6, 
                 damping_ratio=0.02, fs=100.0, dt=0.01, 
                 downsample_factor=10):
        """
        初始化MDOF系统
        
        参数:
            n_dof: 自由度数量 (传感器数量)
            mass: 每个节点的质量
            k_base: 层间刚度基准值 (N/m)
            damping_ratio: 阻尼比
            fs: 采样频率
            dt: 积分时间步长 (需满足稳定性条件)
            downsample_factor: 下采样因子 (积分后降采样)
        """
        self.n_dof = n_dof
        self.mass = mass
        self.k_base = k_base
        self.damping_ratio = damping_ratio
        self.fs = fs
        self.dt = dt
        self.downsample_factor = int(downsample_factor)
        self.record_every = int(self.downsample_factor)
        
        # 构建健康状态的系统矩阵
        self.M = np.eye(n_dof) * mass
        self.K_healthy = self._build_stiffness_matrix(k_base)
        self.C_healthy = self._build_damping_matrix(self.M, self.K_healthy, damping_ratio)
        
        # 预计算M的逆（用于Numba加速，避免每次求解）
        self.M_inv = np.linalg.inv(self.M)
        
        # 当前系统状态
        self.K_current = self.K_healthy.copy()
        
    def _build_stiffness_matrix(self, k_val):
        """
        构建三对角刚度矩阵 (模拟剪切型框架)
        K[i,i] = 2*k, K[i,i+1] = K[i+1,i] = -k
        """
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_dof):
            if i > 0: K[i, i-1] -= k_val
            K[i, i] += k_val
            if i < self.n_dof - 1: K[i, i+1] -= k_val
        # 修正边界
        K[0, 0] = k_val
        K[self.n_dof-1, self.n_dof-1] = k_val
        return K
    
    def _build_damping_matrix(self, M, K, zeta):
        """
        构建Rayleigh阻尼矩阵 C = alpha*M + beta*K
        """
        # 简化假设：使用近似固有频率
        omega_approx = np.sqrt(np.mean(np.diag(K) / np.diag(M)))
        alpha = 2 * zeta * omega_approx * 0.01
        beta = 2 * zeta / omega_approx * 0.99
        return alpha * M + beta * K
    
    def apply_damage(self, damaged_dofs, severity=0.3):
        """
        施加损伤：降低指定自由度的刚度
        
        参数:
            damaged_dofs: 发生损伤的自由度列表 (索引从0开始)
            severity: 刚度降低程度 (0.0 - 1.0)
        """
        self.K_current = self.K_healthy.copy()
        for dof in damaged_dofs:
            # 降低主对角线
            self.K_current[dof, dof] *= (1 - severity)
            
            # 降低关联的非对角线元素 (耦合项)
            if dof < self.n_dof - 1:
                self.K_current[dof, dof+1] *= (1 - severity)
                self.K_current[dof+1, dof] *= (1 - severity)
            if dof > 0:
                self.K_current[dof, dof-1] *= (1 - severity)
                self.K_current[dof-1, dof] *= (1 - severity)
    
    def generate_excitation(self, duration, seed=None):
        """
        生成激励信号（借鉴data_provider4.py的优化逻辑）
        
        特性:
            1. 基于白噪声
            2. 分段应用幅值调制 (随机游走)
            3. 带通滤波 (模拟海洋波浪的主频范围)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_points = int(duration * self.fs)
        white_noise = np.random.randn(n_points)
        
        # 分段应用幅值调制
        baseline_points_per_segment = max(2000, n_points // 20)
        num_segments = int(np.ceil(n_points / baseline_points_per_segment))
        segment_length = n_points // num_segments
        
        excitation = np.zeros((n_points, self.n_dof))
        
        amplitudes = np.zeros(num_segments)
        amplitudes[0] = np.random.uniform(0.5, 1.0)
        for i in range(1, num_segments):
            # 随机游走幅值
            delta = np.random.uniform(-0.2, 0.2)
            amplitudes[i] = np.clip(amplitudes[i-1] + delta, 0.5, 1.0)
        
        # 对每段进行滤波（只生成一列，然后扩展到所有DOF）
        nyquist = self.fs / 2
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, n_points)
            segment = white_noise[start_idx:end_idx] * amplitudes[i]
            
            # 带通滤波 (模拟海洋波浪频率 0.01fs ~ 0.1fs)
            f_low = np.random.uniform(0.01 * self.fs, 0.05 * self.fs)
            f_high = f_low + 0.05 * self.fs
            low = max(0.0001, min(0.9999, f_low / nyquist))
            high = max(low + 0.001, min(0.9999, f_high / nyquist))
            
            try:
                sos = signal.butter(4, [low, high], btype='band', output='sos')
                segment_filtered = signal.sosfiltfilt(sos, segment)
            except:
                segment_filtered = segment
            
            # 归一化
            max_val = np.max(np.abs(segment_filtered))
            if max_val > 1e-6:
                segment_filtered = segment_filtered / max_val * 0.5
            else:
                segment_filtered = np.random.uniform(-0.5, 0.5, size=segment_filtered.shape)
            
            # 扩展到所有DOF（假设激励主要作用在顶层）
            excitation[start_idx:end_idx, -1] = segment_filtered
        
        return excitation
    
    def integrate_segment(self, y0, excitation):
        """
        积分一个段（使用Numba加速的版本）
        """
        return _integrate_segment_mdo(
            y0, excitation, self.dt, 
            self.M_inv, self.K_current, self.C_healthy,
            self.record_every
        )
    
    def generate_feature_image(self, damaged_dofs=None, 
                            noise_level=0.05,
                            damage_subtlety=0.7,
                            add_texture=False,
                            random_seed=None):
        """
        增强版图像生成器：随机损伤形态 + 几何变换 + 干扰图案
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # ==============================================================================
        # 1. 基础画布设置 (使用更大的画布以容纳旋转和缩放)
        # ==============================================================================
        target_size = 224
        canvas_size = 300  # 留出余量用于旋转裁剪
        bg_base = np.random.randint(215, 235)
        bg_color = (bg_base, bg_base, bg_base + 5)
        
        img = Image.new('RGB', (canvas_size, canvas_size), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 计算布局参数 (相对于 300x300 画布)
        margin = int(canvas_size * 0.15)
        draw_width = canvas_size - 2 * margin
        step = draw_width // (self.n_dof + 1)
        
        points = []
        for i in range(self.n_dof):
            x = margin + (i + 1) * step
            y = canvas_size // 2
            points.append((x, y))
            # 绘制节点
            draw.ellipse([x-4, y-4, x+4, y+4], fill=(30, 30, 30))
        
        # 绘制连接线
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=(60, 60, 60), width=3)

        # ==============================================================================
        # 2. 随机化损伤表现形式 (随机选择: 晕染/裂纹/腐蚀)
        # ==============================================================================
        if damaged_dofs:
            for dof in damaged_dofs:
                if dof < len(points):
                    cx, cy = points[dof]
                    # 随机选择损伤类型: 0=晕染(原版), 1=裂纹, 2=腐蚀
                    damage_type = np.random.randint(0, 3) 
                    
                    # 基础可见性因子
                    visibility_factor = max(0.05, 1.0 - damage_subtlety)
                    
                    if damage_type == 0: 
                        # --- 类型0: 晕染 (多层渐变圆圈) ---
                        max_radius = int(30 * visibility_factor)
                        if max_radius > 3:
                            for r in range(max_radius, 3, -3):
                                intensity = int(25 * visibility_factor * (1 - r/max_radius))
                                color = (min(255, bg_base + intensity), bg_base, bg_base)
                                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=1)
                    
                    elif damage_type == 1:
                        # --- 类型1: 裂纹 (向外延伸的细线) ---
                        num_cracks = np.random.randint(2, 5)
                        crack_color = (bg_base + int(30*visibility_factor), bg_base, bg_base)
                        for _ in range(num_cracks):
                            angle = np.random.uniform(0, 2*np.pi)
                            length = np.random.uniform(10, 25) * visibility_factor
                            end_x = cx + length * np.cos(angle)
                            end_y = cy + length * np.sin(angle)
                            draw.line([cx, cy, end_x, end_y], fill=crack_color, width=1)
                    
                    elif damage_type == 2:
                        # --- 类型2: 腐蚀斑点 (节点周围的随机色块) ---
                        spot_radius = np.random.uniform(6, 12) * visibility_factor
                        offset_x = np.random.uniform(-10, 10)
                        offset_y = np.random.uniform(-10, 10)
                        # 较暗的腐蚀色
                        spot_color = (max(0, bg_base - int(40*visibility_factor)), 
                                       max(0, bg_base - int(20*visibility_factor)), 
                                       max(0, bg_base - int(20*visibility_factor)))
                        draw.ellipse([cx+offset_x-spot_radius, cy+offset_y-spot_radius, 
                                      cx+offset_x+spot_radius, cy+offset_y+spot_radius], 
                                     fill=spot_color)

                    # 节点本身的细微变色 (所有类型共有)
                    node_shift = int(15 * visibility_factor)
                    draw.ellipse([cx-4, cy-4, cx+4, cy+4], 
                            fill=(30 + node_shift, 30 - node_shift//2, 30 - node_shift//2))

        # ==============================================================================
        # 3. 几何变换 (旋转 + 缩放 + 中心裁剪)
        # ==============================================================================
        # 旋转: -15度 到 15度
        angle = np.random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg_color)
        
        # 缩放: 0.8 到 1.2 倍
        scale = np.random.uniform(0.8, 1.2)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        
        # 中心裁剪回 target_size (224)
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))

        # 转换为数组
        img_array = np.array(img).astype(np.float32) / 255.0

        # ==============================================================================
        # 4. 添加随机干扰图案 (纹理/网格/划痕)
        # ==============================================================================
        if add_texture:
            #pattern_type = np.random.randint(0, 3)
            pattern_type = 2
            
            if pattern_type == 0:
                # --- 干扰A: 浅色网格 ---
                # 添加非常淡的水平和垂直线条
                grid_intensity = np.random.uniform(0.05, 0.1)
                img_array[:, ::20, :] += grid_intensity  # 横线
                img_array[::20, :, :] += grid_intensity  # 竖线
                
            elif pattern_type == 1:
                # --- 干扰B: 随机划痕 ---
                # 在全图范围内随机添加几条细线
                num_scratches = np.random.randint(3, 8)
                for _ in range(num_scratches):
                    y_start = np.random.randint(0, target_size)
                    x_start = np.random.randint(0, target_size)
                    length = np.random.randint(20, 80)
                    angle = np.random.uniform(0, 2*np.pi)
                    x_end = int(np.clip(x_start + length * np.cos(angle), 0, target_size))
                    y_end = int(np.clip(y_start + length * np.sin(angle), 0, target_size))
                    
                    # 使用简单的数组操作模拟划痕 (降低亮度)
                    # 这里简化为一条随机像素线
                    rr, cc = line(y_start, x_start, y_end, x_end)
                    # 简单检查边界
                    valid = (rr >= 0) & (rr < target_size) & (cc >= 0) & (cc < target_size)
                    img_array[rr[valid], cc[valid], :] *= np.random.uniform(0.2, 0.7)
                    
            elif pattern_type == 2:
                # --- 干扰C: 局部色块/污渍 ---
                # 随机位置添加模糊色块
                num_stains = np.random.randint(2, 5)
                for _ in range(num_stains):
                    cx = np.random.randint(0, target_size)
                    cy = np.random.randint(0, target_size)
                    r = np.random.randint(10, 30)
                    y_grid, x_grid = np.ogrid[:target_size, :target_size]
                    mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= r**2
                    stain_intensity = np.random.uniform(0.05, 0.2)
                    stain_color = np.random.choice([1, -1]) # 变亮或变暗
                    img_array[mask] += stain_intensity * stain_color

        # ==============================================================================
        # 5. 基础高斯噪声 (原有逻辑)
        # ==============================================================================
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
        
        # Clip最终结果
        img_array = np.clip(img_array, 0, 1)
        
        # 转换为 (C, H, W)
        return np.transpose(img_array, (2, 0, 1))

# 需要在文件顶部添加 skimage.draw 的导入，用于绘制划痕
# from skimage.draw import line

# ==============================================================================
# 3. 智能缓存管理器
# ==============================================================================

class MDOFDataCache:
    """智能缓存管理器"""
    
    def __init__(self, cache_dir='./mdof_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, n_samples, n_dof, segment_time, seed, noise_level, damage_subtlety, add_texture):
        """生成唯一的缓存键"""
        params_str = f"{n_samples}_{n_dof}_{segment_time}_{seed}_{noise_level}_{damage_subtlety}_{add_texture}"
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def load_or_generate(self, n_samples, n_dof=10, segment_time=10.0, 
                         force_regen=False, generator_func=None, seed=42, noise_level=0.05, damage_subtlety=0.7, add_texture=False):
        """加载缓存或生成新数据"""
        cache_key = self._get_cache_key(n_samples, n_dof, segment_time, seed, noise_level, damage_subtlety, add_texture)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
        
        # 尝试加载缓存
        if not force_regen and os.path.exists(cache_file):
            print(f"[Cache] ✓ 从缓存加载: {cache_file}")
            data = np.load(cache_file)
            return data['signals'], data['images'], data['labels']
        
        # 生成新数据
        print(f"[Cache] ⚙️  生成新数据 (n_samples={n_samples}, n_dof={n_dof})...")
        if generator_func:
            signals, images, labels = generator_func(n_samples, n_dof, segment_time, noise_level, damage_subtlety, add_texture)
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
# 4. 并行数据生成逻辑
# ==============================================================================

def generate_single_sample(seed, label, sim_params,
                            noise_level=0.05,
                            damage_subtlety=0.7,
                            add_texture=False):
    """
    生成单个样本（用于并行处理）
    """
    # 创建仿真器实例
    sim = OptimizedMDOFSimulator(**sim_params)
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 生成激励
    duration = 10.0  # 10秒
    excitation = sim.generate_excitation(duration, seed=seed)
    
    # 应用损伤 (根据标签映射到损伤位置)
    # 映射逻辑: 0=健康, 1=中间, 2=底部, 3=多处
    damage_scenarios = {
        0: [],           
        1: [sim.n_dof//2],       
        2: [sim.n_dof//3],       
        3: [sim.n_dof//2, sim.n_dof-1],  
    }
    
    damaged_dofs = damage_scenarios.get(label, [])
    if damaged_dofs:
        # 随机化损伤严重程度 (20% - 50%)
        severity = np.random.uniform(0.2, 0.5)
        sim.apply_damage(damaged_dofs, severity=severity)
    
    # 积分
    y0 = np.zeros(sim.n_dof * 2)
    response, _ = sim.integrate_segment(y0, excitation)
    
    # 转置为 (n_dof, n_timepoints) 以匹配模型输入习惯
    response = response.T
    
    # 生成图像
    img = sim.generate_feature_image(
        damaged_dofs=damaged_dofs,
        noise_level=noise_level,
        damage_subtlety=damage_subtlety,
        add_texture=add_texture,
        random_seed=seed
    )
    
    
    return response, img, label

def generate_mdof_dataset_parallel(n_samples=100, n_dof=10, 
                                   segment_time=10.0, n_jobs=-1,
                                   force_regen=False, seed=42,
                                   noise_level=0.05,
                                   damage_subtlety=0.7,
                                   add_texture=False):
    """
    并行生成MDOF数据集（整合所有优化）
        noise_level: 图像噪声水平 (0-1)
        damage_subtlety: 损伤隐蔽程度 (0-1，越大越隐蔽)
        add_texture: 是否添加纹理噪声
    返回:
        signals: (n_samples, n_dof, n_timepoints)
        images: (n_samples, 3, 224, 224)
        labels: (n_samples,)
    """
    # 仿真器参数
    # 仿真参数配置字典
    sim_params = {
        'n_dof': n_dof,           # 自由度数量
        'mass': 100.0,           # 质量，单位为kg
        'k_base': 5e6,           # 基础刚度系数
        'damping_ratio': 0.02,   # 阻尼比
        'fs': 100.0,             # 采样频率，单位为Hz
        'dt': 0.01,              # 时间步长，单位为秒
        'downsample_factor': 10  # 下采样因子
    }
    
    # 使用缓存
    cache = MDOFDataCache()
    
    def generator_func(n_samples, n_dof, segment_time, noise_level, damage_subtlety, add_texture):
        """内部生成函数"""
        # 创建任务
        np.random.seed(seed)
        tasks = []
        for i in range(n_samples):
            sample_seed = np.random.randint(0, 2**31-1)
            label = np.random.randint(0, 4)
            tasks.append((sample_seed, label, sim_params, 
                         noise_level, damage_subtlety, add_texture))
        
        
        # 并行执行 (使用 threading 后端以兼容 Numba)
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(generate_single_sample)(s, l, p, nl, ds, at)
            for s, l, p, nl, ds, at in tqdm(tasks, desc="并行生成", leave=False)
        )
        
        # 整理结果
        signals = np.array([r[0] for r in results])
        images = np.array([r[1] for r in results])
        labels = np.array([r[2] for r in results])
        
        return signals, images, labels
    
    # 加载或生成数据
    signals, images, labels = cache.load_or_generate(
        n_samples, n_dof, segment_time, force_regen, 
        generator_func, seed,
        noise_level, damage_subtlety, add_texture
    )
    
    return signals, images, labels

# ==============================================================================
# 5. 主程序与测试代码
# ==============================================================================

def visualize_results(signals, images, labels, save_path='mdof_simulation_test.png'):
    """简单的可视化函数"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 选择一个健康样本和一个损伤样本
    healthy_idx = np.where(labels == 0)[0][0]
    damage_idx = np.where(labels == 1)[0][0]
    
    # 健康样本信号 (取第一个传感器)
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
    
    # 健康样本图像
    axes[1, 0].imshow(np.transpose(images[healthy_idx], (1, 2, 0)))
    axes[1, 0].set_title('健康样本 - 结构图像')
    axes[1, 0].axis('off')
    
    # 损伤样本图像
    axes[1, 1].imshow(np.transpose(images[damage_idx], (1, 2, 0)))
    axes[1, 1].set_title('损伤样本 - 结构图像')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存: {save_path}")


def visualize_damage_subtlety_comparison():
    """
    可视化对比不同隐蔽程度的损伤图像
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    
    # 创建仿真器
    sim = OptimizedMDOFSimulator(n_dof=10)
    
    # 损伤位置
    damaged_dof = sim.n_dof // 2
    
    # 不同参数组合
    configs = [
        ("原始版本", 0.0, 0.0, False),
        ("低隐蔽 + 低噪声", 0.02, 0.3, False),
        ("中等隐蔽 + 中等噪声", 0.05, 0.7, False),
        ("高隐蔽 + 高噪声 + 纹理", 0.08, 0.9, True),
        ("极隐蔽 + 极高噪声", 0.12, 0.95, True),
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    
    for col, (name, noise, subtlety, texture) in enumerate(configs):
        # 生成损伤图像
        img = sim.generate_feature_image(
            damaged_dofs=[damaged_dof],
            noise_level=noise,
            damage_subtlety=subtlety,
            add_texture=texture,
            random_seed=42
        )
        # img 的形状是 (3, 224, 224)
        
        # 显示图像 (需要转置为 (224, 224, 3))
        axes[0, col].imshow(np.transpose(img, (1, 2, 0)))
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].axis('off')
        
        # 显示差值图（相对于健康图像）
        healthy_img = sim.generate_feature_image(
            damaged_dofs=None,
            noise_level=noise,
            damage_subtlety=subtlety,
            add_texture=texture,
            random_seed=42
        )
        # healthy_img 的形状是 (3, 224, 224)
        diff_img = np.abs(img - healthy_img)
        diff_img = np.clip(diff_img * 5, 0, 1)  # 放大显示
        # diff_img 的形状是 (3, 224, 224)
        
        # 显示差值图 (需要转置为 (224, 224, 3))
        axes[1, col].imshow(np.transpose(diff_img, (1, 2, 0)))
        axes[1, col].set_title(f'差值图 (x5)', fontsize=10)
        axes[1, col].axis('off')
    
    plt.suptitle('损伤隐蔽程度对比 (损伤位于中间节点)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('damage_subtlety_comparison_noisier.png', dpi=150, bbox_inches='tight')
    print("✓ 可视化对比图已保存: damage_subtlety_comparison_noisier.png")


if __name__ == "__main__":
    print("=" * 70)
    print("优化后的MDOF仿真系统")
    print("=" * 70)
    
    visualize_damage_subtlety_comparison()

    # 测试参数
    n_samples = 100  # 样本数
    n_dof = 10 # 自由度
    
    # 生成数据
    # 记录程序开始执行的时间
    start_time = time.time()

    
    # 使用并行方式生成多自由度(MDOF)数据集
    # 参数说明：
    signals, images, labels = generate_mdof_dataset_parallel(  # 返回信号、图像和标签三个数据
        n_samples=n_samples,        # 样本数量
        n_dof=n_dof,                # 自由度数量
        segment_time=10.0,          # 每个片段的时间长度(秒)
        n_jobs=-1,                  # 使用所有可用的CPU核心进行并行计算
        force_regen=False,  # 设为 True 强制重新生成数据集
        seed=42,                    # 随机种子，确保结果可复现
        noise_level=0.05,      # 噪声水平
        damage_subtlety=0.2,   # 损伤隐蔽程度
        add_texture=True        # 添加纹理
    )
    end_time = time.time()
    
    # 打印结果
    print(f"\n{'='*70}")
    print("数据生成完成！")
    print(f"{'='*70}")
    print(f"样本数: {n_samples}")
    print(f"信号形状: {signals.shape} (n_samples, n_dof, n_timepoints)")
    print(f"图像形状: {images.shape} (n_samples, C, H, W)")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每个样本: {(end_time - start_time)/n_samples*1000:.2f} ms")
    print(f"{'='*70}")
    
    # 可视化
    visualize_results(signals, images, labels)
    
    print("\n提示: 将 signals, images, labels 输入到你的 MLP-ResNet50 模型中进行训练。")
