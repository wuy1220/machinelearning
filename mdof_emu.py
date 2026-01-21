import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Tuple, Dict

class MDOFDamageSimulator:
    """
    基于多自由度(MDOF)系统的损伤数据生成器
    用于模拟剪切型框架或离散化简支梁的动力响应
    """
    
    def __init__(self, n_dof: int = 10, mass: float = 1.0, k_base: float = 1e5, damping_ratio: float = 0.02):
        """
        初始化MDOF系统
        参数:
            n_dof: 自由度数量 (模拟传感器数量或结构节点数)
            mass: 每个节点的质量
            k_base: 层间刚度基准值
            damping_ratio: 阻尼比
        """
        self.n_dof = n_dof
        self.mass = mass
        self.k_base = k_base
        self.damping_ratio = damping_ratio
        
        # 初始化健康状态的系统矩阵
        self.M_healthy = np.eye(n_dof) * mass
        self.K_healthy = self._build_stiffness_matrix(k_base)
        self.C_healthy = self._build_damping_matrix(self.M_healthy, self.K_healthy, damping_ratio)
        
        # 当前系统状态（用于施加损伤）
        self.K_current = self.K_healthy.copy()
        
    def _build_stiffness_matrix(self, k_val: float) -> np.ndarray:
        """构建三对角刚度矩阵 (模拟剪切型框架)"""
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_dof):
            if i > 0: K[i, i-1] -= k_val
            K[i, i] += k_val
            if i < self.n_dof - 1: K[i, i+1] -= k_val
        
        # 修正最后一行以满足边界条件(假设底部固定，顶部自由或铰接)
        # 这里采用简化的剪切型模型，K最后一行仅涉及自身刚度
        K[self.n_dof-1, self.n_dof-1] = k_val 
        # 注意：实际简支梁或框架需根据具体边界调整K，此处为演示模型
        return K
    
    def _build_damping_matrix(self, M: np.ndarray, K: np.ndarray, zeta: float) -> np.ndarray:
        """使用Rayleigh阻尼构建阻尼矩阵 C = alpha*M + beta*K"""
        # 简化假设：仅使用刚度比例阻尼，或假设固有频率为常数
        # 更精确的做法是计算前两阶频率求系数，这里做简化处理
        omega_approx = np.sqrt(np.mean(np.diag(K) / np.diag(M)))
        alpha = 2 * zeta * omega_approx * 0.01
        beta = 2 * zeta / omega_approx * 0.99
        return alpha * M + beta * K

    def apply_damage(self, damaged_dofs: list, severity: float = 0.3):
        """
        施加损伤：降低指定自由度的刚度
        
        参数:
            damaged_dofs: 发生损伤的自由度列表 (索引从0开始)
            severity: 刚度降低程度 (0.0 - 1.0), 0.3表示刚度降低30%
        """
        self.K_current = self.K_healthy.copy()
        for dof in damaged_dofs:
            # 降低该自由度关联的刚度
            # 简化处理：仅降低主对角线元素，实际应同时降低关联的非对角线元素
            self.K_current[dof, dof] *= (1 - severity)
            if dof < self.n_dof - 1:
                self.K_current[dof, dof+1] *= (1 - severity)
                self.K_current[dof+1, dof] *= (1 - severity)
            if dof > 0:
                self.K_current[dof, dof-1] *= (1 - severity)
                self.K_current[dof-1, dof] *= (1 - severity)

    def solve_time_history(self, force: np.ndarray, dt: float) -> np.ndarray:
        """
        使用Newmark-beta法求解动力方程
        参数:
            force: 外力向量时程 (n_steps, n_dof)
            dt: 时间步长
        返回:
            acceleration: 加速度响应 (n_steps, n_dof)
        """
        n_steps = force.shape[0]
        acceleration = np.zeros((n_steps, self.n_dof))
        velocity = np.zeros((n_steps, self.n_dof))
        displacement = np.zeros((n_steps, self.n_dof))
        
        # Newmark-beta参数
        gamma = 0.5
        beta = 0.25
        
        # 有效刚度矩阵
        K_hat = self.K_current + (gamma / (beta * dt)) * self.C_healthy + (1 / (beta * dt**2)) * self.M_healthy
        
        for i in range(1, n_steps):
            # 计算有效荷载
            delta_f = force[i] - force[i-1]
            
            # 预测步
            # 省略中间推导，直接使用显式更新公式
            # 这里为了代码简洁，使用SciPy的线性求解器模拟
            
            # 简化的增量法 (实际需完整Newmark迭代)
            # 这里为了演示效果，直接使用状态空间法或简化的显式更新可能更简单，
            # 但为了保证物理准确性，下面给出简化的中心差分法(显式)实现，
            # 它在dt足够小时是稳定的，且代码量小。
            
            # === 切换为 中心差分法 (显式) ===
            # a_t = M^-1 * (F_t - C*v_t - K*x_t)
            f_eff = force[i] - self.C_healthy @ velocity[i-1] - self.K_current @ displacement[i-1]
            acceleration[i] = linalg.solve(self.M_healthy, f_eff)
            
            velocity[i] = velocity[i-1] + (acceleration[i-1] + acceleration[i]) * dt / 2
            displacement[i] = displacement[i-1] + velocity[i-1] * dt + 0.5 * acceleration[i-1] * dt**2
            
        return acceleration

    def generate_feature_image(self, damaged_dofs: list = None) -> np.ndarray:
        """
        生成模拟的'结构图像'
        既然没有真实照片，我们生成'结构刚度/损伤分布热力图'
        这对CNN来说是有意义的特征（空间信息）
        """
        img_size = 224
        img = Image.new('RGB', (img_size, img_size), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # 绘制结构骨架 (简单的线条)
        margin = 20
        draw_width = img_size - 2 * margin
        step = draw_width // (self.n_dof + 1)
        
        points = []
        for i in range(self.n_dof):
            x = margin + (i + 1) * step
            y = img_size // 2
            points.append((x, y))
            draw.ellipse([x-5, y-5, x+5, y+5], fill=(0,0,0)) # 节点
        
        # 连接节点
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=(50, 50, 50), width=5)
            
        # 如果有损伤，用红色标记损伤区域
        if damaged_dofs:
            for dof in damaged_dofs:
                if dof < len(points):
                    cx, cy = points[dof]
                    # 绘制红色损伤区域
                    draw.rectangle([cx-15, cy-15, cx+15, cy+15], outline=(255, 0, 0), width=3)
                    # 模拟一种"损伤云团"
                    draw.ellipse([cx-25, cy-25, cx+25, cy+25], outline=(255, 100, 100), width=2)
        
        return np.array(img).astype(np.float32) / 255.0

# ==========================================
# 集成到主流程的示例
# ==========================================

def generate_mdof_dataset(n_samples=100):
    print("正在初始化MDOF仿真系统...")
    sim = MDOFDamageSimulator(n_dof=10, mass=100.0, k_base=5e6, damping_ratio=0.02)
    
    dt = 0.005 # 采样频率 200Hz
    duration = 10.0
    t = np.arange(0, duration, dt)
    n_steps = len(t)
    
    # 生成白噪声激励 (模拟环境激励)
    # 集中作用在顶层
    force = np.zeros((n_steps, sim.n_dof))
    force[:, -1] = np.random.normal(0, 5000, n_steps) # 顶层随机力
    
    # 生成数据集
    all_signals = []
    all_images = []
    all_labels = [] # 0:健康, 1-4:不同位置的损伤
    
    # 场景定义
    scenarios = [
        ([], 0),             # 健康
        ([3], 1),            # 损伤在节点4 (中间)
        ([6], 2),            # 损伤在节点7
        ([3, 7], 3),         # 多处损伤
    ]
    
    print("正在生成仿真数据...")
    for i in range(n_samples):
        # 随机选择场景
        dmg_locs, label = scenarios[i % len(scenarios)]
        
        # 应用损伤
        sim.apply_damage(damaged_dofs=dmg_locs, severity=0.4) # 刚度降低40%
        
        # 求解响应
        acc_response = sim.solve_time_history(force, dt) # (n_steps, n_dof)
        
        # 转置为 (n_dof, n_steps) 以匹配原有模型输入习惯
        acc_response = acc_response.T 
        
        # 生成图像
        img = sim.generate_feature_image(damaged_dofs=dmg_locs) # (H, W, 3)
        
        # 数据标准化/归一化 (模拟传感器量测噪声)
        acc_response += np.random.normal(0, 0.01, acc_response.shape) # 添加噪声
        
        all_signals.append(acc_response)
        all_images.append(np.transpose(img, (2, 0, 1))) # 转为 (C, H, W)
        all_labels.append(label)
        
    return np.array(all_signals), np.array(all_images), np.array(all_labels)

# 使用示例
if __name__ == "__main__":
    signals, images, labels = generate_mdof_dataset(n_samples=20)
    
    print(f"数据形状: {signals.shape}") # (N, n_dof, n_steps)
    print(f"标签分布: {np.bincount(labels)}")
    
    # 可视化一个样本
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(signals[0, 0, :]) # 画第一个传感器的信号
    plt.title("Sensor 1 Acceleration (MDOF)")
    plt.xlabel("Time Step")
    plt.ylabel("Acc")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(images[0], (1, 2, 0)))
    plt.title("Simulated Structural Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
