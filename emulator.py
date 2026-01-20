import xara
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 模型定义 (参考论文 Section 3.1)
# ==========================================
def build_jacket_model(damaged=False):
    """
    构建导管架模型
    damaged: 布尔值，True表示引入损伤
    返回: xara.Model 实例
    """
    # 使用 xara.Model 实例化模型（替代全局 opensees）
    model = xara.Model(ndm=3, ndf=6)  # 3D 模型, 每节点 6 个自由度

    # -----------------------
    # 参数设置
    # -----------------------
    # 导管架尺寸
    width = 10.0        # 米
    depth = 20.0        # 米 (水深)
    deck_height = 25.0  # 甲板高度
    
    # 材料参数
    E_healthy = 2.1e11    # 健康弹性模量
    E_damaged = 1.05e11   # 损伤弹性模量 (降低50%)
    A = 0.01              # 截面积
    Iz = 0.0001           # 惯性矩
    Iy = 0.0001
    G = 8.0e10            # 剪切模量
    J = 0.0002            # 扭转常数
    mass_dens = 7850.0    # 密度

    # -----------------------
    # 定义截面
    # -----------------------
    # 创建两个截面：健康(1)和损伤(2)
    model.section('Elastic', 1, E_healthy, A, Iz, Iy, G, J)
    model.section('Elastic', 2, E_damaged, A, Iz, Iy, G, J)
    
    # -----------------------
    # 定义节点
    # -----------------------
    # 底部节点 (节点 1-4) - 固定约束
    model.node(1, 0.0, 0.0, 0.0)
    model.node(2, width, 0.0, 0.0)
    model.node(3, width, width, 0.0)
    model.node(4, 0.0, width, 0.0)

    # 顶部节点 (节点 5-8)
    model.node(5, 0.0, 0.0, deck_height)
    model.node(6, width, 0.0, deck_height)
    model.node(7, width, width, deck_height)
    model.node(8, 0.0, width, deck_height)

    # 中间节点 (节点 9-12) - 增加刚度
    mid_h = depth / 2.0
    model.node(9, 0.0, 0.0, mid_h)
    model.node(10, width, 0.0, mid_h)
    model.node(11, width, width, mid_h)
    model.node(12, 0.0, width, mid_h)

    # -----------------------
    # 定义边界条件
    # -----------------------
    # 固定海底节点 (1-4)
    for i in range(1, 5):
        model.fix(i, 1, 1, 1, 1, 1, 1)

    # -----------------------
    # 定义几何变换
    # -----------------------
    model.geomTransf('Linear', 1, 0.0, 0.0, 1.0)
    
    # -----------------------
    # 定义单元
    # -----------------------
    # 腿柱单元 (腿)
    cols = [(1,9), (2,10), (3,11), (4,12), (9,5), (10,6), (11,7), (12,8)]
    for i, (n1, n2) in enumerate(cols, 1):
        # 如果指定了损伤，让第一个腿柱单元使用损伤截面
        sec_tag = 2 if (damaged and i == 1) else 1
        model.element('elasticBeamColumn', i, n1, n2, sec_tag, 1)

    # 交叉支撑单元
    braces = [(1,10), (2,9), (5,12), (6,11), (9,12), (10,11)]
    elem_start = len(cols) + 1
    for i, (n1, n2) in enumerate(braces, elem_start):
        model.element('elasticBeamColumn', i, n1, n2, 1, 1)  # 支撑使用健康截面

    # -----------------------
    # 定义阻尼 (Rayleigh damping)
    # -----------------------
    # C = alpha*M + beta*K
    model.rayleigh(0.05, 0.0, 0.001, 0.0)

    return model

# ==========================================
# 2. 运行分析函数
# ==========================================
def run_analysis(model, duration=30.0, dt=0.01):
    """
    运行瞬态动力分析并提取节点加速度响应
    """
    # 定义数值积分器
    model.integrator('Newmark', 0.5, 0.25)
    # 定义系统
    model.system('BandGeneral')
    # 定义分析类型
    model.analysis('Transient')
    
    # 准备存储数组
    time_steps = int(duration / dt)
    disp_node5 = np.zeros(time_steps)
    times = np.zeros(time_steps)
    
    # 运行分析循环
    for i in range(time_steps):
        ok = model.analyze(1, dt)
        if ok != 0:
            print(f"分析失败于步骤 {i}")
            break
        # 记录节点5的Z向位移 (dof=3)
        disp_node5[i] = model.nodeDisp(5, 3)
        times[i] = model.getTime()
    
    # 计算加速度 (参考论文 Eq. 6 的离散差分思路)
    # a(t) approx (d(t+dt) - 2d(t) + d(t-dt)) / dt^2
    acc_node5 = np.zeros_like(disp_node5)
    # 使用中心差分计算内部点
    acc_node5[1:-1] = (disp_node5[2:] - 2*disp_node5[1:-1] + disp_node5[:-2]) / (dt**2)
    # 边界点使用前向/后向差分
    acc_node5[0] = (disp_node5[2] - 2*disp_node5[1] + disp_node5[0]) / (dt**2)
    acc_node5[-1] = acc_node5[-2]
    
    return times, acc_node5

# ==========================================
# 3. 执行流程
# ==========================================

print("--- 场景 1: 健康结构 ---")
healthy_model = build_jacket_model(damaged=False)
time_healthy, acc_healthy = run_analysis(healthy_model)

print(f"提取到 {len(time_healthy)} 个时间步数据")
print(f"健康结构加速度样本: {acc_healthy[1000:1010]}")


print("\n--- 场景 2: 损伤结构 (修改 E 值) ---")
damaged_model = build_jacket_model(damaged=True)
time_damaged, acc_damaged = run_analysis(damaged_model)

print(f"损伤结构加速度样本: {acc_damaged[1000:1010]}")


# ==========================================
# 4. 数据可视化
# ==========================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time_healthy[1000:1200], acc_healthy[1000:1200], label='Healthy')
plt.plot(time_damaged[1000:1200], acc_damaged[1000:1200], label='Damaged', linestyle='--')
plt.title("Acceleration Response (Node 5)")
plt.xlabel("Time (s)")
plt.ylabel("Acc (m/s^2)")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(acc_healthy, bins=50, alpha=0.5, label='Healthy')
plt.hist(acc_damaged, bins=50, alpha=0.5, label='Damaged')
plt.title("Distribution Comparison")
plt.legend()

plt.tight_layout()
plt.show()

# 输出简单的统计特征
print("\n--- 统计特征对比 ---")
print(f"健康结构加速度 RMS: {np.sqrt(np.mean(acc_healthy**2)):.6f} m/s²")
print(f"损伤结构加速度 RMS: {np.sqrt(np.mean(acc_damaged**2)):.6f} m/s²")
print(f"健康结构加速度最大值: {np.max(np.abs(acc_healthy)):.6f} m/s²")
print(f"损伤结构加速度最大值: {np.max(np.abs(acc_damaged)):.6f} m/s²")
