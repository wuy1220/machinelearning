import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GVRDatasetInspector:
    """
    GVR 特征图数据集检查工具
    
    功能：
    1. 加载分片存储的 HDF5 数据
    2. 可视化生成的 RGB 特征图及其 R/G/B 分量
    3. 可视化原始加速度信号，并标记特征图对应的时间窗口位置
    4. 显示损伤标签和元数据信息
    """
    
    def __init__(self, data_dir, dt=0.005):
        """
        Args:
            data_dir: 数据生成目录 (包含 .h5 文件和 metadata.json)
            dt: 采样时间间隔，用于绘制时间轴，默认 0.005
        """
        self.data_dir = data_dir
        self.dt = dt
        self.metadata_path = os.path.join(data_dir, 'metadata.json')
        
        # 加载元数据
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
            
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Inspector initialized: {len(self.metadata)} scenarios loaded from {data_dir}")

    def _load_scenario_data(self, meta_idx):
        """根据元数据索引加载单个场景的数据"""
        meta = self.metadata[meta_idx]
        shard_id = meta['shard_id']
        group_name = meta['group_name']
        
        shard_path = os.path.join(self.data_dir, f'data_shard_{shard_id:04d}.h5')
        
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
            
        with h5py.File(shard_path, 'r') as f:
            grp = f[group_name]
            data = {
                'acceleration': grp['acceleration'][:],
                'feature_maps': grp['feature_maps'][:],
                'labels': grp['labels'][:],
                'damage_class': grp['damage_class'][0],
                'attrs': dict(grp.attrs) # 获取所有属性
            }
            
        return data

    def inspect_scenario(self, meta_idx, window_idx='middle', save_path=None):
        """
        检查指定场景的特征图
        
        Args:
            meta_idx: 元数据列表中的索引 (0 到 total_scenarios-1)
            window_idx: 要查看哪个窗口的特征图。
                        'middle' (默认), 'random', 或具体的整数索引 (0, 1, 2...)
            save_path: 如果提供，将图像保存到该路径
        """
        # 1. 加载数据
        data = self._load_scenario_data(meta_idx)
        acc = data['acceleration']
        feat_maps = data['feature_maps']
        attrs = data['attrs']
        
        # 参数获取
        window_length = attrs.get('window_length', 2000)
        step_size = attrs.get('step_size', 50)
        num_windows = feat_maps.shape[0]
        
        # 2. 确定要展示的窗口索引
        if isinstance(window_idx, str):
            if window_idx == 'middle':
                target_w = num_windows // 2
            elif window_idx == 'random':
                target_w = np.random.randint(0, num_windows)
            else:
                target_w = 0
        else:
            target_w = window_idx
            if target_w >= num_windows:
                target_w = num_windows - 1
                
        current_img = feat_maps[target_w] # (224, 224, 3)
        
        # 3. 准备绘图
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 3, 2])
        
        # --- 顶部：标题信息 ---
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        info_text = (
            f"Scenario Index: {meta_idx} | ID: {attrs['scenario_id']} | Window: {target_w}/{num_windows-1}\n"
            f"Damage Class: {data['damage_class']} | Damaged DOFs: {attrs['damaged_dofs']} | Severity: {attrs['severity_ratios']}"
        )
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.5))

        # --- 中部：特征图可视化 ---
        # RGB 图像
        ax_rgb = fig.add_subplot(gs[1, 0])
        ax_rgb.imshow(current_img)
        ax_rgb.set_title("GVR Feature Map (RGB)", fontsize=12)
        ax_rgb.set_xlabel("Sensor Position (X)")
        ax_rgb.set_ylabel("Intra-window Time (Y)")
        
        # R 通道 (DI')
        ax_r = fig.add_subplot(gs[1, 1])
        ax_r.imshow(current_img[:, :, 0], cmap='Reds', vmin=0, vmax=1)
        ax_r.set_title("R Channel: DI' (Trend)", fontsize=12)
        ax_r.axis('off')
        
        # G 通道 (DI'')
        ax_g = fig.add_subplot(gs[1, 2])
        ax_g.imshow(current_img[:, :, 1], cmap='Greens', vmin=0, vmax=1)
        ax_g.set_title("G Channel: DI'' (Mutation)", fontsize=12)
        ax_g.axis('off')
        
        # --- 底部：原始信号与窗口标记 ---
        ax_sig = fig.add_subplot(gs[2, :])
        
        # 为了清晰，只绘制前几个传感器或所有传感器的平均值
        # 这里绘制所有传感器的加速度
        time_axis = np.arange(acc.shape[0]) * self.dt
        
        # 绘制所有通道（半透明）
        for ch in range(acc.shape[1]):
            ax_sig.plot(time_axis, acc[:, ch] + ch*0.5, color='gray', alpha=0.3, linewidth=0.5)
            
        # 计算当前窗口的时间范围
        win_start_time = target_w * step_size * self.dt
        win_end_time = (target_w * step_size + window_length) * self.dt
        
        # 标记窗口区域
        ax_sig.axvspan(win_start_time, win_end_time, color='yellow', alpha=0.2, label='Current Window')
        ax_sig.set_title(f"Raw Acceleration Signals (Offset for visibility)\nWindow Span: [{win_start_time:.2f}s, {win_end_time:.2f}s]", fontsize=12)
        ax_sig.set_xlabel("Time (s)")
        ax_sig.set_ylabel("Sensor Channels (Offset)")
        ax_sig.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Image saved to {save_path}")
        else:
            plt.show()
            
    def batch_inspect(self, num_samples=5, window_choice='middle'):
        """随机抽取几个场景进行快速检查"""
        indices = np.random.choice(len(self.metadata), size=min(num_samples, len(self.metadata)), replace=False)
        for idx in indices:
            print(f"\nInspecting metadata index {idx}...")
            self.inspect_scenario(idx, window_idx=window_choice)

# ===== 使用示例 =====
if __name__ == "__main__":
    # 修改这里为您实际生成的数据目录
    DATA_DIR = './jacket_damage_data_timespace3' 
    
    if os.path.exists(DATA_DIR):
        inspector = GVRDatasetInspector(DATA_DIR)
        
        # 1. 检查特定的场景 (比如第5个)
        #inspector.inspect_scenario(meta_idx=5, window_idx='middle')
        
        # 2. 随机检查 3 个场景
        inspector.batch_inspect(num_samples=10)
    else:
        print(f"Directory {DATA_DIR} does not exist. Please update the path.")
