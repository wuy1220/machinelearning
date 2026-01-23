import numpy as np
import h5py
import os
from typing import Tuple

def load_h5_dataset(data_dir: str = './jacket_damage_data') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从压缩后的 HDF5 文件加载数据
    修正逻辑：每个样本只对应其窗口时间段的加速度片段，而非全长信号
    """
    h5_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.h5')],
        key=lambda x: int(x.replace('scenario_', '').replace('.h5', ''))
    )
    
    if len(h5_files) == 0:
        raise ValueError(f"在 {data_dir} 目录中未找到.h5文件。")

    print(f"找到 {len(h5_files)} 个HDF5文件，正在加载...")
    
    signals_list = []
    images_list = []
    labels_list = []
    
    for h5_file in h5_files:
        file_path = os.path.join(data_dir, h5_file)
        with h5py.File(file_path, 'r') as hf:
            # 读取完整加速度数据
            acc = hf['acceleration'][:]  # shape: (time_steps, n_dof)
            feat_maps = hf['feature_maps'][:]  # shape: (n_windows, 224, 224, 3)
            damage_class = int(hf['damage_class'][0])
            
            # 读取窗口参数 (如果新数据包含该属性)
            window_len = hf.attrs.get('window_length', 2000)
            step_size = hf.attrs.get('step_size', 50)
            
            # 转置: (time_steps, n_dof) -> (n_dof, time_steps) 方便后续切片
            acc = acc.T 
            
            n_windows = feat_maps.shape[0]
            n_dof = acc.shape[0]
            
            # === 关键修正：按窗口切片加速度信号 ===
            # 之前的逻辑是每个窗口都存了60秒的信号，现在只存窗口对应的2秒(2000点)
            # 这样数据量在内存中会大幅减少，且逻辑更正确
            for i in range(n_windows):
                # 计算当前窗口的起始和结束索引
                start_idx = i * step_size
                end_idx = start_idx + window_len
                
                # 边界检查（防止最后一个窗口越界）
                if end_idx > acc.shape[1]:
                    # 如果越界，进行填充或截断，这里选择截断以保持维度一致
                    window_acc = acc[:, start_idx:]
                    # 填充0以保持长度一致
                    pad_width = window_len - (end_idx - start_idx)
                    window_acc = np.pad(window_acc, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    window_acc = acc[:, start_idx:end_idx]
                
                signals_list.append(window_acc)
                images_list.append(feat_maps[i])
                labels_list.append(damage_class)
    
    # 转换为numpy数组
    signals = np.array(signals_list)  # (n_samples, n_dof, window_len)
    images = np.array(images_list)   # (n_samples, 224, 224, 3)
    labels = np.array(labels_list)
    
    # 转换图像维度: (N, H, W, C) -> (N, C, H, W)
    images = images.transpose(0, 3, 1, 2)
    
    print(f"✓ 数据加载完成 (高效模式):")
    print(f"  - 样本总数: {len(signals)}")
    print(f"  - 信号形状: {signals.shape} (已切片为窗口长度)")
    print(f"  - 图像形状: {images.shape}")
    print(f"  - 标签分布: {np.bincount(labels)}")
    
    return signals, images, labels
