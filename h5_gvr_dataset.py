import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import stats
import random

class TimeSeriesCompose:
    """组合多个时序增强操作"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal

class AddGaussianNoise:
    """添加高斯噪声"""
    def __init__(self, mean=0.0, std=0.01):
        """
        Args:
            std: 噪声标准差 (相对于信号标准差的倍数)
                 例如 0.01 表示添加信号幅度 1% 的噪声
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor shape: (Time,)
        if self.std > 0:
            noise = torch.randn(tensor.size()) * self.std * torch.std(tensor) + self.mean
            return tensor + noise
        return tensor

class RandomScaling:
    """随机幅度缩放"""
    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Args:
            scale_range: 缩放因子范围 (min, max)
        """
        self.scale_range = scale_range

    def __call__(self, tensor):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        return tensor * scale

class RandomTimeShift:
    """随机时间平移"""
    def __init__(self, max_shift=10):
        """
        Args:
            max_shift: 最大平移点数
        """
        self.max_shift = max_shift

    def __call__(self, tensor):
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return tensor
        # torch.roll 实现循环移位
        return torch.roll(tensor, shifts=shift, dims=0)

class RandomCutout:
    """随机遮盖一小段信号"""
    def __init__(self, max_len=50):
        self.max_len = max_len

    def __call__(self, tensor):
        length = tensor.shape[0]
        if length <= 10: return tensor
        
        cut_len = random.randint(5, self.max_len)
        start_pos = random.randint(0, length - cut_len)
        
        tensor[start_pos : start_pos + cut_len] = 0.0
        return tensor


class H5GVRDataset(Dataset):
    """
    适配新仿真数据的 PyTorch Dataset
    读取 HDF5 文件，提取 GVR 图像和对应的窗口加速度信号特征
    """
    def __init__(self, data_dir, window_length=3000, transform=None):
        self.data_dir = data_dir
        self.window_length = window_length
        self.transform = transform
        
        # 扫描并构建索引
        self.h5_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith('data_shard_') and f.endswith('.h5')],
            key=lambda x: int(x.split('_')[2].split('.')[0])
        )
        
        self.sample_metadata = []
        print(f"正在扫描数据目录: {data_dir} ...")
        
        for fname in self.h5_files:
            fpath = os.path.join(data_dir, fname)
            with h5py.File(fpath, 'r') as hf:
                for group_name in hf.keys():
                    grp = hf[group_name]
                    # 每个场景包含多个窗口的图像
                    n_windows = grp['feature_maps'].shape[0]
                    label = int(grp['damage_class'][0])
                    step_size = grp.attrs.get('step_size', 50)
                    
                    # 记录每一个窗口作为一个样本
                    for win_idx in range(n_windows):
                        self.sample_metadata.append({
                            'fname': fname,
                            'group': group_name,
                            'win_idx': win_idx,
                            'step_size': step_size,
                            'label': label
                        })
        
        print(f"✓ 扫描完成，共发现 {len(self.sample_metadata)} 个样本窗口")

    def __len__(self):
        return len(self.sample_metadata)

    def _extract_features(self, signal):
        """
        提取时序特征 (FFT + 统计量)，与 run_3.py 中的逻辑一致
        输出维度: 200 (FFT) + 16 (Stats) = 216
        """
        # 1. FFT 特征
        fft_vals = np.abs(np.fft.rfft(signal))
        fft_features = np.log1p(fft_vals[:200])
        # 简单标准化
        fft_features = (fft_features - np.mean(fft_features)) / (np.std(fft_features) + 1e-8)
        
        # 2. 统计特征
        stats_features = np.array([
            np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
            np.percentile(signal, 25), np.percentile(signal, 75),
            stats.skew(signal), stats.kurtosis(signal),
            np.sqrt(np.mean(signal**2)), np.ptp(signal),
            np.sum(np.abs(np.diff(signal))) / len(signal),
            np.mean(np.abs(signal - np.mean(signal))),
            np.var(signal), np.median(np.abs(signal - np.median(signal))),
            np.sum(signal**2) / len(signal), np.sum(np.abs(signal))
        ])
        stats_features = (stats_features - np.mean(stats_features)) / (np.std(stats_features) + 1e-8)
        
        return np.concatenate([fft_features, stats_features])

    def __getitem__(self, idx):
        meta = self.sample_metadata[idx]
        fpath = os.path.join(self.data_dir, meta['fname'])
        
        with h5py.File(fpath, 'r') as hf:
            grp = hf[meta['group']]
            
            # 1. 读取图像
            # HDF5 存储为 (H, W, C)，PyTorch 需要 (C, H, W)
            img = grp['feature_maps'][meta['win_idx']]
            img = img.transpose(2, 0, 1).astype(np.float32)
            
            # 2. 读取加速度信号
            # 计算当前窗口对应的信号起止位置
            start_t = meta['win_idx'] * meta['step_size']
            end_t = start_t + self.window_length
            
            acc_window = grp['acceleration'][start_t:end_t, :]
            
            # 边界处理：如果长度不足，补零
            if acc_window.shape[0] < self.window_length:
                padding = self.window_length - acc_window.shape[0]
                acc_window = np.pad(acc_window, ((0, padding), (0, 0)), mode='constant')
            
            # 3. 生成时序特征向量
            # 策略：对所有传感器求平均，变成单通道信号
            signal_1d = np.mean(acc_window, axis=1)
            #time_series = self._extract_features(signal_1d)
            time_series = signal_1d

            # 4. 读取标签
            label = meta['label']

        # 转为 Tensor
        time_series_tensor = torch.from_numpy(time_series).float()  # Shape: (216,)
        image_tensor = torch.from_numpy(img).float()                # Shape: (3, 224, 224)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 图像增强
        if self.transform:
            image_tensor = self.transform(image_tensor)
        # 时序增强    
        if self.ts_transform:
            time_series_tensor = self.ts_transform(time_series_tensor)
            

        return time_series_tensor, image_tensor, label_tensor
