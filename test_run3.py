import numpy as np
import torch
from model1 import OffshoreDamageDetectionSystem
import h5py
from load_h5 import load_h5_dataset
import matplotlib.pyplot as plt
import os
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import bisect
from sklearn.model_selection import train_test_split


class H5LazyDataset(Dataset):
    """
    惰性加载 HDF5 数据集，避免一次性将全部数据读入内存
    """
    def __init__(self, data_dir, feature_selection='mean', transform=None):
        self.data_dir = data_dir
        self.feature_selection = feature_selection
        self.transform = transform
        
        # 1. 扫描目录，构建文件索引映射
        self.h5_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.h5')],
            key=lambda x: int(x.replace('scenario_', '').replace('.h5', ''))
        )
        
        # 预计算每个文件的样本数，构建累积索引表 (cumulative_samples)
        # 例如: [201, 402, 603, ...] 代表第1个文件结束于索引201，第2个于402...
        self.cumulative_samples = []
        self.file_metadata = [] # 存储每个文件的 window_length, step_size 等
        self.acc_cache = []

        total_samples = 0
        for fname in self.h5_files:
            fpath = os.path.join(data_dir, fname)
            with h5py.File(fpath, 'r') as hf:
                n_windows = hf['feature_maps'].shape[0]
                win_len = hf.attrs.get('window_length', 2000)
                step_size = hf.attrs.get('step_size', 50)
                acc_data = hf['acceleration'][:].T
                self.acc_cache.append(acc_data)

            total_samples += n_windows
            self.cumulative_samples.append(total_samples)
            self.file_metadata.append({
                'path': fpath,
                'window_length': win_len,
                'step_size': step_size
            })
            
        self.total_samples = total_samples
        print(f"[LazyDataset] 初始化完成，共 {len(self.h5_files)} 个文件，{self.total_samples} 个样本")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 1. 定位 idx 对应的文件和局部窗口索引
        # bisect_right 找到 idx 落在哪个文件的区间内
        file_idx = bisect.bisect_right(self.cumulative_samples, idx)
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_samples[file_idx - 1]
            
        meta = self.file_metadata[file_idx]
        fpath = meta['path']
        win_len = meta['window_length']
        step_size = meta['step_size']
        
        # 2. 从文件中读取所需数据 (核心优化：只读取这一个样本)
        with h5py.File(fpath, 'r') as hf:
            # 读取标签和图像 (图像很小，直接读)
            label = int(hf['damage_class'][0])
            image = hf['feature_maps'][local_idx] # (H, W, 3) uint8 or float
            
            # 读取加速度信号 (这里需要读取全长，然后切片)
            # 注意：new_mdof_v1.py 保存的 acc 是 (time_steps, n_dof)
            # load_h5.py 中做了转置 acc.T -> (n_dof, time_steps)
            acc_full = hf['acceleration'][:] 
            acc_full = acc_full.T # 转置为 (n_dof, time_steps) 以便切片
            
            # 模拟 load_h5.py 的切片逻辑
            start_idx = local_idx * step_size
            end_idx = start_idx + win_len
            
            if end_idx > acc_full.shape[1]:
                # 边界处理
                window_acc = acc_full[:, start_idx:]
                pad_width = win_len - (end_idx - start_idx)
                window_acc = np.pad(window_acc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                window_acc = acc_full[:, start_idx:end_idx]
        
        # 3. 图像维度处理 (H, W, C) -> (C, H, W) 并转为 float
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        # 4. 特征提取 (复用 model1 中的逻辑)
        # 避免依赖外部类，这里直接实现简单的特征提取
        if self.feature_selection == 'mean':
            window_acc = np.mean(window_acc, axis=0) # 平均所有传感器
        elif self.feature_selection == 'first_sensor':
            window_acc = window_acc[0, :]
        
        # 提取 16 维统计特征
        features = self._extract_statistical_features(window_acc)
        
        # 5. 转换为 Tensor
        time_series = torch.FloatTensor(features)
        image_tensor = torch.FloatTensor(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 6. 图像变换
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return time_series, image_tensor, label_tensor

    def _extract_statistical_features(self, signal_window):
        """提取 16 维统计特征"""
        features = [
            np.mean(signal_window),
            np.std(signal_window),
            np.max(signal_window),
            np.min(signal_window),
            np.percentile(signal_window, 25),
            np.percentile(signal_window, 75),
            stats.skew(signal_window),
            stats.kurtosis(signal_window),
            np.sqrt(np.mean(signal_window**2)),  # RMS
            np.ptp(signal_window),  # Peak-to-peak
            np.sum(np.abs(np.diff(signal_window))) / len(signal_window),
            np.mean(np.abs(signal_window - np.mean(signal_window))),
            np.var(signal_window),
            np.median(np.abs(signal_window - np.median(signal_window))),
            np.sum(signal_window**2) / len(signal_window),
            np.sum(np.abs(signal_window)),
        ]
        return np.array(features)


def main_with_new_simulator():
    """
    使用新仿真器生成的HDF5数据训练损伤检测模型
    """
    print("=" * 70)
    print("使用新仿真器 (HDF5) 数据训练损伤检测模型")
    print("=" * 70)
    
    # ==================== 参数配置 ====================
    DATA_DIR = './jacket_damage_data'
    NUM_CLASSES = 5  # 0=健康, 1-3=单损伤不同程度, 4=多损伤
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    
    # ==================== 1. 加载数据 ====================
    print("\n[步骤1] 加载仿真数据...")
    
    if not os.path.exists(DATA_DIR):
        print(f"✗ 错误: 数据目录 {DATA_DIR} 不存在！")
        print("  请先运行以下命令生成数据:")
        print("  python new_mdof_v1.py")
        return
    
    full_dataset = H5LazyDataset(
        data_dir=DATA_DIR,
        feature_selection='mean',
        transform=None  # 暂时不传 transform，后续在 system 里统一加
    )
    
    # 检查数据量
    n_samples = len(full_dataset)
    if n_samples < 5000:
        print(f"\n⚠️  警告: 当前样本数 ({n_samples}) 可能不足以充分训练深度学习模型！")
        print("  建议: 在 new_mdof_v1.py 中增加 num_scenarios 到至少 100")
        print(f"  当前每个场景约产生 {n_samples // len(os.listdir(DATA_DIR))} 个样本")
    else:
        print(f"\n✓ 样本数 ({n_samples}) 足够进行训练")
    
    # ==================== 2. 初始化检测系统 ====================
    print("\n[步骤2] 初始化检测系统...")
    
    detection_system = OffshoreDamageDetectionSystem(
        num_classes=NUM_CLASSES,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"  ✓ 设备: {detection_system.device}")
    print(f"  ✓ 类别数: {NUM_CLASSES}")
    print(f"  ✓ 模型参数量: {sum(p.numel() for p in detection_system.model.parameters()):,}")
    
    full_dataset.transform = detection_system.train_transform # 验证和测试集会单独覆盖

    # ==================== 3. 划分数据集 (关键修改) ====================
    print("\n[步骤3] 划分数据集索引...")
    
    # 生成索引列表 [0, 1, 2, ..., n_samples-1]
    indices = np.arange(n_samples)
    
    # 使用 train_test_split 只划分"索引"，极其节省内存
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )
    
    print(f"  ✓ 训练集索引数: {len(train_idx)}")
    print(f"  ✓ 验证集索引数: {len(val_idx)}")
    print(f"  ✓ 测试集索引数: {len(test_idx)}")

    # 使用 Subset 创建子数据集 (这只是对索引的包装，不加载实际数据)
    # 训练集使用训练时的 transform (包含数据增强)
    train_dataset = Subset(full_dataset, train_idx)
    # 验证集和测试集需要重新创建一个 dataset 实例或修改 transform 属性，
    # 因为原始 dataset 的 transform 可能是 train_transform
    # 这里为了简单，我们克隆一个 dataset 用于 val/test
    val_dataset_base = H5LazyDataset(DATA_DIR, feature_selection='mean', transform=detection_system.test_transform)
    test_dataset_base = H5LazyDataset(DATA_DIR, feature_selection='mean', transform=detection_system.test_transform)
    
    val_dataset = Subset(val_dataset_base, val_idx)
    test_dataset = Subset(test_dataset_base, test_idx)

    # ==================== 4. 创建 DataLoader ====================
    print("\n[步骤4] 创建 DataLoader...")
    batch_size = BATCH_SIZE
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 注意：如果是惰性加载，num_workers > 0 可能会导致文件读取冲突或预处理负担，
    # 如果内存够用可以设为 2 或 4，否则保持 0 (调试模式)
    
    # ==================== 4. 训练模型 ====================
    print("\n[步骤5] 训练模型...")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  早停耐心值: {EARLY_STOPPING_PATIENCE}")
    
    history = detection_system.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # 绘制训练曲线
    detection_system.plot_training_history(
        history, 
        save_path='training_history.png'
    )
    
    # ==================== 5. 评估模型 ====================
    print("\n[步骤5] 评估模型性能...")
    
    metrics = detection_system.evaluate(test_loader)
    
    print(f"\n{'='*50}")
    print("测试集性能:")
    print(f"{'='*50}")
    print(f"准确率:   {metrics['accuracy']:.4f}")
    print(f"精确率:   {metrics['precision_macro']:.4f}")
    print(f"召回率:   {metrics['recall_macro']:.4f}")
    print(f"F1分数:   {metrics['f1_macro']:.4f}")
    
    # 绘制混淆矩阵
    class_names = ['健康', '轻微损伤', '中度损伤', '重度损伤', '多损伤']
    detection_system.plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names,
        save_path='confusion_matrix.png'
    )
    
    # ==================== 6. 保存模型 ====================
    print("\n[步骤6] 保存模型...")
    model_path = 'new_simulator_trained_model.pth'
    torch.save(detection_system.model.state_dict(), model_path)
    print(f"  ✓ 模型已保存: {model_path}")
    
    print("\n" + "=" * 70)
    print("✓ 训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    main_with_new_simulator()
