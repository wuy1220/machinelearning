import numpy as np
import torch
#from model1 import OffshoreDamageDetectionSystem
from model1_mn import OffshoreDamageDetectionSystem
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
    优化版数据集：
    1. HDF5 按需切片，减少 IO 读取量
    2. 预计算并缓存统计特征到 RAM，消除重复计算
    3. 利用 32GB 内存优势
    """
    def __init__(self, data_dir, feature_selection='mean', transform=None):
        self.data_dir = data_dir
        self.feature_selection = feature_selection
        self.transform = transform
        
        # 1. 扫描目录
        self.h5_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.h5')],
            key=lambda x: int(x.replace('scenario_', '').replace('.h5', ''))
        )
        
        # 2. 构建索引和元数据
        self.cumulative_samples = []
        self.file_metadata = []
        total_samples = 0
        
        # === 新增：预计算特征缓存 ===
        # 预估内存占用：100场景 * 1160样本 * 16维 * 4字节 ≈ 7.4MB，完全安全
        print("[Optimization] 正在预计算统计特征到内存 (这需要几分钟，但会极大加速训练)...")
        self.feature_cache = [] 
        
        for fname in self.h5_files:
            fpath = os.path.join(data_dir, fname)
            with h5py.File(fpath, 'r') as hf:
                n_windows = hf['feature_maps'].shape[0]
                win_len = hf.attrs.get('window_length', 2000)
                step_size = hf.attrs.get('step_size', 50)
                
                # --- 关键优化点 A：在这里一次性提取当前文件的所有特征 ---
                # 读取当前文件的所有加速度数据 (如果内存允许，甚至可以读全文件进一步加速)
                # 但为了安全，我们遍历窗口切片读取
                for i in range(n_windows):
                    start_idx = i * step_size
                    end_idx = start_idx + win_len
                    
                    # === 关键优化点 B：HDF5 按需切片读取 ===
                    # 不再读取 [:] (全部)，只读取 [start_idx:end_idx]
                    window_acc = hf['acceleration'][start_idx:end_idx, :].T 
                    
                    # 边界处理
                    if window_acc.shape[1] < win_len:
                        pad_width = win_len - window_acc.shape[1]
                        window_acc = np.pad(window_acc, ((0, 0), (0, pad_width)), mode='constant')
                    
                    # 特征选择
                    if self.feature_selection == 'mean':
                        signal = np.mean(window_acc, axis=0)
                    else:
                        signal = window_acc[0, :]
                    
                    # 提取特征并存入缓存
                    feat = self._extract_statistical_features(signal)
                    self.feature_cache.append(feat)
                    
                # 元数据记录
                total_samples += n_windows
                self.cumulative_samples.append(total_samples)
                self.file_metadata.append({
                    'path': fpath,
                    'window_length': win_len,
                    'step_size': step_size
                })
        
        # 将列表转为 numpy 数组，加快索引访问速度
        self.feature_cache = np.array(self.feature_cache, dtype=np.float32)
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        self.total_samples = total_samples
        print(f"[Optimization] 特征缓存完成！")
        print(f"  - 缓存特征形状: {self.feature_cache.shape}")
        print(f"  - 预估内存占用: {self.feature_cache.nbytes / 1024 / 1024:.2f} MB")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 1. 定位文件
        file_idx = bisect.bisect_right(self.cumulative_samples, idx)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_samples[file_idx - 1]
            
        meta = self.file_metadata[file_idx]
        fpath = meta['path']
        
        # 2. 读取图像 (图像还是得从文件读，因为占内存太大)
        with h5py.File(fpath, 'r') as hf:
            label = int(hf['damage_class'][0])
            image = hf['feature_maps'][local_idx] # (H, W, 3)
        
        # 3. 图像维度处理
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        # 4. === 关键优化点 C：直接从 RAM 读取特征，跳过所有计算 ===
        time_series = self.feature_cache[idx]
        
        # 5. 转为 Tensor
        time_series_tensor = torch.from_numpy(time_series) # 已是 float32
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 6. 图像变换
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return time_series_tensor, image_tensor, label_tensor

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
    DATA_DIR = './jacket_damage_data_timespace'
    NUM_CLASSES = 2  # 0=健康, 1=损伤
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5
    
    # ==================== 1. 加载数据 ====================
    print("\n[步骤1] 加载仿真数据...")
    
    if not os.path.exists(DATA_DIR):
        print(f"✗ 错误: 数据目录 {DATA_DIR} 不存在！")
        return
    
    full_dataset = H5LazyDataset(
        data_dir=DATA_DIR,
        feature_selection='mean',
        transform=None  # 暂时不传 transform，后续在 system 里统一加
    )
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(full_dataset.feature_cache)

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
    print("\n[步骤3] 划分数据集 (按Scenario)...")

    # 1. 构建场景索引列表
    # H5LazyDataset 已经有了 file_metadata，我们需要找出每个文件对应的样本范围
    scenario_indices = [] # 存储每个场景的样本索引范围 [(start_idx, end_idx, file_idx), ...]

    current_idx = 0
    # 注意：这里需要根据你的 H5LazyDataset 结构来获取每个文件的样本数
    # 假设你可以从 dataset 获取文件列表和窗口数
    for file_idx, fname in enumerate(full_dataset.h5_files):
        fpath = os.path.join(DATA_DIR, fname)
        with h5py.File(fpath, 'r') as hf:
            n_windows = hf['feature_maps'].shape[0]
            end_idx = current_idx + n_windows
            scenario_indices.append({
                'file_idx': file_idx,
                'start': current_idx,
                'end': end_idx,
                'label': int(hf['damage_class'][0]) # 用于分层抽样
            })
            current_idx = end_idx

    # 2. 对场景进行划分 (而不是对样本)
    n_scenarios = len(scenario_indices)
    scenario_indices_list = np.arange(n_scenarios)

    # 获取每个场景的标签列表用于分层
    scenario_labels = [s['label'] for s in scenario_indices]

    # 划分场景索引
    train_scenario_idx, temp_scenario_idx = train_test_split(
        scenario_indices_list, 
        test_size=0.4, 
        random_state=42,
        stratify=scenario_labels # 保持健康/损伤比例
    )

    val_scenario_idx, test_scenario_idx = train_test_split(
        temp_scenario_idx, 
        test_size=0.5, 
        random_state=42,
        stratify=[scenario_labels[i] for i in temp_scenario_idx]
    )

    # 3. 将场景索引展开回样本索引
    def expand_indices(scenario_idx_list, scenario_indices):
        sample_indices = []
        for s_idx in scenario_idx_list:
            s_info = scenario_indices[s_idx]
            sample_indices.extend(range(s_info['start'], s_info['end']))
        return np.array(sample_indices)

    train_idx = expand_indices(train_scenario_idx, scenario_indices)
    val_idx = expand_indices(val_scenario_idx, scenario_indices)
    test_idx = expand_indices(test_scenario_idx, scenario_indices)

    print(f"  ✓ 训练集场景数: {len(train_scenario_idx)}, 样本数: {len(train_idx)}")
    print(f"  ✓ 验证集场景数: {len(val_scenario_idx)}, 样本数: {len(val_idx)}")
    print(f"  ✓ 测试集场景数: {len(test_scenario_idx)}, 样本数: {len(test_idx)}")

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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
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
    class_names = ['健康', '损伤']
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
