import numpy as np
import torch
import torch.nn as nn
#from model1 import OffshoreDamageDetectionSystem
from model1_mn_cnn import OffshoreDamageDetectionSystem
import h5py
import matplotlib.pyplot as plt
import os
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


class H5LazyDataset(Dataset):
    def __init__(self, data_dir, window_length=2000, transform=None, is_training=True):
        self.data_dir = data_dir
        self.window_length = window_length 
        self.transform = transform
        self.is_training = is_training
        
        # 1. 扫描目录中的分片文件
        self.h5_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith('data_shard_') and f.endswith('.h5')],
            key=lambda x: int(x.replace('data_shard_', '').replace('.h5', ''))
        )
        
        # 2. 构建索引：遍历所有分片文件 -> 遍历所有Group -> 记录所有Window
        self.sample_metadata = [] 
        total_samples = 0
        
        print(f"[Optimization] 发现 {len(self.h5_files)} 个数据分片，正在构建索引...")
        
        for fname in self.h5_files:
            fpath = os.path.join(data_dir, fname)
            with h5py.File(fpath, 'r') as hf:
                # 遍历该文件内的所有组 (每个组代表一个场景)
                group_names = sorted(hf.keys())
                
                for group_name in group_names:
                    grp = hf[group_name]
                    n_windows = grp['feature_maps'].shape[0]
                    win_len_file = grp.attrs.get('window_length', 2000)
                    step_size = grp.attrs.get('step_size', 50)
                    label = int(grp['damage_class'][0])
                    
                    # 记录每个窗口的元数据
                    for i in range(n_windows):
                        self.sample_metadata.append({
                            'shard_name': fname,       # 新增：分片文件名
                            'group_name': group_name,   # 新增：场景组名
                            'window_idx': i,
                            'window_length': win_len_file,
                            'step_size': step_size,
                            'label': label
                        })
                        total_samples += 1
        
        self.total_samples = total_samples
        print(f"[Optimization] 索引构建完成，总样本数: {total_samples}")

    def __len__(self):
        return self.total_samples

    def _extract_statistical_features(self, signal_window):
        """
        提取统计特征（修改版，返回归一化后的特征）
        """
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
            np.sum(np.abs(np.diff(signal_window))) / len(signal_window),  # 振动烈度
            np.mean(np.abs(signal_window - np.mean(signal_window))),  # 平均绝对偏差
            np.var(signal_window),  # 方差
            np.median(np.abs(signal_window - np.median(signal_window))),  # 中位数绝对偏差
            np.sum(signal_window**2) / len(signal_window),  # 均方值
            np.sum(np.abs(signal_window)),  # 绝对值和
        ]
        return np.array(features)

    def _extract_fft_features(self, signal_windowed, fft_dim=200):
            """
            提取FFT特征（归一化后）
            """
            # 计算 FFT
            fft_vals = np.fft.rfft(signal_windowed)
            fft_abs = np.abs(fft_vals)
            
            # 截取低频部分
            fft_features = fft_abs[:fft_dim]
            
            # Log 变换
            fft_features = np.log1p(fft_features)
            
            # 归一化
            fft_mean = np.mean(fft_features)
            fft_std = np.std(fft_features) + 1e-8
            fft_features = (fft_features - fft_mean) / fft_std
            
            return fft_features

    def __getitem__(self, idx):
        meta = self.sample_metadata[idx]
        fpath = os.path.join(self.data_dir, meta['shard_name'])
        
        with h5py.File(fpath, 'r') as hf:
            # 进入特定的组
            grp = hf[meta['group_name']]
            
            label = meta['label']
            image = grp['feature_maps'][meta['window_idx']] 
            
            # === 读取原始信号 ===
            start_idx = meta['window_idx'] * meta['step_size']
            end_idx = start_idx + self.window_length
            raw_acc = grp['acceleration'][start_idx:end_idx, :]
            
            # 边界处理
            if raw_acc.shape[0] < self.window_length:
                pad_width = self.window_length - raw_acc.shape[0]
                raw_acc = np.pad(raw_acc, ((0, pad_width), (0, 0)), mode='constant')
            
            # 1. 对所有传感器求平均 (1D 信号)
            time_series = np.mean(raw_acc, axis=1)
            
            # 2. 加 Hanning 窗
            window = np.hanning(len(time_series))
            time_series_windowed = time_series * window
            
            # ==================== 新增：统计特征提取 ====================
            stats_features = self._extract_statistical_features(time_series)
            
            # 归一化统计特征
            stats_mean = np.mean(stats_features)
            stats_std = np.std(stats_features) + 1e-8
            stats_features = (stats_features - stats_mean) / stats_std
            
            # ==================== FFT特征提取 ====================
            fft_dim = 200  # FFT维度
            fft_features = self._extract_fft_features(time_series_windowed, fft_dim=fft_dim)
            
            # ==================== 拼接特征 ====================
            # 最终特征 = FFT特征(200维) + 统计特征(16维) = 216维
            time_series = np.concatenate([fft_features, stats_features])  

            # === 时间序列数据增强 ===
            if self.is_training:
                noise_std = 0.01
                noise = np.random.normal(0, noise_std, time_series.shape)
                time_series = time_series + noise
                
                scale = np.random.uniform(0.95, 1.05)
                time_series = time_series * scale

                if np.random.random() < 0.05:  # 5% 概率
                    mask = np.random.random(time_series.shape) > 0.05
                    time_series = time_series * mask

        # 图像维度处理
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        # 转为 Tensor
        time_series_tensor = torch.from_numpy(time_series).float()
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 图像变换
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return time_series_tensor, image_tensor, label_tensor


def run_ablation_study(model, test_loader, device, class_names):
    """
    执行消融实验，评估各模态的独立贡献
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 初始化统计字典
    results = {
        'Full Model': {'correct': 0, 'total': 0, 'loss': 0.0},
        'Image Only': {'correct': 0, 'total': 0, 'loss': 0.0},
        'Time Series Only': {'correct': 0, 'total': 0, 'loss': 0.0}
    }
    
    print("\n" + "="*60)
    print("开始进行消融实验")
    print("="*60)
    
    with torch.no_grad():
        for time_series, images, labels in test_loader:
            time_series = time_series.to(device)
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            
            # 1. Full Model (正常推理)
            outputs_full = model(time_series, images)
            loss_full = criterion(outputs_full, labels)
            _, predicted_full = torch.max(outputs_full.data, 1)
            results['Full Model']['correct'] += (predicted_full == labels).sum().item()
            results['Full Model']['total'] += batch_size
            results['Full Model']['loss'] += loss_full.item() * batch_size
            
            # 2. Image Only (将时序输入置零)
            zero_time_series = torch.zeros_like(time_series)
            outputs_img = model(zero_time_series, images)
            loss_img = criterion(outputs_img, labels)
            _, predicted_img = torch.max(outputs_img.data, 1)
            results['Image Only']['correct'] += (predicted_img == labels).sum().item()
            results['Image Only']['total'] += batch_size
            results['Image Only']['loss'] += loss_img.item() * batch_size
            
            # 3. Time Series Only (将图像输入置零)
            # 注意：由于图像已归一化，置零代表“中性”输入（相当于原始均值）
            # 对于 MobileNet 这种归一化过的模型，零向量通常作为“空”输入是可以接受的
            zero_images = torch.zeros_like(images)
            outputs_time = model(time_series, zero_images)
            loss_time = criterion(outputs_time, labels)
            _, predicted_time = torch.max(outputs_time.data, 1)
            results['Time Series Only']['correct'] += (predicted_time == labels).sum().item()
            results['Time Series Only']['total'] += batch_size
            results['Time Series Only']['loss'] += loss_time.item() * batch_size

    # 计算并打印结果
    print(f"{'模型配置':<20s} | {'Loss':<10s} | {'Accuracy':<10s} | {'贡献度':<10s}")
    print("-" * 60)
    
    full_acc = 0
    full_loss = 0
    
    for name, res in results.items():
        acc = 100 * res['correct'] / res['total']
        loss = res['loss'] / res['total']
        
        if name == 'Full Model':
            full_acc = acc
            full_loss = loss
            marker = "(Baseline)"
        else:
            diff = acc - full_acc
            marker = f"({diff:+.2f}%)"
            
        print(f"{name:<20s} | {loss:<10.4f} | {acc:<10.2f}% | {marker}")
    
    print("-" * 60)
    
    # 简单的结论分析
    img_only_acc = 100 * results['Image Only']['correct'] / results['Image Only']['total']
    time_only_acc = 100 * results['Time Series Only']['correct'] / results['Time Series Only']['total']
    
    if full_acc > img_only_acc and full_acc > time_only_acc:
        print("✓ 结论: 模型有效融合了两个模态的信息，融合效果优于单模态。")
    elif abs(full_acc - img_only_acc) < 1.0:
        print("⚠ 结论: 模型性能主要依赖于图像分支，时序特征贡献较小。")
    elif abs(full_acc - time_only_acc) < 1.0:
        print("⚠ 结论: 模型性能主要依赖于时序分支，图像特征贡献较小。")
    else:
        print("ℹ 结论: 多模态融合带来了提升，但其中一个模态起主导作用。")

    print("="*60)

def main_with_new_simulator():
    """
    使用新仿真器生成的HDF5数据训练损伤检测模型
    """
    print("=" * 70)
    print("使用新仿真器 (HDF5) 数据训练损伤检测模型")
    print("=" * 70)
    
    # ==================== 参数配置 ====================
    DATA_DIR = './jacket_damage_data_timespace2'
    NUM_CLASSES = 2  
    BATCH_SIZE = 32  # 采用 lamb 优化器时，batch_size 应当设置较大的值, 但大 batch 会导致泛化能力下降
    EPOCHS = 30
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 6
    
    # ==================== 1. 加载数据 ====================
    print("\n[步骤1] 加载仿真数据...")
    
    if not os.path.exists(DATA_DIR):
        print(f"✗ 错误: 数据目录 {DATA_DIR} 不存在！")
        return
    
    full_dataset = H5LazyDataset(
        data_dir=DATA_DIR,
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

    # ==================== 3. 划分数据集 (适配分片结构) ====================
    print("\n[步骤3] 划分数据集 (适配分片结构)...")

    scenario_indices = [] # 存储每个场景的样本索引范围 [(start_idx, end_idx, ...), ...]

    current_idx = 0
    
    # 遍历所有分片文件
    for file_idx, fname in enumerate(full_dataset.h5_files):
        fpath = os.path.join(DATA_DIR, fname)
        with h5py.File(fpath, 'r') as hf:
            # 【关键修改】：遍历该分片文件内的所有组（每个组代表一个场景）
            # 新结构中，feature_maps 不在根目录，而在 scenario_xxxx 组下
            group_names = sorted(hf.keys())
            
            for group_name in group_names:
                # 打开具体的场景组
                grp = hf[group_name]
                
                # 读取该场景包含的窗口数量
                n_windows = grp['feature_maps'].shape[0]
                end_idx = current_idx + n_windows
                label = int(grp['damage_class'][0]) # 用于分层抽样
                
                scenario_indices.append({
                    'file_idx': file_idx,
                    'start': current_idx,
                    'end': end_idx,
                    'label': label,
                    'group_name': group_name # 记录组名方便调试
                })
                
                # 更新下一个场景的起始索引
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
    train_dataset_base = H5LazyDataset(
        data_dir=DATA_DIR,
        transform=detection_system.train_transform,
        is_training=True  # === 关键：训练模式，应用数据增强 ===
    )
    train_dataset = Subset(train_dataset_base, train_idx)
    # 验证集和测试集需要重新创建一个 dataset 实例或修改 transform 属性，
    # 因为原始 dataset 的 transform 可能是 train_transform
    # 这里为了简单，我们克隆一个 dataset 用于 val/test
    val_dataset_base = H5LazyDataset(DATA_DIR, transform=detection_system.valid_transform, is_training=False)
    test_dataset_base = H5LazyDataset(DATA_DIR, transform=detection_system.valid_transform, is_training=False)
    
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
    
    # === [新增] 加载在验证集上表现最好的模型 ===
    best_model_path = 'best_damage_detector.pth'
    if os.path.exists(best_model_path):
        print(f"  正在加载最佳模型: {best_model_path}")
        detection_system.model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"  ⚠ 警告: 未找到 {best_model_path}，将使用训练结束时的模型（可能过拟合）")

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
    # ==================== 新增：消融实验 ====================
    # 在常规评估之后执行
    run_ablation_study(
        model=detection_system.model, 
        test_loader=test_loader, 
        device=detection_system.device,
        class_names=class_names
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
