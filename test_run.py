# ============================================================================
# 验证脚本 V2: 集成数据生成器与损伤检测模型
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sys

# 设置matplotlib后端（避免交互模式警告）
import matplotlib
matplotlib.use('Agg')

# 导入必要的模块（确保model.py在当前路径或已安装）
try:
    from model import (
        OffshoreDamageDetectionSystem, 
        MultiModalDamageDetector,
        SignalPreprocessor,
        OffshoreStructureDataset
    )
    print("✓ 成功导入模型模块")
except ImportError as e:
    print(f"✗ 导入model模块失败: {e}")
    print("请确保model.py文件在同一目录下或PYTHONPATH中")
    sys.exit(1)

# ============================================================================

def extract_signal_features(signal: np.ndarray) -> np.ndarray:
    """
    从信号窗口提取16个统计特征，匹配模型 MLP 输入维度
    """
    return np.array([
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        skew(signal),
        kurtosis(signal),
        np.sqrt(np.mean(signal**2)),  # RMS
        np.ptp(signal),  # Peak-to-peak
        np.sum(np.abs(np.diff(signal))) / len(signal),  # 振动烈度
        np.mean(np.abs(signal - np.mean(signal))),  # 平均绝对偏差
        np.var(signal),  # 方差
        np.median(np.abs(signal - np.median(signal))),  # 中位数绝对偏差
        np.sum(signal**2) / len(signal),  # 均方值
        np.sum(np.abs(signal))  # 绝对值和
    ])


class SimpleMultiModalDataset(Dataset):
    """
    简单的多模态数据集类
    """
    def __init__(self, time_series_data, image_data, labels):
        self.time_series = torch.FloatTensor(time_series_data)
        self.images = torch.FloatTensor(image_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.time_series[idx], self.images[idx], self.labels[idx]


def run_model_verification():
    print("\n" + "=" * 80)
    print("开始模型验证流程 V2")
    print("=" * 80)

    # ==================== 1. 加载数据 ====================
    print("\n[步骤 1] 加载数据...")
    try:
        data = np.load('damage_detection_dataset.npz')
        raw_signals = data['acceleration_signals']  # Shape: (N, 5, 3000)
        images = data['images']                      # Shape: (N, 3, 224, 224)
        labels = data['labels']                      # Shape: (N,)
        print(f"  ✓ 成功加载数据: {raw_signals.shape[0]} 个样本")
        print(f"  ✓ 原始信号形状: {raw_signals.shape}")
        print(f"  ✓ 图像数据形状: {images.shape}")
        print(f"  ✓ 标签形状: {labels.shape}")
    except FileNotFoundError:
        print("  ✗ 未找到数据集文件，请先运行数据生成器的主函数。")
        return

    # ==================== 2. 特征工程 ====================
    print("\n[步骤 2] 特征提取与预处理...")
    
    # 提取特征：选择第一个传感器（索引0）的特征作为输入
    # Shape: (N, 16)
    time_series_features = np.array([extract_signal_features(s[0]) for s in raw_signals])
    
    # 归一化特征
    scaler = MinMaxScaler()
    time_series_features = scaler.fit_transform(time_series_features)
    
    print(f"  ✓ 提取特征形状: {time_series_features.shape}")
    print(f"  ✓ 特征范围: [{time_series_features.min():.4f}, {time_series_features.max():.4f}]")
    
    # 检查是否有NaN或Inf
    if np.any(np.isnan(time_series_features)) or np.any(np.isinf(time_series_features)):
        print("  ⚠ 警告: 检测到NaN或Inf值，尝试修复...")
        time_series_features = np.nan_to_num(time_series_features, nan=0.0, posinf=1.0, neginf=0.0)

    # ==================== 3. 初始化模型 ====================
    print("\n[步骤 3] 初始化模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  ✓ 使用设备: {device}")
    
    # 模型参数
    NUM_CLASSES = 4  # 数据生成器生成的是4个类别
    BATCH_SIZE = 8
    
    # 创建模型实例
    model = MultiModalDamageDetector(
        num_classes=NUM_CLASSES,
        mlp_input_dim=16,
        use_pretrained_resnet=False  # 使用False以加快加载速度
    ).to(device)
    
    print(f"  ✓ 模型已创建，类别数: {NUM_CLASSES}")
    print(f"  ✓ 总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 4. 划分数据集 ====================
    print("\n[步骤 4] 划分数据集...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        time_series_features, labels, images, 
        test_size=0.3, 
        random_state=42, 
        stratify=labels
    )
    
    # 划分验证集
    X_train, X_val, y_train, y_val, img_train, img_val = train_test_split(
        X_train, y_train, img_train,
        test_size=0.2,  # 验证集占训练集的20%
        random_state=42,
        stratify=y_train
    )
    
    print(f"  ✓ 训练集: {len(X_train)} 样本")
    print(f"  ✓ 验证集: {len(X_val)} 样本")
    print(f"  ✓ 测试集: {len(X_test)} 样本")
    
    # 创建数据集和数据加载器
    train_dataset = SimpleMultiModalDataset(X_train, img_train, y_train)
    val_dataset = SimpleMultiModalDataset(X_val, img_val, y_val)
    test_dataset = SimpleMultiModalDataset(X_test, img_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  ✓ 数据加载器已创建，批大小: {BATCH_SIZE}")

    # ==================== 5. 训练模型 ====================
    print("\n[步骤 5] 开始训练...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 训练参数
    MAX_EPOCHS = 20
    best_val_loss = float('inf')
    early_stopping_patience = 7
    patience_counter = 0
    
    # 记录历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(MAX_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (time_series, images, batch_labels) in enumerate(train_loader):
            time_series = time_series.to(device)
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(time_series, images)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for time_series, images, batch_labels in val_loader:
                time_series = time_series.to(device)
                images = images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(time_series, images)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] - "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  早停触发于 epoch {epoch+1}")
                break

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_v2.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 训练曲线已保存: training_history_v2.png")

    # ==================== 6. 评估模型 ====================
    print("\n[步骤 6] 评估模型性能...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for time_series, images, batch_labels in test_loader:
            time_series = time_series.to(device)
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(time_series, images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # 计算指标
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"  ✓ 测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=['Healthy', 'Dmg1', 'Dmg2', 'Dmg3'],
                                zero_division=0))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['Healthy', 'Dmg1', 'Dmg2', 'Dmg3'], rotation=45)
    plt.yticks(tick_marks, ['Healthy', 'Dmg1', 'Dmg2', 'Dmg3'])
    
    # 在单元格中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 混淆矩阵已保存: confusion_matrix_v2.png")

    # ==================== 7. GVR 损伤定位演示 ====================
    print("\n[步骤 7] 演示 GVR 损伤定位功能...")
    
    # 创建预处理器
    preprocessor = SignalPreprocessor()
    
    # 获取健康基准（类别0的样本）
    healthy_indices = np.where(labels == 0)[0]
    healthy_samples = raw_signals[healthy_indices[:5]]  # 取前5个健康样本
    healthy_baseline = healthy_samples.mean(axis=0)  # Shape: (5, 3000)
    
    # 获取一个损伤样本（类别1）
    damage_indices = np.where(labels == 1)[0]
    damaged_sample = raw_signals[damage_indices[0]]  # Shape: (5, 3000)
    
    print("  分析损伤样本 (类别1)...")
    
    # 对比每个传感器
    n_sensors = damaged_sample.shape[0]
    damage_probabilities = {}
    
    for i in range(n_sensors):
        # 获取当前传感器信号
        sig_damaged = damaged_sample[i]
        sig_healthy = healthy_baseline[i]
        
        # 计算损伤指数
        di = preprocessor.calculate_damage_index(sig_healthy, sig_damaged, window_size=1000)
        
        # 计算梯度变化率
        gvr = preprocessor.calculate_gvr(di)
        
        # 计算损伤概率
        prob = preprocessor.detect_damage_probability(gvr, threshold=0.5)
        damage_probabilities[f"传感器 {i+1}"] = prob
    
    print("\n  各传感器损伤概率 (基于 GVR 分析):")
    print("  " + "-" * 50)
    for sensor, prob in sorted(damage_probabilities.items(), key=lambda x: x[1], reverse=True):
        status = "[ ⚠️ 损伤疑似 ]" if prob > 10 else "[ ✓ 正常 ]"
        bar = "█" * int(prob // 5) + "░" * (20 - int(prob // 5))
        print(f"  {sensor}: {prob:5.2f}%  {bar}  {status}")
    
    # 可视化GVR分析
    plt.figure(figsize=(12, 4))
    
    for i in range(min(3, n_sensors)):  # 只显示前3个传感器
        plt.subplot(1, 3, i+1)
        sig_damaged = damaged_sample[i]
        sig_healthy = healthy_baseline[i]
        di = preprocessor.calculate_damage_index(sig_healthy, sig_damaged, window_size=1000)
        gvr = preprocessor.calculate_gvr(di)
        
        plt.plot(gvr, label=f'传感器 {i+1}')
        plt.xlabel('窗口')
        plt.ylabel('GVR')
        plt.title(f'传感器 {i+1} 的 GVR 曲线')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('gvr_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ GVR分析图已保存: gvr_analysis.png")

    print("\n" + "=" * 80)
    print("验证流程结束！")
    print("=" * 80)


if __name__ == "__main__":
    run_model_verification()
