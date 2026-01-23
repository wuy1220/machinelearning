"""
海洋导管架平台损伤检测系统
基于MLP-ResNet50多模态深度学习框架
参考论文: "Multimodal deep learning with integrated automatic labeling for structural damage detection"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from scipy import signal, stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 信号预处理模块（Butterworth滤波 + 损伤指数计算）
# ============================================================================

class SignalPreprocessor:
    """
    信号预处理类，实现Butterworth低通滤波和损伤指数计算
    """
    
    def __init__(self, fs: float = 100, cutoff_freq: float = 25, order: int = 4):
        """
        初始化预处理器
        
        参数:
            fs: 采样频率 (Hz)
            cutoff_freq: 截止频率 (Hz)
            order: 滤波器阶数
        """
        self.fs = fs
        self.cutoff_freq = cutoff_freq
        self.order = order
        
        # 设计Butterworth低通滤波器
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
    def butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """
        应用Butterworth低通滤波器
        
        参数:
            data: 输入信号 (n_samples, n_channels)
            
        返回:
            filtered_data: 滤波后的信号
        """
        if data.ndim == 1:
            return signal.filtfilt(self.b, self.a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = signal.filtfilt(self.b, self.a, data[:, i])
            return filtered
    
    def calculate_damage_index(self, healthy_signal: np.ndarray, 
                              damaged_signal: np.ndarray, 
                              window_size: int = 3000) -> np.ndarray:
        """
        计算损伤指数(Damage Index, DI)
        
        参数:
            healthy_signal: 健康状态信号
            damaged_signal: 损伤状态信号
            window_size: 滑动窗口大小
            
        返回:
            damage_index: 损伤指数序列
        """
        n_samples = min(len(healthy_signal), len(damaged_signal))
        n_windows = n_samples // window_size
        
        damage_index = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            
            xh = healthy_signal[start:end]
            xd = damaged_signal[start:end]
            
            # 计算损伤指数 DI_j = sum(xd_ij - xh_ij) / (sum(xh_ij^2) + epsilon)
            epsilon = 1e-8
            di = np.sum(xd - xh) / (np.sum(xh ** 2) + epsilon)
            damage_index[i] = np.abs(di)
            
        return damage_index
    
    def calculate_gvr(self, damage_index: np.ndarray) -> np.ndarray:
        """
        计算梯度变化率(Gradient Variation Rate, GVR)
        
        参数:
            damage_index: 损伤指数序列
            
        返回:
            gvr: 梯度变化率
        """
        # 一阶差分
        di_prime = np.diff(damage_index, n=1)
        di_prime = np.pad(di_prime, (0, 1), mode='edge')
        
        # 二阶梯度（绝对值）
        di_double_prime = np.abs(np.diff(di_prime, n=1))
        di_double_prime = np.pad(di_double_prime, (0, 1), mode='edge')
        
        return di_double_prime
    
    def detect_damage_probability(self, gvr: np.ndarray, 
                                  threshold: float = 0.1) -> float:
        """
        统计损伤发生概率
        
        参数:
            gvr: 梯度变化率序列
            threshold: 检测阈值
            
        返回:
            probability: 损伤发生概率
        """
        fault_occurrences = np.sum(gvr > threshold)
        total_windows = len(gvr)
        probability = (fault_occurrences / total_windows) * 100
        return probability


# ============================================================================
# 2. 数据集类
# ============================================================================

class OffshoreStructureDataset(Dataset):
    """
    海洋结构多模态数据集
    """
    
    def __init__(self, time_series_data: np.ndarray, 
                 image_data: np.ndarray, 
                 labels: np.ndarray,
                 transform=None):
        """
        参数:
            time_series_data: 时间序列数据 (n_samples, n_features)
            image_data: 图像数据 (n_samples, C, H, W)
            labels: 损伤标签 (n_samples,)
            transform: 图像变换
        """
        self.time_series = torch.FloatTensor(time_series_data)
        self.images = torch.FloatTensor(image_data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        time_series = self.time_series[idx]
        image = self.images[idx]
        label = self.labels[idx]
        
        # 应用图像变换
        if self.transform:
            image = self.transform(image)
            
        return time_series, image, label




class MultiModalDamageDetector(nn.Module):
    """
    多模态损伤检测网络
    融合MLP（时间序列）和ResNet50（图像）特征
    """
    
    def __init__(self, num_classes: int, 
                 mlp_input_dim: int = 16,
                 mlp_hidden_dims: List[int] = [64, 32, 16],
                 use_pretrained_resnet: bool = True):
        """
        参数:
            num_classes: 损伤类别数
            mlp_input_dim: MLP输入维度
            mlp_hidden_dims: MLP隐藏层维度列表
            use_pretrained_resnet: 是否使用预训练的ResNet50
        """
        super(MultiModalDamageDetector, self).__init__()
        
        # ==================== MLP分支（时间序列特征提取）====================
        mlp_layers = []
        input_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_output_dim = mlp_hidden_dims[-1]
        
        # ==================== ResNet50分支（图像特征提取）==================
        # 加载预训练的ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 
                                 if use_pretrained_resnet else None)
        
        # 移除最后的分类层，保留特征提取部分
        modules = list(resnet.children())[:-1]  # 保留到全局平均池化层
        self.resnet = nn.Sequential(*modules)
        
        # ResNet50特征降维层
        resnet_feature_dim = 2048
        self.resnet_fc = nn.Sequential(
            nn.Linear(resnet_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.mlp_output_dim)
        )
        
        # ==================== 特征融合与分类层 ====================
        fused_dim = self.mlp_output_dim * 2
        self.fusion_layers = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, time_series: torch.Tensor, 
                images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            time_series: 时间序列输入 (batch_size, mlp_input_dim)
            images: 图像输入 (batch_size, 3, 224, 224)
            
        返回:
            logits: 分类输出 (batch_size, num_classes)
        """
        # MLP分支
        mlp_features = self.mlp(time_series)  # (batch_size, mlp_output_dim)
        
        # ResNet50分支
        resnet_features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)  # (batch_size, 2048)
        resnet_features = self.resnet_fc(resnet_features)  # (batch_size, mlp_output_dim)
        
        # 特征融合（拼接）
        fused_features = torch.cat([mlp_features, resnet_features], dim=1)
        
        # 分类
        logits = self.fusion_layers(fused_features)
        
        return logits


# ============================================================================
# 4. 损伤检测系统（集成GVR分析 + 深度学习）
# ============================================================================

class OffshoreDamageDetectionSystem:
    """
    海洋结构损伤检测系统
    整合损伤定位和分类功能
    """
    
    def __init__(self, num_classes: int = 5, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_classes = num_classes
        self.device = device
        
        # 初始化各组件
        self.preprocessor = SignalPreprocessor()
        self.model = MultiModalDamageDetector(num_classes=num_classes).to(device)
        self.scaler = MinMaxScaler()
        
        # 定义图像变换
        self.image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])

        self.train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.test_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    

    def load_simulation_data(self, 
                            signals: np.ndarray, 
                            images: np.ndarray, 
                            labels: np.ndarray,
                            feature_selection: str = 'first_sensor',
                            sensor_indices: List[int] = None,
                            normalize: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载并处理仿真生成器生成的数据
        
        参数:
            signals: 仿真信号数据 (n_samples, n_dof, n_timepoints)
            images: 图像数据 (n_samples, 3, 224, 224)
            labels: 损伤标签 (n_samples,)
            feature_selection: 特征选择策略
                - 'first_sensor': 使用第一个传感器
                - 'mean': 使用所有传感器的平均
                - 'concatenate': 拼接多个传感器的特征
                - 'specified': 使用指定的传感器索引
            sensor_indices: 指定的传感器索引列表（仅当feature_selection='specified'时使用）
            normalize: 是否对特征进行归一化
        
        返回:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        print("\n" + "=" * 60)
        print("加载仿真生成器数据")
        print("=" * 60)
        
        n_samples, n_dof, n_timepoints = signals.shape
        print(f"原始信号形状: {signals.shape}")
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        
        # ==================== 1. 特征提取 ====================
        print("\n[步骤1] 提取时间序列特征...")
        print(f"  特征选择策略: {feature_selection}")
        
        # 根据策略提取特征
        if feature_selection == 'first_sensor':
            # 使用第一个传感器
            time_series_features = np.array([
                self._extract_statistical_features(signals[i, 0, :]) 
                for i in range(n_samples)
            ])
            print(f"  ✓ 使用第一个传感器")
            
        elif feature_selection == 'mean':
            # 使用所有传感器的平均
            averaged_signals = signals.mean(axis=1)  # (n_samples, n_timepoints)
            time_series_features = np.array([
                self._extract_statistical_features(averaged_signals[i, :]) 
                for i in range(n_samples)
            ])
            print(f"  ✓ 使用所有传感器平均")
            
        elif feature_selection == 'concatenate':
            # 拼接多个传感器的特征（最多3个，避免维度过高）
            n_sensors_to_use = min(3, n_dof)
            features_per_sensor = 16
            time_series_features = np.zeros((n_samples, n_sensors_to_use * features_per_sensor))
            
            for i in range(n_samples):
                for j in range(n_sensors_to_use):
                    sensor_idx = j
                    feat = self._extract_statistical_features(signals[i, sensor_idx, :])
                    time_series_features[i, j*features_per_sensor:(j+1)*features_per_sensor] = feat
            
            print(f"  ✓ 拼接 {n_sensors_to_use} 个传感器的特征 (共 {n_sensors_to_use*features_per_sensor} 维)")
            
        elif feature_selection == 'specified':
            # 使用指定的传感器
            if sensor_indices is None:
                sensor_indices = [0, n_dof//2, n_dof-1]  # 默认：首、中、尾
                print(f"  ⚠ 未指定传感器索引，使用默认: {sensor_indices}")
            
            features_per_sensor = 16
            time_series_features = np.zeros((n_samples, len(sensor_indices) * features_per_sensor))
            
            for i in range(n_samples):
                for j, sensor_idx in enumerate(sensor_indices):
                    feat = self._extract_statistical_features(signals[i, sensor_idx, :])
                    time_series_features[i, j*features_per_sensor:(j+1)*features_per_sensor] = feat
            
            print(f"  ✓ 使用指定传感器 {sensor_indices}")
        
        else:
            raise ValueError(f"未知的特征选择策略: {feature_selection}")
        
        # 检查NaN和Inf
        if np.any(np.isnan(time_series_features)) or np.any(np.isinf(time_series_features)):
            print("  ⚠ 检测到NaN或Inf值，正在修复...")
            time_series_features = np.nan_to_num(time_series_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"  ✓ 特征形状: {time_series_features.shape}")
        
        # ==================== 2. 特征归一化 ====================
        if normalize:
            print("\n[步骤2] 归一化特征...")
            if not hasattr(self, 'scaler'):
                self.scaler = MinMaxScaler()
            
            time_series_features = self.scaler.fit_transform(time_series_features)
            print(f"  ✓ 特征范围: [{time_series_features.min():.4f}, {time_series_features.max():.4f}]")
        
        # ==================== 3. 图像归一化 ====================
        print("\n[步骤3] 处理图像数据...")
        # 确保图像在 [0, 1] 范围内（用于后续的ImageNet标准化）
        images = np.clip(images, 0, 1)
        print(f"  ✓ 图像范围: [{images.min():.4f}, {images.max():.4f}]")
        
        # ==================== 4. 划分数据集 ====================
        print("\n[步骤4] 划分训练/验证/测试集...")
        
        # 使用分层抽样确保标签分布均匀
        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
        
        # 第一次划分：训练集 vs 临时集
        X_train, X_temp, y_train, y_temp, img_train, img_temp = train_test_split(
            time_series_features, labels, images,
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=labels
        )
        
        # 第二次划分：验证集 vs 测试集
        X_val, X_test, y_val, y_test, img_val, img_test = train_test_split(
            X_temp, y_temp, img_temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42,
            stratify=y_temp
        )
        
        print(f"  ✓ 训练集: {len(X_train)} 样本 (标签分布: {np.bincount(y_train)})")
        print(f"  ✓ 验证集: {len(X_val)} 样本 (标签分布: {np.bincount(y_val)})")
        print(f"  ✓ 测试集: {len(X_test)} 样本 (标签分布: {np.bincount(y_test)})")
        
        # ==================== 5. 创建数据加载器 ====================
        print("\n[步骤5] 创建数据加载器...")
        
        # 更新模型的MLP输入维度（如果需要）
        mlp_input_dim = X_train.shape[1]
        if self.model.mlp[0].in_features != mlp_input_dim:
            print(f"  ⚠ 调整MLP输入维度: {self.model.mlp[0].in_features} -> {mlp_input_dim}")
            self.model.mlp[0] = nn.Linear(mlp_input_dim, self.model.mlp[0].out_features)
            self.model.to(self.device)
        
        # 创建数据集
        train_dataset = OffshoreStructureDataset(
            X_train, img_train, y_train,
            transform=self.train_transform
        )
        val_dataset = OffshoreStructureDataset(
            X_val, img_val, y_val,
            transform=self.test_transform
        )
        test_dataset = OffshoreStructureDataset(
            X_test, img_test, y_test,
            transform=self.test_transform
        )
        
        # 创建数据加载器
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  ✓ 批大小: {batch_size}")
        print(f"  ✓ 训练批数: {len(train_loader)}")
        print(f"  ✓ 验证批数: {len(val_loader)}")
        print(f"  ✓ 测试批数: {len(test_loader)}")
        
        print("\n" + "=" * 60)
        print("✓ 数据加载完成！")
        print("=" * 60)
        
        del self.raw_signals 
        del self.raw_images
        import gc
        gc.collect()
        return train_loader, val_loader, test_loader



        
    def prepare_training_data(self, 
                             acceleration_signals: Dict[int, np.ndarray],
                             damage_labels: Dict[int, int],
                             healthy_baseline: np.ndarray,
                             window_size: int = 3000,
                             step_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据：自动标注 + 数据分割
        
        参数:
            acceleration_signals: 各测点的加速度信号 {测点ID: 信号数组}
            damage_labels: 损伤标签 {测点ID: 损伤类别}
            healthy_baseline: 健康状态基准信号
            window_size: 滑动窗口大小
            step_size: 滑动步长
            
        返回:
            time_series_data: 时间序列特征
            image_features: 图像特征（可扩展）
            labels: 损伤标签
        """
        time_series_samples = []
        labels_list = []
        
        # 对每个测点的信号进行处理
        for sensor_id, signal_data in acceleration_signals.items():
            # 滤波
            filtered_signal = self.preprocessor.butterworth_filter(signal_data)
            
            # 计算损伤指数和GVR
            di = self.preprocessor.calculate_damage_index(healthy_baseline, filtered_signal, window_size)
            gvr = self.preprocessor.calculate_gvr(di)
            
            # 滑动窗口分割
            n_samples = len(filtered_signal)
            for i in range(0, n_samples - window_size, step_size):
                window_data = filtered_signal[i:i+window_size]
                
                # 提取统计特征作为MLP输入
                features = self._extract_statistical_features(window_data)
                
                # 根据测点ID分配标签
                label = damage_labels.get(sensor_id, 0)  # 0表示健康
                labels_list.append(label)
                time_series_samples.append(features)
                
        # 转换为numpy数组
        time_series_data = np.array(time_series_samples)
        time_series_data = self.scaler.fit_transform(time_series_data)
        
        # 生成模拟图像特征（实际应用中应使用真实结构图像）
        image_features = self._generate_simulated_image_features(len(time_series_data))
        
        labels = np.array(labels_list)
        
        return time_series_data, image_features, labels
    
    def _extract_statistical_features(self, signal_window: np.ndarray) -> np.ndarray:
        """
        从信号窗口提取统计特征
        
        参数:
            signal_window: 信号窗口
            
        返回:
            features: 特征向量
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
    
    def _generate_simulated_image_features(self, n_samples: int) -> np.ndarray:
        """
        生成模拟图像特征（占位符，实际应用中应使用真实结构图像）
        """
        # 模拟生成224x224x3的RGB图像
        image_features = np.random.rand(n_samples, 3, 224, 224)
        return image_features
    
    def train(self, train_loader: DataLoader, 
              val_loader: DataLoader,
              epochs: int = 50,
              learning_rate: float = 0.001,
              early_stopping_patience: int = 10) -> Dict[str, List]:
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
            
        返回:
            history: 训练历史 {loss, accuracy, val_loss, val_accuracy}
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (time_series, images, labels) in enumerate(train_loader):
                time_series = time_series.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(time_series, images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for time_series, images, labels in val_loader:
                    time_series = time_series.to(self.device)
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(time_series, images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 计算指标
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 记录历史
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_damage_detector.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            # 打印训练进度
            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            test_loader: 测试数据加载器
            
        返回:
            metrics: 性能指标字典
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for time_series, images, labels in test_loader:
                time_series = time_series.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(time_series, images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算混淆矩阵和分类报告
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                      output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics
    
    def detect_damage_location(self, 
                              current_signals: Dict[int, np.ndarray],
                              healthy_baseline: Dict[int, np.ndarray],
                              threshold: float = 0.1) -> Dict[int, float]:
        """
        基于GVR分析进行损伤定位
        
        参数:
            current_signals: 当前信号 {测点ID: 信号}
            healthy_baseline: 健康基准信号
            threshold: 检测阈值
            
        返回:
            damage_probabilities: 各测点的损伤概率 {测点ID: 概率}
        """
        damage_probabilities = {}
        
        for sensor_id, signal_data in current_signals.items():
            if sensor_id not in healthy_baseline:
                continue
                
            # 滤波
            filtered_signal = self.preprocessor.butterworth_filter(signal_data)
            filtered_baseline = self.preprocessor.butterworth_filter(healthy_baseline[sensor_id])
            
            # 计算GVR和损伤概率
            di = self.preprocessor.calculate_damage_index(
                filtered_baseline, filtered_signal, window_size=3000
            )
            gvr = self.preprocessor.calculate_gvr(di)
            probability = self.preprocessor.detect_damage_probability(gvr, threshold)
            
            damage_probabilities[sensor_id] = probability
        
        return damage_probabilities
    
    def predict_damage_class(self, 
                             time_series_features: np.ndarray,
                             image_features: np.ndarray) -> Tuple[int, float]:
        """
        预测损伤类别
        
        参数:
            time_series_features: 时间序列特征
            image_features: 图像特征
            
        返回:
            predicted_class: 预测类别
            confidence: 置信度
        """
        self.model.eval()
        
        # 数据预处理
        time_series_features = self.scaler.transform(time_series_features)
        
        with torch.no_grad():
            time_series_tensor = torch.FloatTensor(time_series_features).unsqueeze(0).to(self.device)
            image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)
            
            outputs = self.model(time_series_tensor, image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        return predicted_class.item(), confidence.item()
    
    def plot_training_history(self, history: Dict[str, List], save_path: str = None):
        """
        绘制训练历史曲线
        
        参数:
            history: 训练历史
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            class_names: List[str] = None,
                            save_path: str = None):
        """
        绘制混淆矩阵
        
        参数:
            cm: 混淆矩阵
            class_names: 类别名称
            save_path: 保存路径（可选）
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在单元格中显示数值
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# 5. 示例使用代码
# ============================================================================

def generate_simulated_data(n_samples: int = 1000, 
                           n_sensors: int = 5,
                           n_classes: int = 5,
                           duration: float = 30.0,
                           fs: float = 1000) -> Tuple[Dict, np.ndarray]:
    """
    生成模拟的海洋导管架平台监测数据
    
    参数:
        n_samples: 样本数
        n_sensors: 传感器数量
        n_classes: 损伤类别数
        duration: 信号持续时间（秒）
        fs: 采样频率
        
    返回:
        signals: 传感器信号字典
        labels: 损伤标签数组
    """
    signals = {}
    labels = np.random.randint(0, n_classes, n_samples)
    
    t = np.linspace(0, duration, int(duration * fs))
    
    for i in range(n_sensors):
        # 生成基础振动信号（多频率分量）
        signal = (0.5 * np.sin(2 * np.pi * 2 * t) +  # 2Hz分量
                 0.3 * np.sin(2 * np.pi * 5 * t) +   # 5Hz分量
                 0.2 * np.sin(2 * np.pi * 10 * t))   # 10Hz分量
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, len(t))
        
        # 根据损伤类别添加特征
        for sample_idx in range(n_samples):
            label = labels[sample_idx]
            if label > 0:  # 有损伤
                # 损伤导致的频率变化和幅值变化
                damage_factor = 0.1 * label
                signal[sample_idx*int(fs):(sample_idx+1)*int(fs)] *= (1 + damage_factor * np.sin(2 * np.pi * 3 * t[sample_idx*int(fs):(sample_idx+1)*int(fs)]))
        
        signals[i] = signal + noise
    
    return signals, labels


def main():
    """
    主函数：演示完整的损伤检测流程
    """
    print("=" * 70)
    print("海洋导管架平台损伤检测系统")
    print("基于MLP-ResNet50多模态深度学习框架")
    print("=" * 70)
    
    # ==================== 参数设置 ====================
    NUM_CLASSES = 5  # 损伤类别数（0=健康，1-4=不同损伤类型）
    NUM_SENSORS = 8  # 传感器数量
    BATCH_SIZE = 16
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # ==================== 生成模拟数据 ====================
    print("\n[1] 生成模拟监测数据...")
    signals, raw_labels = generate_simulated_data(
        n_samples=2000,
        n_sensors=NUM_SENSORS,
        n_classes=NUM_CLASSES
    )
    print(f"  ✓ 生成 {NUM_SENSORS} 个传感器信号，总时长 {len(signals[0])/1000:.1f} 秒")
    print(f"  ✓ 包含 {NUM_CLASSES} 个损伤类别")
    
    # 创建健康基准信号（所有样本的平均）
    healthy_baseline = np.mean([signals[i] for i in range(NUM_SENSORS)], axis=0)
    
    # 分配标签到传感器
    sensor_labels = {}
    for sensor_id in range(NUM_SENSORS):
        sensor_labels[sensor_id] = raw_labels[sensor_id % NUM_CLASSES]
    
    # ==================== 初始化检测系统 ====================
    print("\n[2] 初始化损伤检测系统...")
    detection_system = OffshoreDamageDetectionSystem(num_classes=NUM_CLASSES)
    print(f"  ✓ 设备: {detection_system.device}")
    print(f"  ✓ 模型参数量: {sum(p.numel() for p in detection_system.model.parameters()):,}")
    
    # ==================== 准备训练数据 ====================
    print("\n[3] 准备训练数据（GVR分析 + 自动标注）...")
    time_series_data, image_features, labels = detection_system.prepare_training_data(
        acceleration_signals=signals,
        damage_labels=sensor_labels,
        healthy_baseline=healthy_baseline,
        window_size=3000,
        step_size=100
    )
    print(f"  ✓ 时间序列样本: {time_series_data.shape}")
    print(f"  ✓ 图像样本: {image_features.shape}")
    print(f"  ✓ 标签分布: {np.bincount(labels)}")
    
    # ==================== 划分数据集 ====================
    print("\n[4] 划分训练/验证/测试集...")
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * TRAIN_RATIO)
    val_end = train_end + int(n_samples * VAL_RATIO)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 创建数据集
    train_dataset = OffshoreStructureDataset(
        time_series_data[train_indices],
        image_features[train_indices],
        labels[train_indices],
        transform=detection_system.image_transform
    )
    
    val_dataset = OffshoreStructureDataset(
        time_series_data[val_indices],
        image_features[val_indices],
        labels[val_indices],
        transform=detection_system.image_transform
    )
    
    test_dataset = OffshoreStructureDataset(
        time_series_data[test_indices],
        image_features[test_indices],
        labels[test_indices],
        transform=detection_system.image_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  ✓ 训练集: {len(train_dataset)} 样本")
    print(f"  ✓ 验证集: {len(val_dataset)} 样本")
    print(f"  ✓ 测试集: {len(test_dataset)} 样本")
    
    # ==================== 训练模型 ====================
    print("\n[5] 训练多模态损伤检测模型...")
    history = detection_system.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    # 绘制训练曲线
    detection_system.plot_training_history(history)
    
    # ==================== 评估模型 ====================
    print("\n[6] 评估模型性能...")
    metrics = detection_system.evaluate(test_loader)
    
    print(f"\n{'='*50}")
    print("测试集性能指标:")
    print(f"{'='*50}")
    print(f"准确率 (Accuracy):     {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision):    {metrics['precision_macro']:.4f}")
    print(f"召回率 (Recall):       {metrics['recall_macro']:.4f}")
    print(f"F1分数 (F1-Score):     {metrics['f1_macro']:.4f}")
    
    # 绘制混淆矩阵
    class_names = [f'Healthy' if i == 0 else f'Damage Type {i}' for i in range(NUM_CLASSES)]
    detection_system.plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # ==================== 损伤定位演示 ====================
    print("\n[7] 演示GVR损伤定位功能...")
    
    # 生成当前监测信号（模拟损伤发生在传感器3和5）
    current_signals = {}
    healthy_baselines = {}
    
    for sensor_id in range(NUM_SENSORS):
        signal = signals[sensor_id].copy()
        healthy_baselines[sensor_id] = signal.copy()
        
        # 在特定传感器添加损伤
        if sensor_id in [2, 4]:  # 传感器3和5（从0开始计数）
            # 添加损伤特征
            damage_idx = len(signal) // 2
            signal[damage_idx:damage_idx+500] *= 1.5  # 局部放大
        
        current_signals[sensor_id] = signal
    
    # 执行损伤定位
    damage_probabilities = detection_system.detect_damage_location(
        current_signals=current_signals,
        healthy_baseline=healthy_baselines,
        threshold=0.05
    )
    
    print("\n各传感器损伤概率:")
    print("-" * 40)
    for sensor_id, probability in sorted(damage_probabilities.items(), 
                                        key=lambda x: x[1], reverse=True):
        status = "⚠️ 损伤" if probability > 10 else "✓ 正常"
        print(f"传感器 {sensor_id+1:2d}: {probability:6.2f}%  {status}")
    
    # ==================== 损伤分类演示 ====================
    print("\n[8] 演示损伤分类功能...")
    
    # 从测试集中选取一个样本进行预测
    test_time_series, test_image, test_label = test_dataset[0]
    
    predicted_class, confidence = detection_system.predict_damage_class(
        time_series_features=test_time_series.numpy(),
        image_features=test_image.numpy()
    )
    
    print(f"\n真实类别: {class_names[test_label]}")
    print(f"预测类别: {class_names[predicted_class]}")
    print(f"置信度: {confidence*100:.2f}%")
    
    # ==================== 保存模型 ====================
    print("\n[9] 保存训练好的模型...")
    torch.save(detection_system.model.state_dict(), 'offshore_damage_detector.pth')
    print("  ✓ 模型已保存到: offshore_damage_detector.pth")
    
    print("\n" + "=" * 70)
    print("损伤检测系统演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
