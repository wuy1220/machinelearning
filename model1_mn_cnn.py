"""
海洋导管架平台损伤检测系统
基于多模态深度学习框架(mobilenet v3 + 1d cnn) + 门控机制 + 异步学习率
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
from pytorch_lamb import Lamb
import warnings
import random
warnings.filterwarnings('ignore')


class SmallRandomTranslation(object):
    """
    对 X 轴（传感器空间轴）做轻微随机平移，模拟传感器位置扰动
    Y 轴保持不变，避免破坏时间顺序
    """
    def __init__(self, max_dx=3):
        self.max_dx = max_dx  # 像素

    def __call__(self, img):
        # img: (C, H, W) Tensor
        dx = random.randint(-self.max_dx, self.max_dx)
        if dx == 0:
            return img
        # 沿 W 维度滚动，实现平移
        return torch.roll(img, shifts=dx, dims=-1)


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
    融合时间序列和图像特征
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
        
        # ==================== 分支 1：1D-CNN (时序特征提取) ====================
        # 输入形状: (batch_size, 1, window_length)
        self.ts_output_dim = 16
        self.ts_cnn = nn.Sequential(
            # 第一层卷积：提取局部波形特征
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 降维
            
            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第三层卷积
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 自适应池化层：无论输入长度如何，强制输出形状为 (Batch, 32, 1)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            # 最终投影：将 32 维特征压缩为 16 维，以便与图像分支融合
            nn.Linear(32, self.ts_output_dim)
        )
        

        
        # ==================== MobileNetV3 分支（图像特征提取）==================
        # 加载预训练的 MobileNetV3 Large (Large版本精度稍好，Small版本更快)
        self.img_output_dim = 16
        mobilenet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 
            if use_pretrained_resnet else None
        )

        # MobileNet 的特征提取部分是 .features，输出形状为 (Batch, 960, 7, 7)
        # 为了和 ResNet 保持一致（输出 (Batch, 960, 1, 1)），我们需要手动添加一个全局平均池化层
        self.resnet = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1))  # 将 (B, 960, 7, 7) 压缩为 (B, 960, 1, 1)
        )

        # 更新特征维度：MobileNetV3 Large 的输出通道数是 960
        resnet_feature_dim = 960 

        # 特征降维层 (根据新维度调整)
        self.resnet_fc = nn.Sequential(
            nn.Linear(resnet_feature_dim, 256),  # 可以适当减小中间层维度，例如从512降到256，进一步加速
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),  
            nn.Linear(256, self.img_output_dim)  # 输出维度与1D-CNN保持一致
        )
          
        # ==================== 特征融合与分类层 ====================
        fused_dim = self.ts_output_dim + self.img_output_dim
        self.fusion_bn = nn.BatchNorm1d(fused_dim)

        # 门控网络：学习一个 [0, 1] 之间的权重向量
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), # 输出维度与输入一致，进行逐元素加权
            nn.Sigmoid()           # 激活函数限制在 0-1
        )


        self.classifier = nn.Sequential(  # 参考论文使用小的全连接层
            nn.Linear(fused_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes) # 你的任务是2分类，论文中是8分类，需根据任务调整
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
        # 1. 1D-CNN 分支处理
        # 1D-CNN 需要 (Batch, Channels, Length) 格式，原始信号只有一维，所以需要增加一个维度
        time_series_input = time_series.unsqueeze(1) 
        ts_features = self.ts_cnn(time_series_input)  # (batch_size, 16)
        
        # ResNet50分支
        resnet_features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)  # (batch_size, 2048)
        img_features = self.resnet_fc(resnet_features)  # (batch_size, mlp_output_dim)
        
        concatenated_features = torch.cat([ts_features, img_features], dim=1)
        concatenated_features = self.fusion_bn(concatenated_features)
        # 4. 门控机制
        # 计算门控权重 g
        gate_weights = self.gate(concatenated_features) 
        # 应用权重： element-wise 相乘
        gated_features = concatenated_features * gate_weights 
        
        # 5. 分类
        logits = self.classifier(gated_features)
        
        
        return logits


# ============================================================================
# 4. 损伤检测系统（集成GVR分析 + 深度学习）
# ============================================================================

class OffshoreDamageDetectionSystem:
    """
    海洋结构损伤检测系统
    整合损伤定位和分类功能
    
    修改说明：
    - 移除了prepare_training_data方法（数据已由仿真器生成）
    - 保留load_simulation_data用于加载HDF5数据
    - 保留训练、评估、预测等核心功能
    """
    
    def __init__(self, num_classes: int = 2, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_classes = num_classes
        self.device = device
        
        # 初始化各组件
        self.preprocessor = SignalPreprocessor()
        self.model = MultiModalDamageDetector(num_classes=num_classes).to(device)
        self.scaler = MinMaxScaler()
        
        # 定义图像变换, gvr的图像变换不能使用 flip 和 rotate
        self.train_transform = transforms.Compose([
            # 1. 沿时间/空间轴的几何增强（不改变物理量，只改变位置）
            # 类比 RandomResizedCrop，但更温和，避免破坏太多结构
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.95, 1.0),  # 最多保留区域，再缩回 224
                ratio=(0.95, 1.05), # 保持接近正方形
            ),

            # 2. X 轴轻微平移，模拟传感器位置不确定性
            SmallRandomTranslation(max_dx=3),

            # 3. 小核高斯模糊，模拟带宽有限与插值平滑
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.1)),

            # 4. 随机擦除：强迫模型不只依赖局部峰值
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.0)),

            # 5. 标准化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.valid_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.valid_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    

    def load_simulation_data(self, 
                            signals: np.ndarray, 
                            images: np.ndarray, 
                            labels: np.ndarray,
                            feature_selection: str = 'first_sensor',
                            sensor_indices: List[int] = None,
                            normalize: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载并处理仿真生成器生成的HDF5数据
        
        参数:
            signals: 仿真信号数据 (n_samples, n_dof, n_timepoints) - 已按窗口切片
            images: 图像数据 (n_samples, 3, 224, 224) - GVR特征图
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

        
        # 创建数据集
        train_dataset = OffshoreStructureDataset(
            X_train, img_train, y_train,
            transform=self.train_transform
        )
        val_dataset = OffshoreStructureDataset(
            X_val, img_val, y_val,
            transform=self.valid_transform
        )
        test_dataset = OffshoreStructureDataset(
            X_test, img_test, y_test,
            transform=self.valid_transform
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
        
        return train_loader, val_loader, test_loader



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
        # 在 train 函数开始时冻结 ResNet
        #for param in self.model.resnet.parameters():
         #   param.requires_grad = False
        #for param in self.model.resnet_fc.parameters():
         #   param.requires_grad = False

        criterion = nn.CrossEntropyLoss(label_smoothing=0.03) # 标签平滑
        #optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 1. 定义参数组
        base_params = []      # 从头训练的参数 (1D-CNN, Gate)
        classifier_params = []  # 分类器参数
        finetune_params = []  # 需要微调的参数

        for name, param in self.model.named_parameters():
            # 注意：根据你的代码，MobileNetV3 存储在 self.resnet 和 self.resnet_fc 中
            if 'resnet' in name:
                finetune_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                base_params.append(param)

        # 调节差异学习率
        base_lr = learning_rate * 0.5
        classifier_lr = learning_rate * 1
        finetune_lr = learning_rate * 1

        # 2. 设置差异学习率
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': base_lr},
            {'params': classifier_params, 'lr': classifier_lr},
            {'params': finetune_params, 'lr': finetune_lr}
            #{'params': self.model.resnet_fc.parameters(), 'lr': finetune_lr}
        ], weight_decay=3e-4)

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
                # === 梯度监控与裁剪 ===
                # 获取各分支的梯度范数
                ts_grad_norm = 0.0
                img_grad_norm = 0.0
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if 'ts_cnn' in name:
                            ts_grad_norm += param.grad.data.norm(2).item() ** 2
                        if 'resnet' in name:
                            img_grad_norm += param.grad.data.norm(2).item() ** 2
                
                ts_grad_norm = ts_grad_norm ** 0.5
                img_grad_norm = img_grad_norm ** 0.5

                # 使用全局裁剪防止爆炸，这能起到隐式的平衡作用
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if (batch_idx + 1) % 20 == 0:  # 按epoch打印进度太慢
                    # 如果图像分支梯度远大于时序分支，可以适当降低其学习率
                    print(f"Batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.4f} TS Grad: {ts_grad_norm:.4f} Img Grad: {img_grad_norm:.4f} Ratio: {img_grad_norm/(ts_grad_norm+1e-8):.2f}")
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
                print(f'\nEpoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%, '
                      f'Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%\n')
            '''
            if epoch + 1 == 1:
                print(f"epoch {epoch+1}: 解冻图像分支，开始联合训练...")
                for param in self.model.resnet.parameters():
                    param.requires_grad = True
                for param in self.model.resnet_fc.parameters(): 
                    param.requires_grad = True
                optimizer.add_param_group({
                    'params': finetune_params, 
                    #'params': self.model.resnet_fc.parameters(),
                    'lr': finetune_lr,  # 使用较小学习率微调
                    'weight_decay': 2e-4
                })
            '''
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
# 5. 示例使用代码（简化版，完整流程见test_run3.py）
# ============================================================================

# 移除的函数：
# - prepare_training_data: 自动标注和数据分割（数据已由仿真器生成）
# - _generate_simulated_image_features: 生成随机模拟图像（现在使用GVR特征图）
# - generate_simulated_data: 生成模拟监测数据（已由new_mdof_v1.py替代）
# - main(): 完整演示流程（已移至test_run3.py）

# 新的数据流程：
# 1. 运行 new_mdof_v1.py 生成 HDF5 数据
# 2. 使用 load_h5.py 加载数据
# 3. 调用 load_simulation_data 处理数据
# 4. 调用 train 训练模型
# 5. 调用 evaluate 评估模型

# 参考完整使用示例，请查看 test_run3.py


if __name__ == "__main__":
    print("=" * 70)
    print("海洋导管架平台损伤检测系统")
    print("基于MLP-ResNet50多模态深度学习框架")
    print("=" * 70)
    print("\n注意：完整的数据生成和训练流程请运行以下脚本：")
    print("  1. python new_mdof_v1.py  - 生成仿真数据")
    print("  2. python test_run3.py   - 加载数据并训练模型")
    print("\n或者直接使用 OffshoreDamageDetectionSystem 类进行自定义训练。")
