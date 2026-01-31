"""
海洋导管架平台损伤检测系统 - 重构版本
基于多模态深度学习框架(mobilenet v3 + 1d cnn) + 门控机制 + 异步学习率
参考论文: "Multimodal deep learning with integrated automatic labeling for structural damage detection"

重构目标：
1. 提高代码可测试性
2. 降低模块耦合度
3. 增强代码可读性和维护性
4. 实现单一职责原则
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
from typing import Tuple, List, Dict, Optional, Union, Any, Protocol
from sklearn.model_selection import train_test_split
from pytorch_lamb import Lamb
import warnings
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import os

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 配置和数据结构
# ============================================================================

class FeatureSelectionStrategy(Enum):
    """特征选择策略枚举"""
    FIRST_SENSOR = "first_sensor"
    MEAN = "mean"
    CONCATENATE = "concatenate"
    SPECIFIED = "specified"


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    label_smoothing: float = 0.03
    weight_decay: float = 3e-4


@dataclass
class ModelConfig:
    """模型配置"""
    num_classes: int = 2
    mlp_input_dim: int = 16
    mlp_hidden_dims: List[int] = None
    use_pretrained: bool = True
    ts_output_dim: int = 16
    img_output_dim: int = 16
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 32, 16]


@dataclass
class SignalConfig:
    """信号处理配置"""
    fs: float = 100.0
    cutoff_freq: float = 25.0
    order: int = 4
    window_size: int = 3000
    threshold: float = 0.1


# ============================================================================
# 接口定义
# ============================================================================

class SignalProcessorInterface(Protocol):
    """信号处理器接口"""
    
    def butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """应用Butterworth滤波器"""
        ...
    
    def calculate_damage_index(self, healthy_signal: np.ndarray, 
                              damaged_signal: np.ndarray) -> np.ndarray:
        """计算损伤指数"""
        ...
    
    def calculate_gvr(self, damage_index: np.ndarray) -> np.ndarray:
        """计算梯度变化率"""
        ...
    
    def detect_damage_probability(self, gvr: np.ndarray) -> float:
        """统计损伤发生概率"""
        ...


class FeatureExtractorInterface(Protocol):
    """特征提取器接口"""
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """从信号中提取特征"""
        ...


class ModelInterface(Protocol):
    """模型接口"""
    
    def forward(self, time_series: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        ...
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   config: TrainingConfig) -> Dict[str, List]:
        """训练模型"""
        ...
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        ...


# ============================================================================
# 信号处理模块
# ============================================================================

class SignalPreprocessor:
    """
    信号预处理类，实现Butterworth低通滤波和损伤指数计算
    遵循单一职责原则，只负责信号处理相关功能
    """
    
    def __init__(self, config: SignalConfig):
        """
        初始化预处理器
        
        参数:
            config: 信号处理配置
        """
        self.config = config
        
        # 设计Butterworth低通滤波器
        nyquist = 0.5 * config.fs
        normal_cutoff = config.cutoff_freq / nyquist
        self.b, self.a = signal.butter(config.order, normal_cutoff, btype='low', analog=False)
        
        logger.info(f"SignalPreprocessor initialized with fs={config.fs}, "
                   f"cutoff_freq={config.cutoff_freq}, order={config.order}")
    
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
                              damaged_signal: np.ndarray) -> np.ndarray:
        """
        计算损伤指数(Damage Index, DI)
        
        参数:
            healthy_signal: 健康状态信号
            damaged_signal: 损伤状态信号
            
        返回:
            damage_index: 损伤指数序列
        """
        n_samples = min(len(healthy_signal), len(damaged_signal))
        n_windows = n_samples // self.config.window_size
        
        damage_index = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * self.config.window_size
            end = start + self.config.window_size
            
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
    
    def detect_damage_probability(self, gvr: np.ndarray) -> float:
        """
        统计损伤发生概率
        
        参数:
            gvr: 梯度变化率序列
            
        返回:
            probability: 损伤发生概率
        """
        fault_occurrences = np.sum(gvr > self.config.threshold)
        total_windows = len(gvr)
        probability = (fault_occurrences / total_windows) * 100
        return probability


class StatisticalFeatureExtractor:
    """统计特征提取器"""
    
    def extract_features(self, signal_window: np.ndarray) -> np.ndarray:
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


# ============================================================================
# 数据变换模块
# ============================================================================

class SmallRandomTranslation:
    """
    对 X 轴（传感器空间轴）做轻微随机平移，模拟传感器位置扰动
    Y 轴保持不变，避免破坏时间顺序
    """
    def __init__(self, max_dx: int = 3):
        self.max_dx = max_dx  # 像素

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C, H, W) Tensor
        dx = random.randint(-self.max_dx, self.max_dx)
        if dx == 0:
            return img
        # 沿 W 维度滚动，实现平移
        return torch.roll(img, shifts=dx, dims=-1)


def create_train_transform() -> transforms.Compose:
    """创建训练数据变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=(224, 224),
            scale=(0.95, 1.0),
            ratio=(0.95, 1.05),
        ),
        SmallRandomTranslation(max_dx=3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.1)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_valid_transform() -> transforms.Compose:
    """创建验证数据变换"""
    return transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# 数据集类
# ============================================================================

class OffshoreStructureDataset(Dataset):
    """
    海洋结构多模态数据集
    只负责数据存储和访问，不包含处理逻辑
    """
    
    def __init__(self, time_series_data: np.ndarray, 
                 image_data: np.ndarray, 
                 labels: np.ndarray,
                 transform: Optional[transforms.Compose] = None):
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
        
        # 数据验证
        assert len(self.time_series) == len(self.images) == len(self.labels), \
            "数据长度不一致"
        
        logger.info(f"Dataset created with {len(self.labels)} samples")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_series = self.time_series[idx]
        image = self.images[idx]
        label = self.labels[idx]
        
        # 应用图像变换
        if self.transform:
            image = self.transform(image)
            
        return time_series, image, label


# ============================================================================
# 模型模块
# ============================================================================

class TimeSeriesCNN(nn.Module):
    """1D-CNN 时间序列特征提取器"""
    
    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, length) -> (batch_size, 1, length)
        x = x.unsqueeze(1)
        return self.layers(x)


class ImageFeatureExtractor(nn.Module):
    """图像特征提取器（基于MobileNetV3）"""
    
    def __init__(self, output_dim: int = 16, use_pretrained: bool = True):
        super().__init__()
        self.output_dim = output_dim
        
        # 加载预训练的 MobileNetV3 Large
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        
        # 特征提取部分
        self.backbone = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 特征降维
        resnet_feature_dim = 960  # MobileNetV3 Large 的输出通道数
        self.fc = nn.Sequential(
            nn.Linear(resnet_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (batch_size, 960, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 960)
        return self.fc(features)


class GatedFusionModule(nn.Module):
    """门控融合模块"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.fusion_bn = nn.BatchNorm1d(input_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, ts_features: torch.Tensor, img_features: torch.Tensor) -> torch.Tensor:
        # 拼接特征
        fused_features = torch.cat([ts_features, img_features], dim=1)
        fused_features = self.fusion_bn(fused_features)
        
        # 应用门控机制
        gate_weights = self.gate(fused_features)
        gated_features = fused_features * gate_weights
        
        return gated_features


class MultiModalDamageDetector(nn.Module):
    """
    多模态损伤检测网络
    融合时间序列和图像特征
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 时间序列特征提取器
        self.ts_cnn = TimeSeriesCNN(output_dim=config.ts_output_dim)
        
        # 图像特征提取器
        self.img_extractor = ImageFeatureExtractor(
            output_dim=config.img_output_dim, 
            use_pretrained=config.use_pretrained
        )
        
        # 门控融合模块
        fused_dim = config.ts_output_dim + config.img_output_dim
        self.fusion = GatedFusionModule(fused_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, config.num_classes)
        )
        
        logger.info(f"MultiModalDamageDetector created with {config.num_classes} classes")
    
    def forward(self, time_series: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # 1. 提取时间序列特征
        ts_features = self.ts_cnn(time_series)
        
        # 2. 提取图像特征
        img_features = self.img_extractor(images)
        
        # 3. 门控融合
        fused_features = self.fusion(ts_features, img_features)
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        return logits


# ============================================================================
# 训练器模块
# ============================================================================

class ModelTrainer:
    """模型训练器，负责训练逻辑"""
    
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self.model.to(device)
        
        logger.info(f"ModelTrainer initialized on device: {device}")
    
    def setup_optimizer(self, learning_rate: float, weight_decay: float) -> optim.Optimizer:
        """设置优化器"""
        # 参数分组
        base_params = []      # 从头训练的参数
        classifier_params = []  # 分类器参数
        finetune_params = []  # 需要微调的参数

        for name, param in self.model.named_parameters():
            if 'img_extractor' in name:
                finetune_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                base_params.append(param)

        # 差异学习率
        base_lr = learning_rate * 0.1
        classifier_lr = learning_rate * 1
        finetune_lr = learning_rate * 1

        optimizer = optim.AdamW([
            {'params': base_params, 'lr': base_lr},
            {'params': classifier_params, 'lr': classifier_lr},
            {'params': finetune_params, 'lr': finetune_lr}
        ], weight_decay=weight_decay)

        return optimizer
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (time_series, images, labels) in enumerate(train_loader):
            time_series = time_series.to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(time_series, images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 梯度裁剪
            self._clip_gradients()
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for time_series, images, labels in val_loader:
                time_series = time_series.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(time_series, images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def _clip_gradients(self):
        """梯度裁剪"""
        # 分层梯度裁剪
        ts_params = [p for n, p in self.model.named_parameters() 
                    if 'ts_cnn' in n and p.grad is not None]
        img_params = [p for n, p in self.model.named_parameters() 
                     if 'img_extractor' in n and p.grad is not None]

        if ts_params:
            torch.nn.utils.clip_grad_norm_(ts_params, max_norm=0.8)
        if img_params:
            torch.nn.utils.clip_grad_norm_(img_params, max_norm=5.0)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              config: TrainingConfig) -> Dict[str, List]:
        """训练模型"""
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        optimizer = self.setup_optimizer(config.learning_rate, config.weight_decay)
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
        
        for epoch in range(config.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # 记录历史
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_damage_detector.pth')
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
            
            logger.info(f'Epoch [{epoch+1}/{config.epochs}], '
                       f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        return history


# ============================================================================
# 数据处理器模块
# ============================================================================

class DataProcessor:
    """数据处理器，负责数据预处理和加载"""
    
    def __init__(self, signal_config: SignalConfig, 
                 feature_extractor: FeatureExtractorInterface):
        self.signal_config = signal_config
        self.feature_extractor = feature_extractor
        self.scaler = MinMaxScaler()
        
        logger.info("DataProcessor initialized")
    
    def extract_time_series_features(self, signals: np.ndarray, 
                                   strategy: FeatureSelectionStrategy,
                                   sensor_indices: Optional[List[int]] = None) -> np.ndarray:
        """提取时间序列特征"""
        n_samples, n_dof, n_timepoints = signals.shape
        
        if strategy == FeatureSelectionStrategy.FIRST_SENSOR:
            features = np.array([
                self.feature_extractor.extract_features(signals[i, 0, :])
                for i in range(n_samples)
            ])
            
        elif strategy == FeatureSelectionStrategy.MEAN:
            averaged_signals = signals.mean(axis=1)
            features = np.array([
                self.feature_extractor.extract_features(averaged_signals[i, :])
                for i in range(n_samples)
            ])
            
        elif strategy == FeatureSelectionStrategy.CONCATENATE:
            n_sensors_to_use = min(3, n_dof)
            features_per_sensor = 16
            features = np.zeros((n_samples, n_sensors_to_use * features_per_sensor))
            
            for i in range(n_samples):
                for j in range(n_sensors_to_use):
                    sensor_idx = j
                    feat = self.feature_extractor.extract_features(signals[i, sensor_idx, :])
                    features[i, j*features_per_sensor:(j+1)*features_per_sensor] = feat
                    
        elif strategy == FeatureSelectionStrategy.SPECIFIED:
            if sensor_indices is None:
                sensor_indices = [0, n_dof//2, n_dof-1]
                logger.warning(f"未指定传感器索引，使用默认: {sensor_indices}")
            
            features_per_sensor = 16
            features = np.zeros((n_samples, len(sensor_indices) * features_per_sensor))
            
            for i in range(n_samples):
                for j, sensor_idx in enumerate(sensor_indices):
                    feat = self.feature_extractor.extract_features(signals[i, sensor_idx, :])
                    features[i, j*features_per_sensor:(j+1)*features_per_sensor] = feat
        
        else:
            raise ValueError(f"未知的特征选择策略: {strategy}")
        
        # 处理NaN和Inf
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("检测到NaN或Inf值，正在修复...")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """归一化特征"""
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)
    
    def process_images(self, images: np.ndarray) -> np.ndarray:
        """处理图像数据"""
        # 确保图像在 [0, 1] 范围内
        images = np.clip(images, 0, 1)
        return images
    
    def split_data(self, time_series_features: np.ndarray, 
                  images: np.ndarray, labels: np.ndarray,
                  train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple:
        """划分数据集"""
        test_ratio = 1.0 - train_ratio - val_ratio
        
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
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                img_train, img_val, img_test)
    
    def create_data_loaders(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           img_train: np.ndarray, img_val: np.ndarray, img_test: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        
        # 创建数据集
        train_dataset = OffshoreStructureDataset(
            X_train, img_train, y_train,
            transform=create_train_transform()
        )
        val_dataset = OffshoreStructureDataset(
            X_val, img_val, y_val,
            transform=create_valid_transform()
        )
        test_dataset = OffshoreStructureDataset(
            X_test, img_test, y_test,
            transform=create_valid_transform()
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data loaders created with batch_size={batch_size}")
        logger.info(f"Train batches: {len(train_loader)}, "
                   f"Val batches: {len(val_loader)}, "
                   f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader


# ============================================================================
# 评估器模块
# ============================================================================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型性能"""
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


# ============================================================================
# 可视化模块
# ============================================================================

class TrainingVisualizer:
    """训练可视化器"""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None):
        """绘制训练历史"""
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
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """绘制混淆矩阵"""
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
# 主系统类
# ============================================================================

class OffshoreDamageDetectionSystem:
    """
    海洋结构损伤检测系统
    整合所有模块，提供统一的接口
    """
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 signal_config: Optional[SignalConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化系统
        
        参数:
            model_config: 模型配置
            signal_config: 信号处理配置
            training_config: 训练配置
            device: 计算设备
        """
        # 使用默认配置或用户提供的配置
        self.model_config = model_config or ModelConfig()
        self.signal_config = signal_config or SignalConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device
        
        # 初始化各组件
        self.signal_processor = SignalPreprocessor(self.signal_config)
        self.feature_extractor = StatisticalFeatureExtractor()
        self.data_processor = DataProcessor(self.signal_config, self.feature_extractor)
        self.model = MultiModalDamageDetector(self.model_config)
        self.trainer = ModelTrainer(self.model, self.device)
        self.evaluator = ModelEvaluator(self.model, self.device)
        self.visualizer = TrainingVisualizer()
        
        logger.info(f"OffshoreDamageDetectionSystem initialized on {device}")
    
    def load_simulation_data(self, 
                           signals: np.ndarray, 
                           images: np.ndarray, 
                           labels: np.ndarray,
                           feature_selection: FeatureSelectionStrategy = FeatureSelectionStrategy.FIRST_SENSOR,
                           sensor_indices: Optional[List[int]] = None,
                           normalize: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载并处理仿真数据
        
        参数:
            signals: 仿真信号数据 (n_samples, n_dof, n_timepoints)
            images: 图像数据 (n_samples, 3, 224, 224)
            labels: 损伤标签 (n_samples,)
            feature_selection: 特征选择策略
            sensor_indices: 指定的传感器索引列表
            normalize: 是否对特征进行归一化
        
        返回:
            train_loader, val_loader, test_loader: 数据加载器
        """
        logger.info("开始加载仿真数据...")
        
        # 1. 提取时间序列特征
        logger.info(f"特征选择策略: {feature_selection.value}")
        time_series_features = self.data_processor.extract_time_series_features(
            signals, feature_selection, sensor_indices
        )
        
        # 2. 归一化特征
        if normalize:
            time_series_features = self.data_processor.normalize_features(
                time_series_features, fit=True
            )
        
        # 3. 处理图像数据
        images = self.data_processor.process_images(images)
        
        # 4. 划分数据集
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         img_train, img_val, img_test) = self.data_processor.split_data(
            time_series_features, images, labels
        )
        
        # 5. 创建数据加载器
        train_loader, val_loader, test_loader = self.data_processor.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            img_train, img_val, img_test,
            self.training_config.batch_size
        )
        
        logger.info("✓ 数据加载完成！")
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """训练模型"""
        logger.info("开始训练模型...")
        history = self.trainer.train(train_loader, val_loader, self.training_config)
        logger.info("✓ 模型训练完成！")
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        logger.info("开始评估模型...")
        metrics = self.evaluator.evaluate(test_loader)
        
        logger.info(f"评估结果:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision_macro']:.4f}")
        logger.info(f"  召回率: {metrics['recall_macro']:.4f}")
        logger.info(f"  F1分数: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def detect_damage_location(self, 
                             current_signals: Dict[int, np.ndarray],
                             healthy_baseline: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        基于GVR分析进行损伤定位
        
        参数:
            current_signals: 当前信号 {测点ID: 信号}
            healthy_baseline: 健康基准信号
            
        返回:
            damage_probabilities: 各测点的损伤概率
        """
        damage_probabilities = {}
        
        for sensor_id, signal_data in current_signals.items():
            if sensor_id not in healthy_baseline:
                logger.warning(f"传感器 {sensor_id} 没有对应的基准信号")
                continue
                
            # 滤波
            filtered_signal = self.signal_processor.butterworth_filter(signal_data)
            filtered_baseline = self.signal_processor.butterworth_filter(healthy_baseline[sensor_id])
            
            # 计算GVR和损伤概率
            di = self.signal_processor.calculate_damage_index(filtered_baseline, filtered_signal)
            gvr = self.signal_processor.calculate_gvr(di)
            probability = self.signal_processor.detect_damage_probability(gvr)
            
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
        time_series_features = self.data_processor.normalize_features(time_series_features)
        
        with torch.no_grad():
            time_series_tensor = torch.FloatTensor(time_series_features).unsqueeze(0).to(self.device)
            image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)
            
            outputs = self.model(time_series_tensor, image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        return predicted_class.item(), confidence.item()
    
    def plot_training_results(self, history: Dict[str, List], 
                            metrics: Dict[str, float],
                            save_dir: str = "results"):
        """绘制训练结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制训练历史
        self.visualizer.plot_training_history(
            history, 
            save_path=os.path.join(save_dir, "training_history.png")
        )
        
        # 绘制混淆矩阵
        self.visualizer.plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=os.path.join(save_dir, "confusion_matrix.png")
        )
        
        logger.info(f"训练结果已保存到 {save_dir} 目录")


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数示例"""
    print("=" * 70)
    print("海洋导管架平台损伤检测系统 - 重构版本")
    print("基于多模态深度学习框架")
    print("=" * 70)
    
    # 创建系统实例
    system = OffshoreDamageDetectionSystem()
    
    print("\n系统已初始化，包含以下模块：")
    print("  - SignalPreprocessor: 信号预处理")
    print("  - StatisticalFeatureExtractor: 统计特征提取")
    print("  - DataProcessor: 数据处理")
    print("  - MultiModalDamageDetector: 多模态损伤检测模型")
    print("  - ModelTrainer: 模型训练器")
    print("  - ModelEvaluator: 模型评估器")
    print("  - TrainingVisualizer: 训练可视化器")
    
    print("\n使用示例：")
    print("  # 1. 加载数据")
    print("  train_loader, val_loader, test_loader = system.load_simulation_data(signals, images, labels)")
    print("  ")
    print("  # 2. 训练模型")
    print("  history = system.train(train_loader, val_loader)")
    print("  ")
    print("  # 3. 评估模型")
    print("  metrics = system.evaluate(test_loader)")
    print("  ")
    print("  # 4. 可视化结果")
    print("  system.plot_training_results(history, metrics)")


if __name__ == "__main__":
    main()
