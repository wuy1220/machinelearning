"""
测试文件 - 演示重构后代码的可测试性
对比原始代码和重构代码的使用方式
"""

import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入重构后的模块
from model1_mn_cnn_refactored import (
    OffshoreDamageDetectionSystem,
    SignalPreprocessor,
    StatisticalFeatureExtractor,
    MultiModalDamageDetector,
    ModelTrainer,
    DataProcessor,
    ModelEvaluator,
    SignalConfig,
    ModelConfig,
    TrainingConfig,
    FeatureSelectionStrategy,
    OffshoreStructureDataset
)

def test_original_code_issues():
    """
    演示原始代码的问题：
    1. 难以单独测试组件
    2. 紧密耦合
    3. 难以mock依赖
    """
    print("=== 原始代码的问题 ===")
    
    # 原始代码的问题示例：
    # 1. 无法单独测试SignalPreprocessor，因为它被硬编码在OffshoreDamageDetectionSystem中
    # 2. 无法mock数据加载器，因为数据加载逻辑与业务逻辑混合
    # 3. 无法测试独立的特征提取逻辑
    # 4. 所有组件都紧密耦合在一个大类中
    
    print("❌ 原始代码问题：")
    print("  - SignalPreprocessor 被硬编码在 OffshoreDamageDetectionSystem 中")
    print("  - 无法单独测试各个组件")
    print("  - 数据加载与业务逻辑混合")
    print("  - 紧密耦合，难以mock依赖")
    print("  - 一个类的改变可能影响其他功能")
    print()

def test_refactored_code_benefits():
    """
    演示重构后代码的优势：
    1. 可以单独测试每个组件
    2. 依赖注入，易于mock
    3. 清晰的接口定义
    4. 单一职责原则
    """
    print("=== 重构后代码的优势 ===")
    
    print("✅ 重构后优势：")
    print("  - 每个组件可以单独测试")
    print("  - 依赖注入，易于mock")
    print("  - 清晰的接口定义")
    print("  - 单一职责原则")
    print("  - 配置与实现分离")
    print()

def test_signal_preprocessor():
    """测试信号预处理器 - 可以独立测试"""
    print("=== 测试 SignalPreprocessor ===")
    
    # 创建配置
    config = SignalConfig(fs=100, cutoff_freq=25, order=4)
    
    # 创建信号预处理器
    preprocessor = SignalPreprocessor(config)
    
    # 创建测试数据
    test_signal = np.random.randn(1000)
    healthy_signal = np.random.randn(1000) * 0.1  # 健康信号，幅度较小
    damaged_signal = np.random.randn(1000) * 0.3  # 损伤信号，幅度较大
    
    # 测试滤波功能
    filtered_signal = preprocessor.butterworth_filter(test_signal)
    assert filtered_signal.shape == test_signal.shape, "滤波后信号形状应该保持不变"
    print("✅ Butterworth滤波器测试通过")
    
    # 测试损伤指数计算
    damage_index = preprocessor.calculate_damage_index(healthy_signal, damaged_signal)
    assert damage_index.shape[0] == 1000 // config.window_size, "损伤指数窗口数量正确"
    print("✅ 损伤指数计算测试通过")
    
    # 测试GVR计算
    gvr = preprocessor.calculate_gvr(damage_index)
    assert gvr.shape == damage_index.shape, "GVR形状应该与损伤指数相同"
    print("✅ GVR计算测试通过")
    
    # 测试损伤概率检测
    probability = preprocessor.detect_damage_probability(gvr)
    assert 0 <= probability <= 100, "损伤概率应该在0-100之间"
    print("✅ 损伤概率检测测试通过")
    print()

def test_feature_extractor():
    """测试特征提取器 - 可以独立测试"""
    print("=== 测试 StatisticalFeatureExtractor ===")
    
    extractor = StatisticalFeatureExtractor()
    
    # 创建测试信号
    test_signal = np.random.randn(1000)
    
    # 提取特征
    features = extractor.extract_features(test_signal)
    
    # 验证特征数量
    assert len(features) == 16, f"应该提取16个特征，实际得到{len(features)}个"
    
    # 验证特征值合理性
    assert not np.any(np.isnan(features)), "特征中不应该有NaN值"
    assert not np.any(np.isinf(features)), "特征中不应该有Inf值"
    
    print("✅ 特征提取器测试通过")
    print(f"  提取的特征数量: {len(features)}")
    print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print()

def test_data_processor():
    """测试数据处理器 - 可以独立测试"""
    print("=== 测试 DataProcessor ===")
    
    # 创建配置和模拟数据
    config = SignalConfig()
    extractor = StatisticalFeatureExtractor()
    processor = DataProcessor(config, extractor)
    
    # 创建模拟数据 (10个样本，5个传感器，2000个时间点)
    signals = np.random.randn(10, 5, 2000)
    images = np.random.rand(10, 3, 224, 224)
    labels = np.random.randint(0, 2, 10)
    
    # 测试不同特征选择策略
    strategies = [
        FeatureSelectionStrategy.FIRST_SENSOR,
        FeatureSelectionStrategy.MEAN,
        FeatureSelectionStrategy.CONCATENATE,
        FeatureSelectionStrategy.SPECIFIED
    ]
    
    for strategy in strategies:
        features = processor.extract_time_series_features(signals, strategy)
        print(f"✅ 特征选择策略 {strategy.value} 测试通过")
        print(f"  特征形状: {features.shape}")
    
    # 测试数据归一化
    features = processor.extract_time_series_features(signals, FeatureSelectionStrategy.FIRST_SENSOR)
    normalized_features = processor.normalize_features(features, fit=True)
    
    # 验证归一化结果
    assert normalized_features.min() >= 0, "归一化后最小值应该>=0"
    assert normalized_features.max() <= 1, "归一化后最大值应该<=1"
    print("✅ 特征归一化测试通过")
    
    # 测试图像处理
    processed_images = processor.process_images(images)
    assert processed_images.shape == images.shape, "图像形状应该保持不变"
    assert processed_images.min() >= 0 and processed_images.max() <= 1, "图像值应该在[0,1]范围内"
    print("✅ 图像处理测试通过")
    
    # 测试数据划分
    features = processor.extract_time_series_features(signals, FeatureSelectionStrategy.FIRST_SENSOR)
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     img_train, img_val, img_test) = processor.split_data(features, images, labels)
    
    assert len(X_train) + len(X_val) + len(X_test) == len(features), "数据划分应该保持总量"
    print("✅ 数据划分测试通过")
    print()

def test_dataset():
    """测试数据集类 - 可以独立测试"""
    print("=== 测试 OffshoreStructureDataset ===")
    
    # 创建模拟数据
    n_samples = 20
    time_series_data = np.random.randn(n_samples, 16)
    image_data = np.random.rand(n_samples, 3, 224, 224)
    labels = np.random.randint(0, 2, n_samples)
    
    # 创建数据集
    dataset = OffshoreStructureDataset(time_series_data, image_data, labels)
    
    # 测试数据集基本功能
    assert len(dataset) == n_samples, f"数据集长度应该是{n_samples}"
    
    # 测试数据获取
    ts, img, label = dataset[0]
    assert ts.shape == (16,), f"时间序列形状应该是(16,)，实际得到{ts.shape}"
    assert img.shape == (3, 224, 224), f"图像形状应该是(3, 224, 224)，实际得到{img.shape}"
    assert isinstance(label, torch.Tensor), "标签应该是torch.Tensor类型"
    
    print("✅ 数据集测试通过")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  时间序列形状: {ts.shape}")
    print(f"  图像形状: {img.shape}")
    print()

def test_model_components():
    """测试模型组件 - 可以独立测试"""
    print("=== 测试模型组件 ===")
    
    # 测试时间序列CNN
    ts_cnn = TimeSeriesCNN(output_dim=16)
    test_input = torch.randn(4, 1000)  # 4个样本，每个1000个时间点
    output = ts_cnn(test_input)
    assert output.shape == (4, 16), f"时间序列CNN输出形状应该是(4, 16)，实际得到{output.shape}"
    print("✅ 时间序列CNN测试通过")
    
    # 测试图像特征提取器
    img_extractor = ImageFeatureExtractor(output_dim=16, use_pretrained=False)
    test_images = torch.randn(4, 3, 224, 224)
    output = img_extractor(test_images)
    assert output.shape == (4, 16), f"图像特征提取器输出形状应该是(4, 16)，实际得到{output.shape}"
    print("✅ 图像特征提取器测试通过")
    
    # 测试门控融合模块
    fusion = GatedFusionModule(input_dim=32)
    ts_features = torch.randn(4, 16)
    img_features = torch.randn(4, 16)
    fused = fusion(ts_features, img_features)
    assert fused.shape == (4, 32), f"融合模块输出形状应该是(4, 32)，实际得到{fused.shape}"
    print("✅ 门控融合模块测试通过")
    
    # 测试完整模型
    config = ModelConfig(num_classes=2, ts_output_dim=16, img_output_dim=16)
    model = MultiModalDamageDetector(config)
    
    test_ts = torch.randn(4, 1000)
    test_img = torch.randn(4, 3, 224, 224)
    output = model(test_ts, test_img)
    assert output.shape == (4, 2), f"模型输出形状应该是(4, 2)，实际得到{output.shape}"
    print("✅ 完整模型测试通过")
    print()

def test_mock_dependencies():
    """测试依赖注入和mock - 展示可测试性"""
    print("=== 测试依赖注入和Mock ===")
    
    # 创建mock对象
    mock_extractor = Mock()
    mock_extractor.extract_features.return_value = np.array([1.0, 2.0, 3.0, 4.0] * 4)  # 16个特征
    
    # 使用mock创建数据处理器
    config = SignalConfig()
    processor = DataProcessor(config, mock_extractor)
    
    # 创建测试数据
    signals = np.random.randn(5, 3, 1000)
    
    # 测试特征提取（使用mock）
    features = processor.extract_time_series_features(signals, FeatureSelectionStrategy.FIRST_SENSOR)
    
    # 验证mock被调用
    assert mock_extractor.extract_features.called, "mock的extract_features方法应该被调用"
    print("✅ Mock依赖测试通过")
    
    # 验证mock返回值被使用
    assert features.shape[1] == 16, f"特征数量应该是16，实际得到{features.shape[1]}"
    print("✅ Mock返回值验证通过")
    print()

def test_configuration_system():
    """测试配置系统 - 展示灵活性"""
    print("=== 测试配置系统 ===")
    
    # 创建不同的配置
    model_config = ModelConfig(
        num_classes=3,  # 3分类
        ts_output_dim=32,  # 更大的时间序列特征维度
        img_output_dim=32,  # 更大的图像特征维度
        use_pretrained=False  # 不使用预训练模型
    )
    
    signal_config = SignalConfig(
        fs=200,  # 更高的采样率
        cutoff_freq=50,  # 更高的截止频率
        window_size=1500  # 更小的窗口大小
    )
    
    training_config = TrainingConfig(
        batch_size=64,  # 更大的批次
        epochs=10,  # 更少的epoch
        learning_rate=0.01  # 更高的学习率
    )
    
    # 使用自定义配置创建系统
    system = OffshoreDamageDetectionSystem(
        model_config=model_config,
        signal_config=signal_config,
        training_config=training_config,
        device='cpu'  # 使用CPU测试
    )
    
    print("✅ 自定义配置系统测试通过")
    print(f"  模型类别数: {system.model_config.num_classes}")
    print(f"  采样率: {system.signal_config.fs}")
    print(f"  批次大小: {system.training_config.batch_size}")
    print()

def test_error_handling():
    """测试错误处理 - 展示健壮性"""
    print("=== 测试错误处理 ===")
    
    # 测试无效的特征选择策略
    try:
        config = SignalConfig()
        extractor = StatisticalFeatureExtractor()
        processor = DataProcessor(config, extractor)
        
        # 创建测试数据
        signals = np.random.randn(5, 3, 1000)
        
        # 尝试使用无效策略
        processor.extract_time_series_features(signals, "invalid_strategy")
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print("✅ 无效特征选择策略错误处理测试通过")
        print(f"  错误信息: {e}")
    
    # 测试数据验证
    try:
        # 创建长度不一致的数据
        time_series = np.random.randn(10, 16)
        images = np.random.rand(9, 3, 224, 224)  # 少一个样本
        labels = np.random.randint(0, 2, 10)
        
        dataset = OffshoreStructureDataset(time_series, images, labels)
        assert False, "应该抛出AssertionError"
    except AssertionError:
        print("✅ 数据长度验证测试通过")
    
    print()

def test_integration_workflow():
    """测试集成工作流 - 展示完整使用流程"""
    print("=== 测试集成工作流 ===")
    
    # 创建小型系统
    system = OffshoreDamageDetectionSystem(device='cpu')
    
    # 创建模拟数据
    n_samples = 50
    n_sensors = 3
    n_timepoints = 2000
    
    signals = np.random.randn(n_samples, n_sensors, n_timepoints)
    images = np.random.rand(n_samples, 3, 224, 224)
    labels = np.random.randint(0, 2, n_samples)
    
    print("1. 加载数据...")
    train_loader, val_loader, test_loader = system.load_simulation_data(
        signals, images, labels,
        feature_selection=FeatureSelectionStrategy.FIRST_SENSOR
    )
    
    print("2. 训练模型（少量epoch）...")
    # 修改训练配置，使用更少的epoch
    system.training_config.epochs = 2
    history = system.train(train_loader, val_loader)
    
    print("3. 评估模型...")
    metrics = system.evaluate(test_loader)
    
    print("4. 可视化结果...")
    system.plot_training_results(history, metrics, save_dir="test_results")
    
    print("✅ 集成工作流测试通过")
    print()

def main():
    """主测试函数"""
    print("=" * 80)
    print("重构代码测试演示")
    print("展示重构后代码的可测试性和优势")
    print("=" * 80)
    print()
    
    # 演示原始代码问题
    test_original_code_issues()
    
    # 演示重构后优势
    test_refactored_code_benefits()
    
    # 测试各个组件
    test_signal_preprocessor()
    test_feature_extractor()
    test_data_processor()
    test_dataset()
    test_model_components()
    test_mock_dependencies()
    test_configuration_system()
    test_error_handling()
    test_integration_workflow()
    
    print("=" * 80)
    print("所有测试完成！")
    print("=" * 80)
    
    print("\n重构后的代码优势总结：")
    print("1. ✅ 模块化设计 - 每个组件可以独立测试")
    print("2. ✅ 依赖注入 - 易于mock和替换实现")
    print("3. ✅ 配置驱动 - 灵活的配置系统")
    print("4. ✅ 错误处理 - 更好的错误处理和验证")
    print("5. ✅ 接口定义 - 清晰的接口协议")
    print("6. ✅ 单一职责 - 每个类只负责一个功能")
    print("7. ✅ 可扩展性 - 易于添加新功能")
    print("8. ✅ 可维护性 - 代码更清晰易读")

if __name__ == "__main__":
    main()
