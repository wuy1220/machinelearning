# ============================================================================
# 示例脚本：使用仿真生成器数据训练模型
# ============================================================================

import numpy as np
import torch
from model import OffshoreDamageDetectionSystem
from mdof_emu_2_noisier import generate_mdof_dataset_parallel
import matplotlib.pyplot as plt

def main_with_simulation_data():
    """
    使用仿真生成器数据训练损伤检测模型的完整流程
    """
    print("=" * 70)
    print("使用仿真生成器数据训练损伤检测模型")
    print("=" * 70)
    
    # ==================== 1. 生成仿真数据 ====================
    print("\n[步骤1] 生成仿真数据...")
    print("  这可能需要几分钟时间，请耐心等待...")
    
    signals, images, labels = generate_mdof_dataset_parallel(
        n_samples=1000,
        n_dof=10,
        segment_time=10.0,
        n_jobs=-1,
        force_regen=False,  # 设为True强制重新生成
        seed=42, 
        noise_level=0,      # 噪声水平
        damage_subtlety=0,   # 损伤隐蔽程度
        add_texture=False        # 添加纹理
    )
    
    print(f"  ✓ 生成 {len(signals)} 个样本")
    print(f"    - 信号形状: {signals.shape}")
    print(f"    - 图像形状: {images.shape}")
    print(f"    - 标签分布: {np.bincount(labels)}")
    
    # ==================== 2. 初始化检测系统 ====================
    print("\n[步骤2] 初始化检测系统...")
    
    NUM_CLASSES = len(np.unique(labels))
    detection_system = OffshoreDamageDetectionSystem(
        num_classes=NUM_CLASSES,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"  ✓ 设备: {detection_system.device}")
    print(f"  ✓ 类别数: {NUM_CLASSES}")
    
    # ==================== 3. 加载仿真数据 ====================
    print("\n[步骤3] 加载并处理仿真数据...")
    
    # 尝试不同的特征选择策略
    train_loader, val_loader, test_loader = detection_system.load_simulation_data(
        signals=signals,
        images=images,
        labels=labels,
        feature_selection='first_sensor',  # 可选: 'first_sensor', 'mean', 'concatenate', 'specified'
        normalize=True
    )
    
    # ==================== 4. 训练模型 ====================
    print("\n[步骤4] 训练模型...")
    
    history = detection_system.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    # 绘制训练曲线
    detection_system.plot_training_history(
        history, 
        save_path='simulation_training_history.png'
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
    class_names = [f'类别{i}' for i in range(NUM_CLASSES)]
    detection_system.plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names,
        save_path='simulation_confusion_matrix.png'
    )
    
    # ==================== 6. 保存模型 ====================
    print("\n[步骤6] 保存模型...")
    torch.save(detection_system.model.state_dict(), 'simulation_trained_model.pth')
    print("  ✓ 模型已保存: simulation_trained_model.pth")
    
    print("\n" + "=" * 70)
    print("✓ 训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    main_with_simulation_data()
