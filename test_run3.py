import numpy as np
import torch
from model1 import OffshoreDamageDetectionSystem
from load_h5 import load_h5_dataset
import matplotlib.pyplot as plt
import os


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
    BATCH_SIZE = 16
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
    
    try:
        signals, images, labels = load_h5_dataset(DATA_DIR)
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    # 检查数据量
    n_samples = len(labels)
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
    
    # ==================== 3. 加载并处理仿真数据 ====================
    print("\n[步骤3] 处理仿真数据...")
    
    # 尝试不同的特征选择策略
    train_loader, val_loader, test_loader = detection_system.load_simulation_data(
        signals=signals,
        images=images,
        labels=labels,
        feature_selection='mean',  # 推荐使用 'mean' 或 'concatenate'
        normalize=True
    )
    
    # ==================== 4. 训练模型 ====================
    print("\n[步骤4] 训练模型...")
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
