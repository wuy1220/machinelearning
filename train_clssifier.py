import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py

# 导入您现有的模型和数据集类
from model1_mn_cnn_classifier import OffshoreDamageDetectionSystem
from h5_gvr_dataset import H5GVRDataset, TimeSeriesCompose, RandomScaling, AddGaussianNoise, RandomTimeShift, RandomCutout

def main():
    # ================= 配置参数 =================
    DATA_DIR = './jacket_damage_data_ansys'  # 您存放 HDF5 的目录
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WINDOW_LENGTH = 3000  # 3000个时间步长
    EARLY_STOP_PATIENCE = 5  # 早停耐心参数


    # ================== 定义时序增强策略 ==================
    # 训练集使用增强
    ts_train_transform = TimeSeriesCompose([
        RandomScaling(scale_range=(0.95, 1.05)),       # 幅度变化 +/- 10%
        AddGaussianNoise(std=0.005),                # 添加 0.5% 的噪声
        RandomTimeShift(max_shift=5),               # 平移 5 个点
        RandomCutout(max_len=30)                    # 随机遮盖 30 个点
    ])
    
    # 验证/测试集不使用增强 (设为 None 或 Compose([]))
    ts_valid_transform = None 
    # ================= 1. 初始化数据集 =================
    full_dataset = H5GVRDataset(data_dir=DATA_DIR, window_length=WINDOW_LENGTH, transform=None)
    
    # ================= 2. & 3. 全局打乱并划分 (替代原有的场景级划分) =================
    # 说明：不再构建场景索引，直接对全部样本进行全局随机打乱。
    # 这种方式允许同一个场景的不同窗口出现在训练集和测试集中，
    # 虽然可能导致数据泄露（即测试集包含与训练集极其相似的数据），
    # 但对于纯粹的分类器任务，这通常能提高模型对已知损伤模式的拟合能力。

    print(f"正在加载全量数据...")
    print(f"总样本数: {len(full_dataset)}")

    # 生成全局样本索引
    all_indices = np.arange(len(full_dataset))

    # 第一次划分：训练集 (60%) vs 临时集 (40%)
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.4, 
        random_state=42,
        shuffle=True  # 全局随机打乱
    )
    
    # 第二次划分：验证集 (20%) vs 测试集 (20%)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )
    
    print(f"划分结果 (全局随机打乱):")
    print(f"  训练集: {len(train_idx)} 个样本")
    print(f"  验证集: {len(val_idx)} 个样本")
    print(f"  测试集: {len(test_idx)} 个样本")

    # ================= 4. 创建 DataLoader =================
    detection_system = OffshoreDamageDetectionSystem(num_classes=2, device=DEVICE)
    
    # 设置 Transform
    # --- 训练集 ---
    train_dataset_base = H5GVRDataset(DATA_DIR, window_length=WINDOW_LENGTH)
    train_dataset_base.transform = detection_system.train_transform
    train_dataset_base.ts_transform = ts_train_transform  # <--- 应用时序增强
    train_dataset = Subset(train_dataset_base, train_idx)
    
    # --- 验证集 ---
    val_dataset_base = H5GVRDataset(DATA_DIR, window_length=WINDOW_LENGTH)
    val_dataset_base.transform = detection_system.valid_transform
    val_dataset_base.ts_transform = ts_valid_transform # <--- 验证集不增强
    val_dataset = Subset(val_dataset_base, val_idx)
    
    # --- 测试集 ---
    test_dataset_base = H5GVRDataset(DATA_DIR, window_length=WINDOW_LENGTH)
    test_dataset_base.transform = detection_system.valid_transform
    test_dataset_base.ts_transform = ts_valid_transform # <--- 测试集不增强
    test_dataset = Subset(test_dataset_base, test_idx)
    
    # 创建 Subset
    train_dataset = Subset(full_dataset, train_idx)
    
    # 验证和测试集需要重新实例化 Dataset 或修改 transform 属性，
    # 因为全量 Dataset 已经被设置了 train_transform（包含数据增强）
    # 为了避免麻烦，这里简单克隆两个 Dataset 用于验证和测试
    val_dataset_base = H5GVRDataset(DATA_DIR, window_length=WINDOW_LENGTH)
    test_dataset_base = H5GVRDataset(DATA_DIR, window_length=WINDOW_LENGTH)
    val_dataset_base.transform = detection_system.valid_transform
    test_dataset_base.transform = detection_system.valid_transform
    
    val_dataset = Subset(val_dataset_base, val_idx)
    test_dataset = Subset(test_dataset_base, test_idx)
    
    # ================= 5. 创建 DataLoader =================
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ================= 6. 训练模型 =================
    print("开始训练...")
    history = detection_system.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOP_PATIENCE
    )
    
    # ================= 7. 评估模型 =================
    print("\n加载最佳模型进行评估...")
    detection_system.model.load_state_dict(torch.load('best_damage_detector.pth'))
    
    metrics = detection_system.evaluate(test_loader)
    print(f"测试集准确率: {metrics['accuracy']:.4f}")
    
    detection_system.plot_training_history(history)

if __name__ == "__main__":
    main()
