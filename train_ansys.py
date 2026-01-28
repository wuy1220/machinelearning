import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py

# 导入您现有的模型和数据集类
from model1_mn_cnn import OffshoreDamageDetectionSystem
from h5_gvr_dataset import H5GVRDataset 

def main():
   # ================= 配置参数 =================
    DATA_DIR = './jacket_damage_data_timespace3'  # 您存放 HDF5 的目录
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ================= 1. 初始化数据集 =================
    full_dataset = H5GVRDataset(data_dir=DATA_DIR, window_length=2000, transform=None)
    
    # ================= 2. 构建场景级索引 (关键步骤) =================
    # 我们不能直接对样本索引进行划分，必须先按“场景”聚合
    
    scenarios_info = []  # 存储: [{'group': 'scenario_000000', 'label': 0, 'start_idx': 0, 'end_idx': 120}, ...]
    
    print("正在构建场景索引以防止数据泄露...")
    
    current_idx = 0
    # 遍历元数据
    for i in range(len(full_dataset)):
        meta = full_dataset.sample_metadata[i]
        
        # 检查是否是新场景（利用 group_name 判断）
        # 由于 sample_metadata 是按顺序排的，如果 group 变了，就是一个新场景
        if i == 0:
            current_group = meta['group']
            start_idx = i
            current_label = meta['label']
        elif meta['group'] != current_group:
            # 场景结束，保存上一个场景的信息
            scenarios_info.append({
                'group': current_group,
                'label': current_label,
                'start': start_idx,
                'end': i  # 不包含 i，因为是左闭右开
            })
            # 开启新场景
            current_group = meta['group']
            start_idx = i
            current_label = meta['label']
            
        # 处理最后一个场景
        if i == len(full_dataset) - 1:
             scenarios_info.append({
                'group': current_group,
                'label': current_label,
                'start': start_idx,
                'end': i + 1
            })

    print(f"共发现 {len(scenarios_info)} 个独立的损伤场景")

    # ================= 3. 场景级划分 (Stratified Split) =================
    # 提取场景索引和标签
    scenario_indices = np.arange(len(scenarios_info))
    scenario_labels = [s['label'] for s in scenarios_info]
    
    # 对场景进行划分，而不是对样本
    train_scenarios_idx, temp_scenarios_idx = train_test_split(
        scenario_indices,
        test_size=0.4, 
        random_state=42,
        stratify=scenario_labels  # 保证训练集和测试集的健康/损伤比例一致
    )
    
    val_scenarios_idx, test_scenarios_idx = train_test_split(
        temp_scenarios_idx,
        test_size=0.5,
        random_state=42,
        stratify=[scenario_labels[i] for i in temp_scenarios_idx]
    )
    
    # ================= 4. 将场景索引还原为样本索引 =================
    def get_sample_indices(scenario_idx_list):
        indices = []
        for s_idx in scenario_idx_list:
            s_info = scenarios_info[s_idx]
            # 将该场景包含的所有样本索引加入列表
            indices.extend(range(s_info['start'], s_info['end']))
        return indices
    
    train_idx = get_sample_indices(train_scenarios_idx)
    val_idx = get_sample_indices(val_scenarios_idx)
    test_idx = get_sample_indices(test_scenarios_idx)
    
    print(f"划分结果 (场景级):")
    print(f"  训练集: {len(train_scenarios_idx)} 个场景 -> {len(train_idx)} 个样本")
    print(f"  验证集: {len(val_scenarios_idx)} 个场景 -> {len(val_idx)} 个样本")
    print(f"  测试集: {len(test_scenarios_idx)} 个场景 -> {len(test_idx)} 个样本")

    # ================= 5. 创建 DataLoader =================
    detection_system = OffshoreDamageDetectionSystem(num_classes=2, device=DEVICE)
    
    # 设置 Transform
    full_dataset.transform = detection_system.train_transform
    
    # 创建 Subset
    train_dataset = Subset(full_dataset, train_idx)
    
    # 验证和测试集需要重新实例化 Dataset 或修改 transform 属性，
    # 因为全量 Dataset 已经被设置了 train_transform（包含数据增强）
    # 为了避免麻烦，这里简单克隆两个 Dataset 用于验证和测试
    val_dataset_base = H5GVRDataset(DATA_DIR, window_length=2000)
    test_dataset_base = H5GVRDataset(DATA_DIR, window_length=2000)
    val_dataset_base.transform = detection_system.valid_transform
    test_dataset_base.transform = detection_system.valid_transform
    
    val_dataset = Subset(val_dataset_base, val_idx)
    test_dataset = Subset(test_dataset_base, test_idx)
    
    # ================= 4. 创建 DataLoader =================
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ================= 5. 训练模型 =================
    print("开始训练...")
    history = detection_system.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=10
    )
    
    # ================= 6. 评估模型 =================
    print("\n加载最佳模型进行评估...")
    detection_system.model.load_state_dict(torch.load('best_damage_detector.pth'))
    
    metrics = detection_system.evaluate(test_loader)
    print(f"测试集准确率: {metrics['accuracy']:.4f}")
    
    detection_system.plot_training_history(history)

if __name__ == "__main__":
    main()