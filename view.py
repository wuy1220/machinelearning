import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def extract_initial_samples(data_dir, num_samples_per_class=3, num_points=200):
    """
    从生成的数据集中提取正常和异常样本的前200个采样点
    
    Args:
        data_dir (str): 数据集根目录 (如 './jacket_damage_data_timespace3')
        num_samples_per_class (int): 每种类别要提取的样本数量
        num_points (int): 提取的采样点数
        
    Returns:
        dict: 包含提取数据的字典 {'healthy': [...], 'damaged': [...]}
    """
    
    # 1. 加载元数据以区分健康和受损样本
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"未找到元数据文件: {metadata_path}")

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    # 分类样本
    healthy_samples_meta = [m for m in all_metadata if m['damage_class'] == 0]
    damaged_samples_meta = [m for m in all_metadata if m['damage_class'] == 1]

    print(f"数据集概况:")
    print(f"  总样本数: {len(all_metadata)}")
    print(f"  健康样本数: {len(healthy_samples_meta)}")
    print(f"  受损样本数: {len(damaged_samples_meta)}")

    # 检查是否有足够的样本
    if len(healthy_samples_meta) < num_samples_per_class:
        print(f"警告: 健康样本不足 {num_samples_per_class} 个，将提取全部 {len(healthy_samples_meta)} 个")
    if len(damaged_samples_meta) < num_samples_per_class:
        print(f"警告: 受损样本不足 {num_samples_per_class} 个，将提取全部 {len(damaged_samples_meta)} 个")

    selected_healthy = random.sample(healthy_samples_meta, min(num_samples_per_class, len(healthy_samples_meta)))
    selected_damaged = random.sample(damaged_samples_meta, min(num_samples_per_class, len(damaged_samples_meta)))

    extracted_data = {
        'healthy': [],
        'damaged': []
    }

    # 2. 提取健康样本数据
    print("\n正在提取健康样本...")
    for meta in selected_healthy:
        shard_file = os.path.join(data_dir, f"data_shard_{meta['shard_id']:04d}.h5")
        
        if not os.path.exists(shard_file):
            print(f"  错误: 文件不存在 {shard_file}")
            continue
            
        with h5py.File(shard_file, 'r') as hf:
            group = hf[meta['group_name']]
            # 加载全部加速度数据，然后切片
            full_acc = group['acceleration'][:]
            
            # 提取前200个点，形状为 (200, num_degrees)
            snippet = full_acc[:num_points, :]
            
            extracted_data['healthy'].append({
                'scenario_id': meta['scenario_id'],
                'data': snippet,
                'damaged_dofs': meta['damaged_dofs']
            })
            print(f"  - 场景 {meta['scenario_id']}: 数据形状 {snippet.shape}")

    # 3. 提取受损样本数据
    print("\n正在提取受损样本...")
    for meta in selected_damaged:
        shard_file = os.path.join(data_dir, f"data_shard_{meta['shard_id']:04d}.h5")
        
        if not os.path.exists(shard_file):
            print(f"  错误: 文件不存在 {shard_file}")
            continue
            
        with h5py.File(shard_file, 'r') as hf:
            group = hf[meta['group_name']]
            full_acc = group['acceleration'][:]
            snippet = full_acc[:num_points, :]
            
            extracted_data['damaged'].append({
                'scenario_id': meta['scenario_id'],
                'data': snippet,
                'damaged_dofs': meta['damaged_dofs']
            })
            print(f"  - 场景 {meta['scenario_id']}: 数据形状 {snippet.shape}, 受损位置: {meta['damaged_dofs']}")

    return extracted_data

def plot_comparison(extracted_data, sensor_idx=0):
    """
    简单的可视化函数，对比健康和受损样本在特定传感器上的前200个点
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制健康样本
    for i, item in enumerate(extracted_data['healthy']):
        plt.plot(item['data'][:, sensor_idx], label=f"健康样本 {i+1} (ID={item['scenario_id']})", linestyle='--')
        
    # 绘制受损样本
    for i, item in enumerate(extracted_data['damaged']):
        plt.plot(item['data'][:, sensor_idx], label=f"受损样本 {i+1} (DOFs={item['damaged_dofs']})", alpha=0.7)
        
    plt.title(f"前{extracted_data['healthy'][0]['data'].shape[0]}个采样点加速度对比 (传感器索引: {sensor_idx})")
    plt.xlabel("采样点")
    plt.ylabel("加速度")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 配置路径，需与生成代码中的 output_dir 一致
    DATA_DIRECTORY = './jacket_damage_data_timespace4'
    
    # 提取数据：每类取 3 个样本，取前 200 个点
    data = extract_initial_samples(
        data_dir=DATA_DIRECTORY,
        num_samples_per_class=6,
        num_points=1000
    )
    
    # 示例：打印第一个健康样本的前5个时间点的数据 (仅第一个传感器)
    if data['healthy']:
        print("\n示例数据 (第一个健康样本, 前5个点, 传感器0):")
        print(data['healthy'][0]['data'][:5, 0])

    # 示例：绘制对比图 (对比传感器 0 的数据)
    # 注意：运行此行需要图形界面支持

    plot_comparison(data, sensor_idx=0)
    plot_comparison(data, sensor_idx=1)
    plot_comparison(data, sensor_idx=4)
    plot_comparison(data, sensor_idx=8)
    plot_comparison(data, sensor_idx=12)
    plot_comparison(data, sensor_idx=15)