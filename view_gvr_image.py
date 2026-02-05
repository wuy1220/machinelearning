import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_gvr_channels(data_dir='./jacket_damage_data_ansys', 
                         output_dir='./gvr_channels_split',
                         save_interval=1):
    """
    将GVR图像的RGB三个通道分别提取并保存为独立的灰度图像。

    Args:
        data_dir: 包含生成的 .h5 分片文件的目录路径
        output_dir: 保存通道图片的根目录
        save_interval: 保存图片的间隔
    """
    if not os.path.exists(data_dir):
        print(f"错误：数据目录不存在: {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    
    if not h5_files:
        print(f"在目录 {data_dir} 中未找到 .h5 文件。")
        return

    print(f"找到 {len(h5_files)} 个数据分片，开始提取通道...")

    # 定义通道配置
    # 通道名，物理含义描述，数据索引
    channel_configs = [
        {
            'name': 'red_DI_prime', 
            'desc': 'Red_Channel (Damage Variation Rate)', 
            'idx': 0
        },
        {
            'name': 'green_DI_double_prime', 
            'desc': 'Green_Channel (Damage Acceleration)', 
            'idx': 1
        },
        {
            'name': 'blue_DI_raw', 
            'desc': 'Blue_Channel (Raw Normalized DI)', 
            'idx': 2
        }
    ]

    for h5_filename in tqdm(h5_files, desc="处理分片文件"):
        file_path = os.path.join(data_dir, h5_filename)
        
        try:
            with h5py.File(file_path, 'r') as hf:
                group_names = list(hf.keys())
                
                for group_name in group_names:
                    grp = hf[group_name]
                    
                    if 'feature_maps' not in grp:
                        continue
                        
                    feature_maps = grp['feature_maps'][:] # Shape: (N, 224, 224, 3)
                    
                    # 获取场景名称用于建文件夹
                    folder_name = grp.attrs.get('folder_name', group_name)
                    safe_folder_name = "".join([c for c in folder_name if c.isalnum() or c in (' ', '-', '_')]).strip()
                    if not safe_folder_name:
                        safe_folder_name = f"scenario_{group_name}"
                    
                    # 为每个场景创建总目录
                    scene_output_dir = os.path.join(output_dir, safe_folder_name)
                    os.makedirs(scene_output_dir, exist_ok=True)
                    
                    num_samples = feature_maps.shape[0]
                    
                    # 循环处理三个通道
                    for conf in channel_configs:
                        # 创建通道子目录
                        channel_subdir = os.path.join(scene_output_dir, conf['name'])
                        os.makedirs(channel_subdir, exist_ok=True)
                        
                        # 提取单通道数据
                        # feature_maps[..., idx] 得到 Shape: (N, 224, 224)
                        single_channel_data = feature_maps[:, :, :, conf['idx']]
                        
                        # 遍历时间步并保存
                        for i in range(0, num_samples, save_interval):
                            img_data = single_channel_data[i]
                            
                            # 确保数据范围
                            img_data = np.clip(img_data, 0, 1)
                            
                            # 构建文件名
                            img_filename = f"frame_{i:05d}.png"
                            save_path = os.path.join(channel_subdir, img_filename)
                            
                            # 使用 matplotlib 保存灰度图
                            # cmap='gray' 会将 0-1 的浮点数映射为黑白像素
                            plt.imsave(
                                save_path, 
                                img_data, 
                                cmap='gray', 
                                vmin=0, 
                                vmax=1
                            )
                            
        except Exception as e:
            print(f"处理文件 {h5_filename} 时发生错误: {e}")

    print(f"\n通道提取完成！文件已按通道分类保存至: {output_dir}")
    print("目录结构示例:")
    print(f"{output_dir}/")
    print(f"  └── {safe_folder_name}/")
    print(f"      ├── red_DI_prime/       (损伤变化率)")
    print(f"      ├── green_DI_double_prime/ (损伤变化加速度)")
    print(f"      └── blue_DI_raw/        (原始损伤指数)")

if __name__ == "__main__":
    # ================= 配置区域 =================
    INPUT_DATA_DIR = './jacket_damage_data_ansys'
    OUTPUT_DIR = './gvr_channels_split'
    
    # 保存间隔
    SAVE_INTERVAL = 50 
    # ===========================================

    extract_gvr_channels(
        data_dir=INPUT_DATA_DIR,
        output_dir=OUTPUT_DIR,
        save_interval=SAVE_INTERVAL
    )
