import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
import h5py
import os
import re
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import json
import random

# ============================================================
# 新增：ANSYS数据加载器
# ============================================================
class ANSYSDataLoader:
    """
    从外部目录加载ANSYS仿真加速度数据
    """
    def __init__(self, data_root: str = './ansys_data', num_degrees: int = 15, num_steps: int = 30000):
        """
        Args:
            data_root: ansys_data根目录
            num_degrees: 传感器/通道数量 (15)
            num_steps: 采样点数量 (30000)
        """
        self.data_root = data_root
        self.num_degrees = num_degrees
        self.num_steps = num_steps
        self.dt = 0.001  # 根据tree.txt中的时间步长推断
        self.healthy_folder_name = '无损'
        
        # 检查目录是否存在
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"数据目录不存在: {self.data_root}")

    def _parse_folder_name(self, folder_name: str):
        """
        从文件夹名称解析损伤信息
        例如: "3号30%损伤" -> {3: 0.3}
        例如: "4号40%+8号40%损伤" -> {4: 0.4, 8: 0.4}
        """
        damaged_dofs = []
        severity_ratios = []
        
        # 移除"损伤"后缀
        temp_name = folder_name.replace("损伤", "")
        
        # 分割多个损伤 (如 "4号40%+8号40%")
        segments = temp_name.split('+')
        
        for seg in segments:
            # 正则匹配 "数字号数字%"
            match = re.search(r'(\d+)号(\d+)%', seg)
            if match:
                dof = int(match.group(1))
                severity = float(match.group(2)) / 100.0
                damaged_dofs.append(dof)
                severity_ratios.append(severity)
        
        return damaged_dofs, severity_ratios

    def load_single_file(self, file_path: str):
        """
        读取单个ANSYS导出的txt文件
        格式: 索引 时间 加速度
        跳过第一行表头
        """
        try:
            # 使用numpy读取，跳过第一行
            # 假设列之间用空白字符分隔
            data = np.loadtxt(file_path, skiprows=1)
            # data shape: (N, 3) or (N, 2)
            # 我们只需要加速度值 (最后一列)
            # 如果数据不足30000点，进行截断或填充? 假设都是完整的
            acc = data[:, -1]
            
            # 确保长度一致
            if len(acc) > self.num_steps:
                acc = acc[:self.num_steps]
            elif len(acc) < self.num_steps:
                # 如果数据不足，进行边缘填充
                acc = np.pad(acc, (0, self.num_steps - len(acc)), 'edge')
                
            return acc
        except Exception as e:
            print(f"读取文件错误 {file_path}: {e}")
            return np.zeros(self.num_steps)

    def load_scenario(self, folder_name: str) -> Dict:
        """
        加载特定场景的所有通道数据
        """
        folder_path = os.path.join(self.data_root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"目录不存在，跳过: {folder_path}")
            return None

        # 初始化响应矩阵
        response = np.zeros((self.num_steps, self.num_degrees))
        
        # 遍历1到15个通道
        # 假设文件名格式为: "文件夹名1.txt", "文件夹名2.txt" ...
        # 或者是 "无损1.txt" 等
        found_files = 0
        for i in range(1, self.num_degrees + 1):
            # 尝试匹配文件名，例如 "无损1.txt" 或 "3号30%损伤1.txt"
            # 注意：tree.txt中显示文件名包含文件夹名前缀
            possible_names = [
                f"{folder_name}{i}.txt",
                f"{i}.txt"
            ]
            
            file_path = None
            for name in possible_names:
                full_path = os.path.join(folder_path, name)
                if os.path.exists(full_path):
                    file_path = full_path
                    break
            
            if file_path:
                response[:, i-1] = self.load_single_file(file_path)
                found_files += 1
            else:
                print(f"警告: 场景 {folder_name} 中找不到通道 {i} 的数据文件")
        
        if found_files == 0:
            print(f"错误: 场景 {folder_name} 中没有找到任何有效数据文件")
            return None

        # 解析标签
        if folder_name == self.healthy_folder_name:
            damaged_dofs = []
            severity_ratios = []
            damage_class = 0
        else:
            damaged_dofs, severity_ratios = self._parse_folder_name(folder_name)
            damage_class = 1 if len(damaged_dofs) > 0 else 0

        return {
            'acceleration': response,
            'damaged_dofs': damaged_dofs,
            'severity_ratios': severity_ratios,
            'damage_class': damage_class,
            'folder_name': folder_name
        }

# ============================================================
# 保持不变：特征提取器
# ============================================================
class TimeStackedGVRFeatureExtractor:
    """
    时序堆叠GVR特征提取器
    """
    def __init__(self, dt, window_length=3000, step_size=50, 
                 num_stack_windows=100, cutoff_freq=5.0):
        self.dt = dt
        self.window_length = window_length
        self.step_size = step_size
        self.num_stack_windows = num_stack_windows
        self.cutoff_freq = cutoff_freq
        nyquist = 0.5 / self.dt
        self.b, self.a = signal.butter(4, cutoff_freq / nyquist, btype='low')
    
    def butterworth_filter(self, data):
        return signal.filtfilt(self.b, self.a, data, axis=0)
    
    def compute_damage_index(self, damaged_signal, healthy_signal):
        num_channels = damaged_signal.shape[1]
        DI = np.zeros(num_channels)
        for ch in range(num_channels):
            numerator = np.sum((damaged_signal[:, ch] - healthy_signal[:, ch]) ** 2)
            denominator = np.sum(healthy_signal[:, ch] ** 2)
            epsilon = 1e-10
            if denominator > epsilon:
                DI[ch] = np.sqrt(numerator) / np.sqrt(denominator)
            else:
                DI[ch] = 0.0
        return DI
    
    def compute_gvr(self, DI_series):
        DI_prime = np.zeros_like(DI_series)
        if DI_series.shape[0] > 1:
            DI_prime[1:] = DI_series[1:] - DI_series[:-1]
        DI_double_prime = np.zeros_like(DI_series)
        if DI_prime.shape[0] > 1:
            DI_double_prime[1:] = np.abs(DI_prime[1:] - DI_prime[:-1])
        return DI_prime, DI_double_prime
    
    def extract_gvr_features(self, damaged_signal, healthy_signal):
        filtered_damaged = self.butterworth_filter(damaged_signal)
        filtered_healthy = self.butterworth_filter(healthy_signal)
        
        if np.allclose(filtered_damaged, filtered_healthy, atol=1e-10):
            noise_level = 1e-6
            signal_std = np.std(filtered_healthy) if np.std(filtered_healthy) > 0 else 1.0
            filtered_healthy_noisy = filtered_healthy + \
                np.random.randn(*filtered_healthy.shape) * noise_level * signal_std
        else:
            filtered_healthy_noisy = filtered_healthy
        
        num_steps = filtered_damaged.shape[0]
        num_windows = (num_steps - self.window_length) // self.step_size + 1
        
        DI_series = []
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_length
            window_damaged = filtered_damaged[start_idx:end_idx]
            window_healthy = filtered_healthy_noisy[start_idx:end_idx] if end_idx <= filtered_healthy_noisy.shape[0] else window_damaged
            
            if window_damaged.shape[0] < self.window_length:
                window_damaged = np.pad(window_damaged, 
                                       ((0, self.window_length - window_damaged.shape[0]), (0, 0)), 
                                       'edge')
                window_healthy = np.pad(window_healthy,
                                       ((0, self.window_length - window_healthy.shape[0]), (0, 0)),
                                       'edge')
            
            DI_window = self.compute_damage_index(window_damaged, window_healthy)
            DI_series.append(DI_window)
        
        DI_series = np.array(DI_series)
        DI_prime, DI_double_prime = self.compute_gvr(DI_series)
        return {'DI': DI_series, 'GVR_prime': DI_prime, 'GVR_double_prime': DI_double_prime}
    
    def generate_intra_window_feature_maps(self, damaged_signal, healthy_signal, 
                                          image_size=(224, 224), 
                                          intra_window_segments=56):
        filt_damaged = self.butterworth_filter(damaged_signal)
        filt_healthy = self.butterworth_filter(healthy_signal)
        
        if np.allclose(filt_damaged, filt_healthy, atol=1e-10):
             noise_level = 1e-6
             std = np.std(filt_healthy) if np.std(filt_healthy) > 0 else 1.0
             filt_healthy = filt_healthy + np.random.randn(*filt_healthy.shape) * noise_level * std

        total_steps = filt_damaged.shape[0]
        num_windows = (total_steps - self.window_length) // self.step_size + 1
        
        feature_maps = []
        num_channels = filt_damaged.shape[1]
        
        x_sensor = np.arange(num_channels)
        x_pixel = np.linspace(0, num_channels - 1, image_size[1])
        y_seg = np.arange(intra_window_segments)
        y_pixel = np.linspace(0, intra_window_segments - 1, image_size[0])

        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_length
            win_d = filt_damaged[start:end]
            win_h = filt_healthy[start:end]
            
            actual_len = win_d.shape[0]
            segment_len = actual_len // intra_window_segments
            if segment_len < 1:
                segment_len = 1
                intra_window_segments = actual_len
            
            trim_len = segment_len * intra_window_segments
            segs_d = win_d[:trim_len].reshape(intra_window_segments, segment_len, num_channels)
            segs_h = win_h[:trim_len].reshape(intra_window_segments, segment_len, num_channels)
            
            diff_power = np.sum((segs_d - segs_h)**2, axis=1)
            di_val = np.sqrt(diff_power)
            healthy_power = np.sum(segs_h**2, axis=1)
            healthy_norm = np.sqrt(healthy_power)
            di_matrix = di_val / (healthy_norm + 1e-10)
            
            di_prime_matrix = np.zeros_like(di_matrix)
            di_prime_matrix[1:] = di_matrix[1:] - di_matrix[:-1]
            di_double_prime_matrix = np.zeros_like(di_prime_matrix)
            di_double_prime_matrix[1:] = np.abs(di_prime_matrix[1:] - di_prime_matrix[:-1])
            
            def normalize_2d(mat):
                min_v = mat.min(axis=0)
                max_v = mat.max(axis=0)
                range_v = max_v - min_v
                range_v[range_v < 1e-10] = 1.0
                return (mat - min_v) / range_v
            
            norm_r = normalize_2d(di_prime_matrix)
            norm_g = normalize_2d(di_double_prime_matrix)
            norm_b = normalize_2d(di_matrix)
            
            img_r = np.zeros((intra_window_segments, image_size[1]))
            img_g = np.zeros((intra_window_segments, image_size[1]))
            img_b = np.zeros((intra_window_segments, image_size[1]))
            
            for t in range(intra_window_segments):
                img_r[t] = np.interp(x_pixel, x_sensor, norm_r[t])
                img_g[t] = np.interp(x_pixel, x_sensor, norm_g[t])
                img_b[t] = np.interp(x_pixel, x_sensor, norm_b[t])
            
            img_r_full = np.zeros((image_size[0], image_size[1]))
            img_g_full = np.zeros((image_size[0], image_size[1]))
            img_b_full = np.zeros((image_size[0], image_size[1]))
            
            for x in range(image_size[1]):
                img_r_full[:, x] = np.interp(y_pixel, y_seg, img_r[:, x])
                img_g_full[:, x] = np.interp(y_pixel, y_seg, img_g[:, x])
                img_b_full[:, x] = np.interp(y_pixel, y_seg, img_b[:, x])
            
            img_rgb = np.stack([img_r_full, img_g_full, img_b_full], axis=2)
            feature_maps.append(img_rgb)
            
        return np.array(feature_maps, dtype=np.float32)

# ============================================================
# 修改后的数据生成器
# ============================================================
class ImprovedDamageDataGenerator:
    """
    改进的损伤数据生成器
    适配外部ANSYS数据加载
    """
    
    def __init__(self, data_loader, gvr_extractor, output_dir='./jacket_damage_data_ansys'):
        self.data_loader = data_loader
        self.gvr_extractor = gvr_extractor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = []
        self.num_degrees = data_loader.num_degrees
    
    def _write_scenario_to_group(self, hf, group_name, data_dict):
        grp = hf.create_group(group_name)
        grp.create_dataset('acceleration', data=data_dict['acceleration'].astype(np.float32), 
                          compression='gzip', compression_opts=4)
        grp.create_dataset('feature_maps', data=data_dict['feature_maps'].astype(np.float32), 
                          compression='gzip', compression_opts=4)
        grp.create_dataset('labels', data=data_dict['labels'].astype(np.uint8))
        grp.create_dataset('damage_class', data=np.array([data_dict['damage_class']], dtype=np.uint8))
        grp.attrs['damaged_dofs'] = np.array(data_dict['damaged_dofs'])
        grp.attrs['severity_ratios'] = np.array(data_dict['severity_ratios'])
        grp.attrs['folder_name'] = data_dict['folder_name']
        grp.attrs['window_length'] = self.gvr_extractor.window_length
        grp.attrs['step_size'] = self.gvr_extractor.step_size
        grp.attrs['num_stack_windows'] = self.gvr_extractor.num_stack_windows

    def generate_from_directory(self, scenarios_per_shard=200):
        """
        扫描目录并生成HDF5数据集
        """
        # 1. 首先加载健康数据 (只加载一次)
        print("正在加载健康基准数据...")
        healthy_data_dict = self.data_loader.load_scenario(self.data_loader.healthy_folder_name)
        if healthy_data_dict is None:
            raise ValueError(f"无法在 {self.data_loader.data_root} 中找到名为 '{self.data_loader.healthy_folder_name}' 的健康数据文件夹")
        
        healthy_response = healthy_data_dict['acceleration']
        print(f"健康数据加载完成，形状: {healthy_response.shape}")

        # 2. 获取所有场景文件夹
        all_folders = [f for f in os.listdir(self.data_loader.data_root) 
                       if os.path.isdir(os.path.join(self.data_loader.data_root, f))]
        
        # 确保处理顺序，健康数据放在最前 (可选，但通常需要)
        if self.data_loader.healthy_folder_name in all_folders:
            all_folders.remove(self.data_loader.healthy_folder_name)
            # 将健康数据也作为一个场景处理
            all_folders.insert(0, self.data_loader.healthy_folder_name)
        
        # 过滤掉可能存在的非场景目录 (如__pycache__)
        valid_folders = []
        for f in all_folders:
            # 简单的过滤逻辑：确保是文件夹
            valid_folders.append(f)
            
        print(f"发现 {len(valid_folders)} 个场景文件夹")

        # 3. 遍历场景处理
        current_shard_idx = 0
        scenarios_in_current_shard = 0
        hf = None

        for folder_name in tqdm(valid_folders, desc="处理场景"):
            # 检查是否需要创建新分片
            if scenarios_in_current_shard == 0:
                if hf is not None:
                    hf.close()
                shard_path = os.path.join(self.output_dir, f'data_shard_{current_shard_idx:04d}.h5')
                print(f"\n创建新分片: {shard_path}")
                hf = h5py.File(shard_path, 'w')
            
            # 加载当前场景数据
            scenario_data = self.data_loader.load_scenario(folder_name)
            if scenario_data is None:
                continue
                
            damaged_response = scenario_data['acceleration']
            
            # 如果是健康场景，将 damaged_response 设为 healthy_response (或直接处理)
            # 为了统一逻辑，即使是健康场景，我们也计算特征
            # 此时 healthy_response == damaged_response (理论上)
            # 但为了计算特征图，函数期望两个输入
            
            # 提取特征
            feature_maps = self.gvr_extractor.generate_intra_window_feature_maps(
                damaged_response, 
                healthy_response, 
                intra_window_segments=56 
            )
            
            num_samples = feature_maps.shape[0]
            
            # 生成标签
            labels = np.zeros(self.num_degrees, dtype=int)
            damaged_dofs = scenario_data['damaged_dofs']
            if damaged_dofs:
                labels[np.array(damaged_dofs)] = 1
            labels_array = np.tile(labels, (num_samples, 1))
            
            damage_class = scenario_data['damage_class']

            # 计算加速度保存长度
            total_steps_needed = (num_samples - 1) * self.gvr_extractor.step_size + \
                               self.gvr_extractor.window_length
            acc_to_save = damaged_response[:total_steps_needed]
            
            data_dict = {
                'acceleration': acc_to_save,
                'feature_maps': feature_maps,
                'labels': labels_array[0],
                'damage_class': damage_class,
                'damaged_dofs': damaged_dofs,
                'severity_ratios': scenario_data['severity_ratios'],
                'folder_name': folder_name,
                'num_samples': num_samples
            }
            
            group_name = f'scenario_{folder_name}'
            self._write_scenario_to_group(hf, group_name, data_dict)
            
            self.metadata.append({
                'scenario_id': folder_name,
                'shard_id': current_shard_idx,
                'group_name': group_name,
                'damaged_dofs': damaged_dofs,
                'severity_ratios': scenario_data['severity_ratios'],
                'damage_class': damage_class,
                'num_samples': num_samples
            })
            
            scenarios_in_current_shard += 1
            if scenarios_in_current_shard >= scenarios_per_shard:
                hf.close()
                hf = None
                scenarios_in_current_shard = 0
                current_shard_idx += 1

        if hf is not None:
            hf.close()
            
        self._save_metadata()
        print("\n所有场景数据处理完成！")

    def _save_metadata(self):
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)


if __name__ == "__main__":
    # ===== 参数配置 =====
    # 根据实际情况修改路径
    data_root = './ansys_data' 
    
    # 初始化ANSYS数据加载器
    print("=" * 60)
    print("初始化ANSYS数据加载器")
    print("=" * 60)
    loader = ANSYSDataLoader(
        data_root=data_root,
        num_degrees=15,
        num_steps=30000
    )
    
    # 初始化时序堆叠GVR特征提取器
    print("\n初始化时序堆叠GVR特征提取器")
    print("=" * 60)
    gvr_extractor = TimeStackedGVRFeatureExtractor(
        dt=loader.dt,  # 使用加载器中的dt
        window_length=3000,          # 单窗口长度
        step_size=50,                # 滑动步长
        num_stack_windows=112,       # 堆叠窗口数（=图像高度）
        cutoff_freq=8.0             # 滤波截止频率
    )
    
    print(f"配置参数:")
    print(f"  - 数据目录: {data_root}")
    print(f"  - 单窗口长度: {gvr_extractor.window_length} 点")
    print(f"  - 滑动步长: {gvr_extractor.step_size} 点")
    print(f"  - 堆叠窗口数: {gvr_extractor.num_stack_windows} (图像高度)")
    
    # 初始化数据生成器
    print("\n初始化数据生成器")
    print("=" * 60)
    generator = ImprovedDamageDataGenerator(
        data_loader=loader,
        gvr_extractor=gvr_extractor,
        output_dir='./jacket_damage_data_ansys'
    )
    
    # 生成数据集
    print("\n开始从外部目录生成数据集")
    print("=" * 60)
    generator.generate_from_directory(
        scenarios_per_shard=200  # 每个h5文件包含200个场景
    )
    
    print("\n" + "=" * 60)
    print("数据生成完毕！")
    print("=" * 60)
