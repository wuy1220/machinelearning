import numpy as np
import matplotlib
matplotlib.use('Agg') # 必须在import pyplot之前设置
import matplotlib.pyplot as plt
import h5py
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


# 全局绘图函数，用于多进程
def plot_single_feature_map(args):
    """
    多进程工作函数：处理单个文件并保存图像
    """
    file_path, output_base_dir, samples_to_extract, class_to_filter = args
    
    try:
        with h5py.File(file_path, 'r') as hf:
            feature_maps = hf['feature_maps'][:]  # (num_samples, 224, 224, 3)
            damage_class = int(hf['damage_class'][0])
            damaged_dofs = list(hf.attrs.get('damaged_dofs', []))
            severity_ratios = list(hf.attrs.get('severity_ratios', []))
            
            # 类别过滤
            if class_to_filter is not None and damage_class != class_to_filter:
                return None

            num_samples = feature_maps.shape[0]
            scenario_id = os.path.basename(file_path).replace('scenario_', '').replace('.h5', '')
            
            # 确定要提取的样本索引
            if samples_to_extract == 'all':
                indices = range(num_samples)
            elif isinstance(samples_to_extract, int):
                # 均匀采样或取中间值
                if num_samples == 0:
                    indices = []
                elif samples_to_extract == 1:
                    indices = [num_samples // 2] # 取最中间的一个
                else:
                    step = max(1, num_samples // samples_to_extract)
                    indices = range(0, num_samples, step)[:samples_to_extract]
            
            saved_files = []
            
            # 确定保存目录
            class_dirs = {
                0: '00_healthy',
                1: '01_light_damage',
                2: '02_moderate_damage',
                3: '03_severe_damage',
                4: '04_multi_damage'
            }
            target_dir = os.path.join(output_base_dir, class_dirs.get(damage_class, 'unknown'))
            os.makedirs(target_dir, exist_ok=True)
            
            for idx in indices:
                feat_map = feature_maps[idx]
                
                # 归一化
                img = feat_map.astype(np.float32)
                for c in range(3):
                    channel = img[:, :, c]
                    c_min, c_max = channel.min(), channel.max()
                    if c_max - c_min > 1e-10:
                        img[:, :, c] = (channel - c_min) / (c_max - c_min)
                img = np.clip(img, 0, 1)
                
                # 绘图
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img, origin='upper')
                ax.axis('off')
                
                # 添加简洁的标题信息
                title_text = f"C{damage_class} | DOFs:{damaged_dofs}"
                ax.set_title(title_text, fontsize=8, color='black', 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))
                
                filename = os.path.join(target_dir, f"s{scenario_id}_idx{idx:03d}.png")
                plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.05)
                plt.close()
                
                saved_files.append(filename)
                
            return {
                'scenario_id': scenario_id,
                'class': damage_class,
                'count': len(saved_files),
                'example_file': saved_files[0] if saved_files else None
            }
            
    except Exception as e:
        return {'error': str(e), 'file': file_path}


class FastGVRImageExtractor:
    """
    优化的GVR特征图提取器
    """
    
    def __init__(self, data_dir, output_dir='./extracted_gvr_fast'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.class_names = {
            0: '健康',
            1: '轻度',
            2: '中度',
            3: '重度',
            4: '多处'
        }

    def extract(self, 
                n_workers=4, 
                samples_per_file=1, 
                max_files=None,
                class_filter=None):
        """
        执行提取
        
        Args:
            n_workers: 进程数，建议设置为CPU核心数
            samples_per_file: 每个文件提取几张图？
                              1 = 只提取中间最典型的一张 (推荐)
                              'all' = 提取所有 (慢)
            max_files: 最大处理文件数，用于快速测试
            class_filter: 只提取特定类别 (0-4)，None表示全部
        """
        h5_files = sorted(glob(os.path.join(self.data_dir, 'scenario_*.h5')))
        if max_files:
            h5_files = h5_files[:max_files]
            
        if not h5_files:
            print("未找到HDF5文件")
            return

        print(f"找到 {len(h5_files)} 个文件")
        print(f"启动 {n_workers} 个进程并行处理...")
        print(f"策略: 每个文件提取 {samples_per_file} 个样本")
        
        # 准备参数
        tasks = [
            (f, self.output_dir, samples_per_file, class_filter) 
            for f in h5_files
        ]
        
        # 多进程处理
        results = []
        with Pool(n_workers) as p:
            with tqdm(total=len(tasks), desc="提取进度") as pbar:
                for res in p.imap_unordered(plot_single_feature_map, tasks):
                    if res:
                        results.append(res)
                    pbar.update(1)
        
        # 统计
        self._print_stats(results)
        
        # 生成概览图
        if samples_per_file > 0:
            print("\n生成类别概览图...")
            self.create_summary_grid(results, output_filename='summary_overview.png')

    def _print_stats(self, results):
        valid_res = [r for r in results if 'error' not in r]
        errors = [r for r in results if 'error' in r]
        
        print("\n" + "="*50)
        print(f"处理完成: 成功 {len(valid_res)} 个, 失败 {len(errors)} 个")
        
        dist = {}
        for r in valid_res:
            cls = r['class']
            dist[cls] = dist.get(cls, 0) + r['count']
            
        print("提取图像统计:")
        for cls in sorted(dist.keys()):
            print(f"  - 类别 {cls} ({self.class_names.get(cls, '未知')}): {dist[cls]} 张")
        if errors:
            print(f"\n错误列表 (前3个):")
            for e in errors[:3]:
                print(f"  {e}")
        print("="*50)

    def create_summary_grid(self, results, output_filename, max_per_class=10):
        """
        生成一张大图，展示各类别的代表性样本
        """
        # 按类别分组
        class_samples = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        for r in results:
            if 'example_file' in r and r['example_file']:
                class_samples[r['class']].append(r['example_file'])
        
        # 限制每个类别的数量
        for k in class_samples:
            if len(class_samples[k]) > max_per_class:
                class_samples[k] = class_samples[k][:max_per_class]
        
        # 计算网格大小
        classes_to_show = [k for k, v in class_samples.items() if len(v) > 0]
        num_classes = len(classes_to_show)
        if num_classes == 0:
            return
            
        max_cols = max(len(class_samples[k]) for k in classes_to_show)
        
        fig, axes = plt.subplots(num_classes, max_cols, 
                                figsize=(max_cols * 1.5, num_classes * 1.5))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        if max_cols == 1:
            axes = axes.reshape(-1, 1)
            
        for row_idx, cls in enumerate(classes_to_show):
            samples = class_samples[cls]
            for col_idx in range(max_cols):
                ax = axes[row_idx, col_idx]
                ax.axis('off')
                
                if col_idx < len(samples):
                    try:
                        img = plt.imread(samples[col_idx])
                        ax.imshow(img)
                    except:
                        pass
                
                # 设置行标签
                if col_idx == 0:
                    label = f"Class {cls}: {self.class_names.get(cls, '')} ({len(samples)})"
                    ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=30, va='center')

        plt.suptitle('GVR 特征图概览', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), dpi=450)
        plt.close()
        print(f"概览图已保存: {output_filename}")


if __name__ == "__main__":
    # ===== 配置区域 =====
    DATA_DIR = './jacket_damage_data_timespace'
    OUTPUT_DIR = './extracted_gvr_fast'
    
    # 核心优化参数
    SAMPLES_PER_FILE = 20  # 【关键优化】1=每个文件只取最中间1张，'all'=取全部(极慢)
    N_WORKERS = 8          # 【关键优化】并行进程数，建议设为CPU核心数
    MAX_FILES = 100      # None=处理所有文件, 50=只处理前50个(测试用)
    # ===================

    extractor = FastGVRImageExtractor(DATA_DIR, OUTPUT_DIR)
    
    extractor.extract(
        n_workers=N_WORKERS,
        samples_per_file=SAMPLES_PER_FILE,
        max_files=MAX_FILES
    )
