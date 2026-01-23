import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import os
from glob import glob
from tqdm import tqdm
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

class GVRImageExtractor:
    """
    GVR特征图提取器
    从HDF5文件中提取特征图并保存为图像
    """
    
    def __init__(self, data_dir, output_dir='./extracted_gvr_images'):
        """
        Args:
            data_dir: 数据集目录（包含HDF5文件）
            output_dir: 输出图像目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.create_subdirs()
    
    def create_subdirs(self):
        """创建输出子目录"""
        subdirs = [
            'all_samples',      # 所有样本
            'healthy',          # 健康样本
            'light_damage',     # 轻度损伤
            'moderate_damage',  # 中度损伤
            'severe_damage',    # 重度损伤
            'multi_damage',     # 多处损伤
            'channels_split'    # 分通道展示
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def get_damage_class_name(self, damage_class):
        """获取损伤类别名称"""
        class_names = {
            0: '健康',
            1: '轻度损伤',
            2: '中度损伤',
            3: '重度损伤',
            4: '多处损伤'
        }
        return class_names.get(damage_class, '未知')
    
    def get_damage_class_dir(self, damage_class):
        """获取损伤类别对应的子目录"""
        class_dirs = {
            0: 'healthy',
            1: 'light_damage',
            2: 'moderate_damage',
            3: 'severe_damage',
            4: 'multi_damage'
        }
        return class_dirs.get(damage_class, 'all_samples')
    
    def normalize_feature_map(self, feature_map):
        """归一化特征图到[0, 1]范围"""
        feature_map = feature_map.astype(np.float32)
        # 逐通道归一化
        for c in range(3):
            channel = feature_map[:, :, c]
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min > 1e-10:
                feature_map[:, :, c] = (channel - c_min) / (c_max - c_min)
        return np.clip(feature_map, 0, 1)
    
    def save_single_feature_map(self, feature_map, filename, 
                                title=None, add_colorbar=True):
        """
        保存单个特征图为RGB图像
        
        Args:
            feature_map: (H, W, 3) 特征图
            filename: 输出文件名
            title: 图像标题
            add_colorbar: 是否添加颜色条
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # 归一化
        img = self.normalize_feature_map(feature_map)
        
        ax.imshow(img, origin='upper')
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.set_xlabel('空间位置 (传感器)', fontsize=10)
        ax.set_ylabel('时间演进 (窗口)', fontsize=10)
        
        # 添加颜色图例
        legend_text = "R: DI' (趋势)\nG: DI'' (突变)\nB: DI (强度)"
        ax.text(0.02, 0.98, legend_text, 
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_channels_split(self, feature_map, filename, title=None):
        """
        将RGB三个通道分开保存
        
        Args:
            feature_map: (H, W, 3) 特征图
            filename: 输出文件名
            title: 图像标题
        """
        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
        
        # 归一化
        img = self.normalize_feature_map(feature_map)
        
        channel_names = ["DI' (变化趋势)", "DI'' (突变特征)", "DI (损伤强度)"]
        channel_titles = ['R Channel', 'G Channel', 'B Channel']
        cmaps = ['Reds', 'Greens', 'Blues']
        
        for i in range(3):
            ax = fig.add_subplot(gs[i])
            im = ax.imshow(img[:, :, i], cmap=cmaps[i], origin='upper', aspect='auto')
            ax.set_title(channel_names[i], fontsize=11, fontweight='bold')
            ax.set_xlabel('空间位置', fontsize=9)
            if i == 0:
                ax.set_ylabel('时间演进', fontsize=9)
        
        # 颜色条
        cax = fig.add_subplot(gs[3])
        plt.colorbar(im, cax=cax, label='归一化强度')
        
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def extract_from_file(self, h5_file, save_all=True, save_split=False):
        """
        从单个HDF5文件中提取特征图
        
        Args:
            h5_file: HDF5文件路径
            save_all: 是否保存到all_samples目录
            save_split: 是否保存分通道图像
            
        Returns:
            提取的信息字典
        """
        file_info = {
            'filename': os.path.basename(h5_file),
            'damage_class': None,
            'num_samples': 0,
            'saved_images': []
        }
        
        try:
            with h5py.File(h5_file, 'r') as hf:
                # 读取特征图
                feature_maps = hf['feature_maps'][:]  # (num_samples, 224, 224, 3)
                
                # 读取损伤类别
                damage_class = hf['damage_class'][0]
                file_info['damage_class'] = int(damage_class)
                
                # 读取标签
                labels = hf['labels'][:]
                damaged_dofs = list(hf.attrs.get('damaged_dofs', []))
                severity_ratios = list(hf.attrs.get('severity_ratios', []))
                
                num_samples = feature_maps.shape[0]
                file_info['num_samples'] = num_samples
                
                # 获取场景ID
                scenario_id = os.path.basename(h5_file).replace('scenario_', '').replace('.h5', '')
                
                # 保存每个样本
                for sample_idx in range(num_samples):
                    feature_map = feature_maps[sample_idx]
                    
                    class_name = self.get_damage_class_name(damage_class)
                    class_dir = self.get_damage_class_dir(damage_class)
                    
                    # 构建文件名
                    base_filename = f"scenario_{scenario_id}_sample_{sample_idx:03d}"
                    
                    if save_all:
                        # 保存到all_samples目录
                        filename_all = os.path.join(
                            self.output_dir, 'all_samples', 
                            f"{base_filename}_class{damage_class}.png"
                        )
                        title = (f"Scenario {scenario_id} | Sample {sample_idx} | "
                                f"{class_name}\n损伤位置: {damaged_dofs}, "
                                f"严重程度: {[f'{s:.2f}' for s in severity_ratios]}")
                        
                        self.save_single_feature_map(
                            feature_map, filename_all, title=title
                        )
                        file_info['saved_images'].append(filename_all)
                    
                    # 保存到对应类别目录
                    filename_class = os.path.join(
                        self.output_dir, class_dir,
                        f"{base_filename}.png"
                    )
                    self.save_single_feature_map(
                        feature_map, filename_class, 
                        title=f"Scenario {scenario_id} | Sample {sample_idx} | {class_name}"
                    )
                    file_info['saved_images'].append(filename_class)
                    
                    # 保存分通道图像
                    if save_split and sample_idx == 0:  # 每个文件只保存第一个样本的分通道图
                        filename_split = os.path.join(
                            self.output_dir, 'channels_split',
                            f"{base_filename}_channels.png"
                        )
                        self.save_channels_split(
                            feature_map, filename_split,
                            title=f"Scenario {scenario_id} | {class_name} | RGB通道分解"
                        )
                        file_info['saved_images'].append(filename_split)
                
        except Exception as e:
            print(f"错误: 处理文件 {h5_file} 时出错: {str(e)}")
            file_info['error'] = str(e)
        
        return file_info
    
    def extract_all_files(self, pattern='scenario_*.h5', 
                         save_all=True, save_split=False,
                         max_files=None):
        """
        从所有HDF5文件中提取特征图
        
        Args:
            pattern: 文件匹配模式
            save_all: 是否保存到all_samples目录
            save_split: 是否保存分通道图像
            max_files: 最大处理文件数（用于测试）
            
        Returns:
            提取统计信息
        """
        # 查找所有HDF5文件
        h5_files = glob(os.path.join(self.data_dir, pattern))
        h5_files.sort()
        
        if max_files:
            h5_files = h5_files[:max_files]
        
        if not h5_files:
            print(f"警告: 在 {self.data_dir} 中没有找到匹配 '{pattern}' 的文件")
            return None
        
        print(f"找到 {len(h5_files)} 个HDF5文件")
        print(f"开始提取GVR特征图...")
        
        # 统计信息
        stats = {
            'total_files': len(h5_files),
            'total_samples': 0,
            'class_distribution': {i: 0 for i in range(5)},
            'files_processed': [],
            'files_with_errors': []
        }
        
        # 处理每个文件
        for h5_file in tqdm(h5_files, desc="提取特征图"):
            file_info = self.extract_from_file(
                h5_file, 
                save_all=save_all, 
                save_split=save_split
            )
            
            stats['files_processed'].append(file_info)
            
            # 更新统计
            if 'error' not in file_info:
                stats['total_samples'] += file_info['num_samples']
                if file_info['damage_class'] is not None:
                    stats['class_distribution'][file_info['damage_class']] += 1
            else:
                stats['files_with_errors'].append(file_info['filename'])
        
        # 保存统计信息
        self._save_statistics(stats)
        
        # 打印摘要
        self._print_summary(stats)
        
        return stats
    
    def _save_statistics(self, stats):
        """保存统计信息到JSON文件"""
        import json
        
        stats_file = os.path.join(self.output_dir, 'extraction_stats.json')
        
        # 转换为可序列化格式
        serializable_stats = stats.copy()
        serializable_stats['files_processed'] = [
            {k: v for k, v in f.items() if k != 'saved_images'}
            for f in stats['files_processed']
        ]
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n统计信息已保存到: {stats_file}")
    
    def _print_summary(self, stats):
        """打印提取摘要"""
        print("\n" + "=" * 60)
        print("GVR特征图提取完成!")
        print("=" * 60)
        print(f"处理文件数: {stats['total_files']}")
        print(f"提取样本数: {stats['total_samples']}")
        
        print("\n各类别分布:")
        class_names = ['健康', '轻度损伤', '中度损伤', '重度损伤', '多处损伤']
        for i, count in stats['class_distribution'].items():
            if count > 0:
                print(f"  {class_names[i]}: {count} 个场景")
        
        if stats['files_with_errors']:
            print(f"\n警告: {len(stats['files_with_errors'])} 个文件处理失败")
            for fn in stats['files_with_errors']:
                print(f"  - {fn}")
        
        print(f"\n输出目录: {self.output_dir}")
        print("  - all_samples/: 所有样本")
        print("  - healthy/: 健康样本")
        print("  - light_damage/: 轻度损伤")
        print("  - moderate_damage/: 中度损伤")
        print("  - severe_damage/: 重度损伤")
        print("  - multi_damage/: 多处损伤")
        print("  - channels_split/: RGB分通道展示")
        print("=" * 60)
    
    def create_summary_grid(self, num_samples_per_class=5, 
                            output_filename='summary_grid.png'):
        """
        创建汇总网格图，展示各类别的代表性样本
        
        Args:
            num_samples_per_class: 每个类别展示的样本数
            output_filename: 输出文件名
        """
        class_dirs = ['healthy', 'light_damage', 'moderate_damage', 
                      'severe_damage', 'multi_damage']
        class_names = ['健康', '轻度损伤', '中度损伤', '重度损伤', '多处损伤']
        
        fig, axes = plt.subplots(len(class_dirs), num_samples_per_class, 
                                figsize=(num_samples_per_class * 3, len(class_dirs) * 3))
        
        if len(class_dirs) == 1:
            axes = axes.reshape(1, -1)
        if num_samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for row, (class_dir, class_name) in enumerate(zip(class_dirs, class_names)):
            # 查找该类别的图像
            img_files = glob(os.path.join(self.output_dir, class_dir, '*.png'))
            img_files.sort()
            
            # 选择前num_samples_per_class个样本
            selected_files = img_files[:num_samples_per_class]
            
            for col in range(num_samples_per_class):
                ax = axes[row, col]
                
                if col < len(selected_files):
                    try:
                        img = plt.imread(selected_files[col])
                        ax.imshow(img)
                        ax.axis('off')
                        
                        # 在第一列添加类别标签
                        if col == 0:
                            ax.set_ylabel(class_name, fontsize=10, fontweight='bold')
                    except Exception as e:
                        ax.text(0.5, 0.5, f'加载失败\n{str(e)[:20]}', 
                               ha='center', va='center', fontsize=8)
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, '无样本', ha='center', va='center', 
                            fontsize=8, color='gray')
                    ax.axis('off')
        
        plt.suptitle('GVR特征图汇总', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        
        print(f"汇总网格图已保存到: {output_path}")


def main():
    """主函数"""
    # 配置参数
    data_dir = './jacket_damage_data_timespace'  # 数据集目录
    output_dir = './extracted_gvr_images'       # 输出目录
    
    print("=" * 60)
    print("GVR特征图提取程序")
    print("=" * 60)
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 创建提取器
    extractor = GVRImageExtractor(data_dir, output_dir)
    
    # 提取所有文件的特征图
    stats = extractor.extract_all_files(
        pattern='scenario_*.h5',
        save_all=True,      # 保存到all_samples目录
        save_split=True,    # 保存分通道图像
        max_files=None      # None表示处理所有文件
    )
    
    # 创建汇总网格图
    if stats and stats['total_samples'] > 0:
        print("\n创建汇总网格图...")
        extractor.create_summary_grid(
            num_samples_per_class=5,
            output_filename='summary_grid.png'
        )
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
