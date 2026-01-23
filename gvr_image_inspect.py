"""
GVR图像抽取和检查工具
从生成的损伤数据中抽取GVR特征图，并按类别组织以便检查。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import os
import json
import sys
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

class GVRImageExtractor:
    """GVR图像抽取和可视化工具"""
    
    def __init__(self, data_dir: str = './jacket_damage_data', 
                 output_dir: str = './gvr_inspection'):
        """
        初始化GVR图像抽取器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出检查图像的目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 类别定义
        self.class_names = [
            '0_healthy',           # 0: 健康
            '1_mild_single',       # 1: 单点轻微损伤 (<30%)
            '2_moderate_single',   # 2: 单点中等损伤 (30-60%)
            '3_severe_single',     # 3: 单点严重损伤 (≥60%)
            '4_multiple_damage'    # 4: 多点损伤
        ]
        
        self.class_labels = [
            '健康',
            '单点轻微损伤 (<30%)',
            '单点中等损伤 (30-60%)',
            '单点严重损伤 (≥60%)',
            '多点损伤'
        ]
        
        self._create_output_directories()
    
    def _create_output_directories(self):
        """创建输出目录结构"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    def check_data_dir(self) -> Tuple[bool, str]:
        """
        检查数据目录是否存在且包含数据
        
        Returns:
            (is_valid, message): 是否有效和相关消息
        """
        if not os.path.exists(self.data_dir):
            return False, f"数据目录不存在: {self.data_dir}"
        
        h5_files = [f for f in os.listdir(self.data_dir) 
                   if f.endswith('.h5')]
        
        if len(h5_files) == 0:
            return False, f"数据目录中没有找到HDF5文件: {self.data_dir}"
        
        return True, f"找到 {len(h5_files)} 个数据文件"
    
    def load_metadata(self) -> List[Dict]:
        """加载元数据"""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告：无法加载元数据: {e}")
        return []
    
    def get_class_name(self, damage_class: int) -> str:
        """根据损伤类别获取类别名称"""
        if 0 <= damage_class < len(self.class_names):
            return self.class_names[damage_class]
        return 'unknown'
    
    def get_class_label(self, damage_class: int) -> str:
        """根据损伤类别获取中文标签"""
        if 0 <= damage_class < len(self.class_labels):
            return self.class_labels[damage_class]
        return '未知'
    
    def extract_gvr_images_from_file(self, file_path: str, 
                                    max_samples_per_file: int = 5,
                                    save_images: bool = True) -> Dict:
        """
        从单个HDF5文件中抽取GVR图像
        
        Args:
            file_path: HDF5文件路径
            max_samples_per_file: 每个文件最多抽取的图像数
            save_images: 是否保存图像
            
        Returns:
            文件信息字典
        """
        file_info = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'scenario_id': None,
            'damage_class': None,
            'damaged_dofs': [],
            'severity_ratios': [],
            'num_feature_maps': 0,
            'extracted_count': 0,
            'feature_maps': None,
            'success': False,
            'error': None
        }
        
        try:
            with h5py.File(file_path, 'r') as hf:
                # 读取基本信息
                file_info['scenario_id'] = int(hf.attrs.get('scenario_id', -1))
                file_info['damaged_dofs'] = hf.attrs.get('damaged_dofs', []).tolist()
                file_info['severity_ratios'] = hf.attrs.get('severity_ratios', []).tolist()
                
                if 'damage_class' in hf:
                    file_info['damage_class'] = int(hf['damage_class'][0])
                else:
                    # 根据受损自由度推断类别
                    if len(file_info['damaged_dofs']) == 0:
                        file_info['damage_class'] = 0
                    elif len(file_info['damaged_dofs']) == 1:
                        sev = file_info['severity_ratios'][0] if file_info['severity_ratios'] else 0.5
                        if sev < 0.3:
                            file_info['damage_class'] = 1
                        elif sev < 0.6:
                            file_info['damage_class'] = 2
                        else:
                            file_info['damage_class'] = 3
                    else:
                        file_info['damage_class'] = 4
                
                # 读取特征图
                if 'feature_maps' in hf:
                    feature_maps = hf['feature_maps'][:]
                    file_info['num_feature_maps'] = feature_maps.shape[0]
                    file_info['feature_maps'] = feature_maps
                else:
                    raise ValueError("文件中没有找到 feature_maps 数据集")
                
                # 决定抽取多少张图像
                num_to_extract = min(max_samples_per_file, feature_maps.shape[0])
                
                # 均匀分布抽取
                if num_to_extract == 1:
                    indices = [feature_maps.shape[0] // 2]
                else:
                    indices = np.linspace(0, feature_maps.shape[0] - 1, 
                                         num_to_extract, dtype=int).tolist()
                
                file_info['extracted_count'] = num_to_extract
                file_info['extracted_indices'] = indices
                
                # 保存图像
                if save_images:
                    self._save_images(feature_maps, indices, file_info)
                
                file_info['success'] = True
                
        except Exception as e:
            file_info['error'] = str(e)
            print(f"警告：处理文件 {file_path} 时出错: {str(e)}")
        
        return file_info
    
    def _save_images(self, feature_maps: np.ndarray, 
                    indices: List[int], 
                    file_info: Dict):
        """保存GVR图像到文件"""
        damage_class = file_info['damage_class']
        class_name = self.get_class_name(damage_class)
        class_label = self.get_class_label(damage_class)
        scenario_id = file_info.get('scenario_id', 'unknown')
        
        output_subdir = os.path.join(self.output_dir, class_name)
        
        for idx, sample_idx in enumerate(indices):
            fig, axes = plt.subplots(1, 1, figsize=(6, 6))
            
            # 提取图像
            img = feature_maps[sample_idx]
            
            # 显示图像
            axes.imshow(img)
            axes.set_title(f'类别 {damage_class}: {class_label}\n'
                          f'样本 {idx+1}/{len(indices)} (窗口 {sample_idx})',
                          fontsize=10)
            axes.axis('off')
            
            # 添加文件信息
            info_text = (f'文件: {file_info["file_name"]}\n'
                        f'DOF: {file_info["damaged_dofs"]}\n'
                        f'严重度: {file_info["severity_ratios"]}')
            fig.text(0.02, 0.02, info_text, fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 保存图像
            output_filename = f'scenario_{scenario_id:04d}_sample_{sample_idx:04d}.png'
            output_path = os.path.join(output_subdir, output_filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
    
    def extract_all_gvr_images(self, max_samples_per_file: int = 5):
        """
        从所有数据文件中抽取GVR图像
        
        Args:
            max_samples_per_file: 每个文件最多抽取的图像数
        """
        # 检查数据目录
        is_valid, message = self.check_data_dir()
        if not is_valid:
            print(f"\n{'='*60}")
            print(f"错误: {message}")
            print(f"{'='*60}")
            return None
        
        # 获取所有HDF5文件
        h5_files = [f for f in os.listdir(self.data_dir) 
                   if f.endswith('.h5')]
        h5_files.sort()
        
        print(f"找到 {len(h5_files)} 个数据文件")
        
        # 加载元数据
        metadata = self.load_metadata()
        print(f"加载了 {len(metadata)} 条元数据")
        
        # 统计信息
        class_counts = {class_name: 0 for class_name in self.class_names}
        total_extracted = 0
        file_infos = []
        
        # 处理每个文件
        print("\n开始抽取GVR图像...")
        for h5_file in tqdm(h5_files, desc="处理文件"):
            file_path = os.path.join(self.data_dir, h5_file)
            file_info = self.extract_gvr_images_from_file(
                file_path, 
                max_samples_per_file=max_samples_per_file,
                save_images=True
            )
            file_infos.append(file_info)
            
            # 更新统计
            if file_info['success']:
                damage_class = file_info.get('damage_class', -1)
                if 0 <= damage_class < len(self.class_names):
                    class_name = self.class_names[damage_class]
                    class_counts[class_name] += file_info['extracted_count']
                
                total_extracted += file_info['extracted_count']
        
        # 打印统计信息
        self._print_summary(class_counts, total_extracted, len(h5_files))
        
        # 保存抽取信息
        self._save_extraction_summary(file_infos, class_counts, total_extracted)
        
        return file_infos
    
    def _print_summary(self, class_counts: Dict, total: int, num_files: int):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("GVR图像抽取完成!")
        print(f"处理文件数: {num_files}")
        print(f"总计抽取: {total} 张图像")
        print("\n各类别抽取数量:")
        for class_name, count in class_counts.items():
            label = self.get_class_label(self.class_names.index(class_name))
            print(f"  {class_name:20s} ({label:15s}): {count:4d} 张")
        print("="*60)
    
    def _save_extraction_summary(self, file_infos: List[Dict], 
                                class_counts: Dict, total: int):
        """保存抽取摘要信息"""
        summary = {
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'total_files': len(file_infos),
            'total_extracted': total,
            'class_counts': class_counts,
            'files': []
        }
        
        for file_info in file_infos:
            summary['files'].append({
                'file_name': file_info['file_name'],
                'scenario_id': file_info['scenario_id'],
                'damage_class': file_info['damage_class'],
                'damaged_dofs': file_info['damaged_dofs'],
                'severity_ratios': file_info['severity_ratios'],
                'num_feature_maps': file_info['num_feature_maps'],
                'extracted_count': file_info['extracted_count'],
                'success': file_info['success'],
                'error': file_info['error']
            })
        
        summary_path = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n抽取摘要已保存至: {summary_path}")


def create_class_comparison_grid(output_dir: str = './gvr_inspection',
                                 images_per_class: int = 6,
                                 grid_filename: str = 'class_comparison_grid.png'):
    """
    创建各类别GVR图像对比网格
    
    Args:
        output_dir: 输出目录
        images_per_class: 每类显示的图像数
        grid_filename: 网格保存文件名
    """
    class_names = [
        '0_healthy', '1_mild_single', '2_moderate_single', 
        '3_severe_single', '4_multiple_damage'
    ]
    
    class_labels = [
        '健康', '单点轻微 (<30%)', '单点中等 (30-60%)',
        '单点严重 (≥60%)', '多点损伤'
    ]
    
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, images_per_class, 
                            figsize=(images_per_class*2.5, num_classes*2.5))
    
    # 处理单行情况
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        
        # 获取该类别的所有图像文件
        if os.path.exists(class_dir):
            img_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            img_files.sort()
            
            # 选择图像
            if len(img_files) > 0:
                selected_files = img_files[:min(images_per_class, len(img_files))]
                
                for img_idx, img_file in enumerate(selected_files):
                    if img_idx < images_per_class:
                        img_path = os.path.join(class_dir, img_file)
                        
                        try:
                            # 读取图像
                            from PIL import Image
                            img = Image.open(img_path)
                            axes[class_idx, img_idx].imshow(img)
                            
                            # 只在第一列添加类别标签
                            if img_idx == 0:
                                axes[class_idx, img_idx].set_ylabel(
                                    class_labels[class_idx], 
                                    fontsize=9, rotation=0, labelpad=45
                                )
                        except Exception as e:
                            print(f"警告：无法读取图像 {img_path}: {e}")
        
        # 移除坐标轴
        for img_idx in range(images_per_class):
            axes[class_idx, img_idx].axis('off')
    
    plt.suptitle('各类别GVR特征图对比', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, grid_filename)
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    print(f"类别对比网格已保存至: {output_path}")


def create_sample_gallery(output_dir: str = './gvr_inspection',
                         samples_per_class: int = 10,
                         gallery_filename: str = 'sample_gallery.png'):
    """
    创建样本画廊图
    
    Args:
        output_dir: 输出目录
        samples_per_class: 每类显示的样本数
        gallery_filename: 画廊保存文件名
    """
    class_names = [
        '0_healthy', '1_mild_single', '2_moderate_single', 
        '3_severe_single', '4_multiple_damage'
    ]
    
    class_labels = [
        '健康', '单点轻微', '单点中等', '单点严重', '多点损伤'
    ]
    
    num_classes = len(class_names)
    
    # 计算总样本数
    total_samples = 0
    class_sample_counts = []
    
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if os.path.exists(class_dir):
            img_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            count = min(samples_per_class, len(img_files))
            total_samples += count
            class_sample_counts.append(count)
        else:
            class_sample_counts.append(0)
    
    if total_samples == 0:
        print("警告：没有找到任何GVR图像，跳过画廊生成")
        return
    
    # 创建画廊布局
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(samples_per_class*2, num_classes*2))
    
    # 处理单行情况
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        
        # 获取该类别的所有图像文件
        if os.path.exists(class_dir):
            img_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            img_files.sort()
            
            # 选择图像
            selected_files = img_files[:min(samples_per_class, len(img_files))]
            
            for img_idx, img_file in enumerate(selected_files):
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # 读取图像
                    from PIL import Image
                    img = Image.open(img_path)
                    axes[class_idx, img_idx].imshow(img)
                    
                    # 只在第一列添加类别标签
                    if img_idx == 0:
                        axes[class_idx, img_idx].set_ylabel(
                            class_labels[class_idx], 
                            fontsize=10, rotation=0, labelpad=40
                        )
                except Exception as e:
                    print(f"警告：无法读取图像 {img_path}: {e}")
        
        # 移除坐标轴
        for img_idx in range(samples_per_class):
            axes[class_idx, img_idx].axis('off')
    
    plt.suptitle('GVR特征图样本画廊', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, gallery_filename)
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    print(f"样本画廊已保存至: {output_path}")


def print_usage():
    """打印使用说明"""
    print("="*70)
    print("GVR图像抽取和检查工具 - 使用说明".center(70))
    print("="*70)
    print("\n【基本用法】")
    print("  from extract_gvr_images import GVRImageExtractor, create_class_comparison_grid")
    print()
    print("  # 创建抽取器")
    print("  extractor = GVRImageExtractor(")
    print("      data_dir='./jacket_damage_data',  # 数据目录")
    print("      output_dir='./gvr_inspection'     # 输出目录")
    print("  )")
    print()
    print("  # 抽取所有GVR图像")
    print("  extractor.extract_all_gvr_images(max_samples_per_file=5)")
    print()
    print("  # 创建类别对比网格图")
    print("  create_class_comparison_grid(images_per_class=5)")
    print()
    print("  # 创建样本画廊")
    print("  create_sample_gallery(samples_per_class=10)")
    print()
    print("【命令行运行】")
    print("  python extract_gvr_images.py")
    print()
    print("【输出目录结构】")
    print("  ./gvr_inspection/")
    print("  ├── 0_healthy/              # 健康样本")
    print("  ├── 1_mild_single/          # 单点轻微损伤")
    print("  ├── 2_moderate_single/      # 单点中等损伤")
    print("  ├── 3_severe_single/        # 单点严重损伤")
    print("  ├── 4_multiple_damage/      # 多点损伤")
    print("  ├── class_comparison_grid.png    # 类别对比网格图")
    print("  ├── sample_gallery.png           # 样本画廊图")
    print("  └── extraction_summary.json      # 抽取摘要信息")
    print()
    print("="*70)


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    print("="*70)
    print("GVR图像抽取和检查工具".center(70))
    print("="*70)
    
    # 创建GVR图像抽取器
    extractor = GVRImageExtractor(
        data_dir='./jacket_damage_data_timespace',
        output_dir='./gvr_inspection'
    )
    
    # 检查数据目录
    is_valid, message = extractor.check_data_dir()
    
    if not is_valid:
        print(f"\n{message}")
        print("\n【提示】")
        print("  1. 请确保已运行 new_mdof_v1.py 生成数据")
        print("  2. 检查数据目录路径是否正确")
        print()
        print("【运行数据生成程序】")
        print("  python new_mdof_v1.py")
        print()
        print("【自定义数据目录】")
        print("  extractor = GVRImageExtractor(data_dir='your_data_dir')")
        print("  extractor.extract_all_gvr_images()")
        print("="*70)
        sys.exit(1)
    
    print(f"\n{message}")
    
    # 抽取所有GVR图像
    print("\n开始抽取GVR图像...")
    print("-"*70)
    file_infos = extractor.extract_all_gvr_images(max_samples_per_file=5)
    
    if file_infos and len(file_infos) > 0:
        # 创建类别对比网格图
        print("\n" + "="*70)
        print("正在创建类别对比网格...")
        print("-"*70)
        create_class_comparison_grid(
            output_dir='./gvr_inspection',
            images_per_class=5,
            grid_filename='class_comparison_grid.png'
        )
        
        # 创建样本画廊
        print("\n正在创建样本画廊...")
        print("-"*70)
        create_sample_gallery(
            output_dir='./gvr_inspection',
            samples_per_class=10,
            gallery_filename='sample_gallery.png'
        )
        
        # 完成提示
        print("\n" + "="*70)
        print("✓ 所有操作完成！".center(70))
        print("="*70)
        print(f"\n请检查输出目录: '{extractor.output_dir}'")
        print("\n【生成的文件】")
        print("  ✓ 按类别分组的GVR图像子目录")
        print("  ✓ class_comparison_grid.png  - 类别对比网格图")
        print("  ✓ sample_gallery.png         - 样本画廊图")
        print("  ✓ extraction_summary.json    - 详细抽取摘要")
        print("\n【检查建议】")
        print("  1. 查看各类别子目录中的图像样本")
        print("  2. 检查 class_comparison_grid.png 快速对比各类别特征")
        print("  3. 查看 extraction_summary.json 了解详细统计信息")
        print("="*70)
    else:
        print("\n【警告】未能成功抽取任何图像，请检查数据文件。")
        print("="*70)
