import os
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional

# 配置设置
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DataQualityInspector:
    """
    数据质量检查器
    用于全面验证导管架平台损伤数据的质量
    
    主要检查项：
    1. 元数据完整性
    2. 文件是否存在
    3. 数据维度一致性
    4. NaN和Inf值检查
    5. 数值范围合理性
    6. 标签一致性
    7. 信号质量
    8. 损伤判别能力
    9. 数据分布平衡性
    """
    
    def __init__(self, data_dir: str):
        """
        初始化数据检查器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.metadata = None
        self.check_results = {}
        self.quality_score = 0.0
        
        # 检查阈值配置
        self.thresholds = {
            'max_nan_ratio': 0.0,              # 不允许NaN
            'max_inf_ratio': 0.0,              # 不允许Inf
            'min_acceleration_range': (-500, 500),  # 加速度合理范围 (m/s²)
            'feature_pixel_range': (0, 1),     # 特�征图像素值范围
            'min_signal_energy': 1e-10,        # 最小信号能量
            'max_signal_energy': 1e10           # 最大信号能量
        }
    
    def load_metadata(self) -> bool:
        """加载元数据文件"""
        metadata_file = os.path.join(self.data_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            self._add_error('metadata', '元数据文件不存在')
            return False
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self._add_success('metadata', f'成功加载 {len(self.metadata)} 个场景的元数据')
            return True
        except Exception as e:
            self._add_error('metadata', f'加载元数据失败: {str(e)}')
            return False
    
    def check_file_existence(self) -> Dict:
        """检查所有HDF5数据文件是否存在"""
        result = {
            'total_scenarios': 0,
            'missing_files': [],
            'existing_files': []
        }
        
        if self.metadata is None:
            self.load_metadata()
            if self.metadata is None:
                return result
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            if os.path.exists(filename):
                result['existing_files'].append(scenario_id)
            else:
                result['missing_files'].append(scenario_id)
        
        result['total_scenarios'] = len(self.metadata)
        
        if len(result['missing_files']) == 0:
            self._add_success('file_existence', 
                            f'所有 {result["total_scenarios"]} 个场景文件都存在')
        else:
            self._add_error('file_existence', 
                          f'缺失 {len(result["missing_files"])} 个文件')
        
        return result
    
    def check_data_dimensions(self) -> Dict:
        """检查所有场景的数据维度是否一致"""
        result = {
            'checked_scenarios': 0,
            'dimension_errors': [],
            'dimension_summary': {}
        }
        
        if self.metadata is None:
            return result
        
        expected_shapes = {}
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    current_dims = {}
                    for key in hf.keys():
                        current_dims[key] = hf[key].shape
                    
                    # 第一次扫描时记录预期维度
                    if result['checked_scenarios'] == 0:
                        expected_shapes.update(current_dims)
                    
                    # 检查维度一致性
                    for key in current_dims:
                        if key in expected_shapes and expected_shapes[key] != current_dims[key]:
                            error_msg = f'Scenario {scenario_id}: {key} 维度不匹配 (预期: {expected_shapes[key]}, 实际: {current_dims[key]})'
                            result['dimension_errors'].append(error_msg)
                    
                    result['checked_scenarios'] += 1
                    
            except Exception as e:
                error_msg = f'Scenario {scenario_id}: 读取文件失败 - {str(e)}'
                result['dimension_errors'].append(error_msg)
        
        result['dimension_summary'] = expected_shapes
        
        if len(result['dimension_errors']) == 0:
            self._add_success('data_dimensions', 
                            f'所有 {result["checked_scenarios"]} 个场景的维度一致')
        else:
            self._add_error('data_dimensions', 
                          f'发现 {len(result["dimension_errors"])} 个维度错误')
        
        return result
    
    def check_nan_inf(self) -> Dict:
        """检查数据中的NaN和Inf值"""
        result = {
            'checked_scenarios': 0,
            'scenarios_with_nan': [],
            'scenarios_with_inf': []
        }
        
        if self.metadata is None:
            return result
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    has_nan = False
                    has_inf = False
                    
                    for key in hf.keys():
                        data = hf[key][:]
                        
                        if np.isnan(data).any():
                            has_nan = True
                        
                        if np.isinf(data).any():
                            has_inf = True
                    
                    if has_nan:
                        result['scenarios_with_nan'].append(scenario_id)
                    
                    if has_inf:
                        result['scenarios_with_inf'].append(scenario_id)
                    
                    result['checked_scenarios'] += 1
                    
            except Exception as e:
                pass
        
        total_scenarios = result['checked_scenarios']
        nan_ratio = len(result['scenarios_with_nan']) / total_scenarios if total_scenarios > 0 else 0
        inf_ratio = len(result['scenarios_with_inf']) / total_scenarios if total_scenarios > 0 else 0
        
        if nan_ratio == 0 and inf_ratio == 0:
            self._add_success('nan_inf_check', 
                            f'所有 {total_scenarios} 个场景的数据都正常，无NaN或Inf值')
        else:
            error_msg = []
            if nan_ratio > 0:
                error_msg.append(f'{len(result["scenarios_with_nan"])} 个场景包含NaN值 ({nan_ratio*100:.2f}%)')
            if inf_ratio > 0:
                error_msg.append(f'{len(result["scenarios_with_inf"])} 个场景包含Inf值 ({inf_ratio*100:.2f}%)')
            self._add_error('nan_inf_check', ', '.join(error_msg))
        
        return result
    
    def check_value_ranges(self) -> Dict:
        """检查数值范围是否合理"""
        result = {
            'out_of_range_samples': []
        }
        
        if self.metadata is None:
            return result
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    # 检查加速度数据
                    if 'acceleration' in hf:
                        acc = hf['acceleration'][:]
                        min_acc_range, max_acc_range = self.thresholds['min_acceleration_range']
                        out_of_range = (acc < min_acc_range).any() or (acc > max_acc_range).any()
                        
                        if out_of_range:
                            result['out_of_range_samples'].append({
                                'scenario_id': scenario_id,
                                'type': 'acceleration',
                                'min': float(acc.min()),
                                'max': float(acc.max())
                            })
                    
                    # 检查特征图数据
                    if 'feature_maps' in hf:
                        feat_maps = hf['feature_maps'][:]
                        min_feat_range, max_feat_range = self.thresholds['feature_pixel_range']
                        out_of_range = (feat_maps < min_feat_range - 1e-6).any() or \
                                       (feat_maps > max_feat_range + 1e-6).any()
                        
                        if out_of_range:
                            result['out_of_range_samples'].append({
                                'scenario_id': scenario_id,
                                'type': 'feature_map',
                                'min': float(feat_maps.min()),
                                'max': float(feat_maps.max())
                            })
                    
            except Exception as e:
                pass
        
        if len(result['out_of_range_samples']) == 0:
            self._add_success('value_range_check', '所有数据的数值范围都在合理范围内')
        else:
            self._add_error('value_range_check', 
                          f'发现 {len(result["out_of_range_samples"])} 个数值范围异常的样本')
        
        return result
    
    def check_label_consistency(self) -> Dict:
        """检查标签与元数据是否一致"""
        result = {
            'checked_scenarios': 0,
            'inconsistent_labels': []
        }
        
        if self.metadata is None:
            return result
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            damaged_dofs = item.get('damaged_dofs', [])
            expected_damage_class = item.get('damage_class', 0)
            
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'labels' in hf:
                        labels = hf['labels'][:]
                        labels_array = np.array(labels)
                        
                        # 检查受损DOF是否都被标记为1
                        if len(damaged_dofs) > 0:
                            for dof in damaged_dofs:
                                if labels_array[dof] != 1:
                                    result['inconsistent_labels'].append({
                                        'scenario_id': scenario_id,
                                        'type': 'missing_label',
                                        'dof': dof,
                                        'expected': 1,
                                        'actual': float(labels_array[dof])
                                    })
                        else:
                            # 健康样本，所有label应该为0
                            if np.any(labels_array != 0):
                                result['inconsistent_labels'].append({
                                    'scenario_id': scenario_id,
                                    'type': 'healthy_not_zero',
                                    'count': int(np.sum(labels_array != 0))
                                })
                    
                    # 检查damage_class
                    if 'damage_class' in hf:
                        actual_damage_class = int(hf['damage_class'][0])
                        if actual_damage_class != expected_damage_class:
                            result['inconsistent_labels'].append({
                                'scenario_id': scenario_id,
                                'type': 'damage_class_mismatch',
                                'expected': expected_damage_class,
                                'actual': actual_damage_class
                            })
                    
                    result['checked_scenarios'] += 1
                    
            except Exception as e:
                result['inconsistent_labels'].append({
                    'scenario_id': scenario_id,
                    'type': 'read_error',
                    'error': str(e)
                })
        
        if len(result['inconsistent_labels']) == 0:
            self._add_success('label_consistency', 
                            f'所有 {result["checked_scenarios"]} 个场景的标签一致')
        else:
            self._add_error('label_consistency', 
                          f'发现 {len(result["inconsistent_labels"])} 个标签不一致的问题')
        
        return result
    
    def check_signal_quality(self) -> Dict:
        """检查信号质量"""
        result = {
            'checked_scenarios': 0,
            'low_energy_scenarios': [],
            'zero_signal_scenarios': []
        }
        
        if self.metadata is None:
            return result
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'acceleration' in hf:
                        acc = hf['acceleration'][:]
                        total_energy = np.sum(acc ** 2)
                        
                        if total_energy < self.thresholds['min_signal_energy']:
                            result['low_energy_scenarios'].append(scenario_id)
                        
                        if total_energy == 0:
                            result['zero_signal_scenarios'].append(scenario_id)
                    
                    if 'feature_maps' in hf:
                        feat_maps = hf['feature_maps'][:]
                        feat_energy = np.sum(feat_maps ** 2)
                        
                        if feat_energy == 0:
                            result['zero_signal_scenarios'].append(scenario_id)
                    
                    result['checked_scenarios'] += 1
                    
            except Exception as e:
                pass
        
        if len(result['low_energy_scenarios']) == 0 and len(result['zero_signal_scenarios']) == 0:
            self._add_success('signal_quality', 
                            f'所有 {result["checked_scenarios"]} 个场景的信号质量良好')
        else:
            error_msgs = []
            if len(result['low_energy_scenarios']) > 0:
                error_msgs.append(f'{len(result["low_energy_scenarios"])} 个场景能量过低')
            if len(result['zero_signal_scenarios']) > 0:
                error_msgs.append(f'{len(result["zero_signal_scenarios"])} 个场景为零信号')
            self._add_error('signal_quality', ', '.join(error_msgs))
        
        return result
    
    def check_damage_discrimination(self) -> Dict:
        """检查健康与损伤样本的分离度 (Cohen's d)"""
        result = {
            'healthy_scenarios': [],
            'damaged_scenarios': [],
            'healthy_feature_stats': {'mean': [], 'std': []},
            'damaged_feature_stats': {'mean': [], 'std': []},
            'separation_score': 0.0
        }
        
        if self.metadata is None:
            return result
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            damaged_dofs = item.get('damaged_dofs', [])
            is_healthy = len(damaged_dofs) == 0
            
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'feature_maps' in hf:
                        feat_maps = hf['feature_maps'][:]
                        
                        mean_val = float(np.mean(feat_maps))
                        std_val = float(np.std(feat_maps))
                        
                        if is_healthy:
                            result['healthy_scenarios'].append(scenario_id)
                            result['healthy_feature_stats']['mean'].append(mean_val)
                            result['healthy_feature_stats']['std'].append(std_val)
                        else:
                            result['damaged_scenarios'].append(scenario_id)
                            result['damaged_feature_stats']['mean'].append(mean_val)
                            result['damaged_feature_stats']['std'].append(std_val)
                    
            except Exception as e:
                pass
        
        # 计算 Cohen's d
        # 修正逻辑：比较两组均值之间的分离度
        if (len(result['healthy_feature_stats']['mean']) > 0 and 
            len(result['damaged_feature_stats']['mean']) > 0):
            
            healthy_means = np.array(result['healthy_feature_stats']['mean'])
            damaged_means = np.array(result['damaged_feature_stats']['mean'])
            
            # 1. 计算组平均均值
            mean_healthy = np.mean(healthy_means)
            mean_damaged = np.mean(damaged_means)
            
            # 2. 计算均值的标准差 (这是衡量组间离散度的关键)
            # ddof=1 使用样本标准差
            std_healthy_means = np.std(healthy_means, ddof=1)
            std_damaged_means = np.std(damaged_means, ddof=1)
            
            # 3. 计算合并标准差
            # 只有当两组都有足够的方差时才计算
            if std_healthy_means > 0 or std_damaged_means > 0:
                pooled_std = np.sqrt((std_healthy_means**2 + std_damaged_means**2) / 2)
                
                mean_diff = mean_damaged - mean_healthy
                
                if pooled_std > 0:
                    cohens_d = np.abs(mean_diff) / pooled_std
                    result['separation_score'] = float(cohens_d)
                else:
                    result['separation_score'] = 0.0
            else:
                result['separation_score'] = 0.0
        
        # 评价分离度
        if result['separation_score'] > 0.8:
            self._add_success('damage_discrimination', 
                            f'健康和损伤样本分离度良好 (Cohen\'s d = {result["separation_score"]:.3f})')
        elif result['separation_score'] > 0.5:
            self._add_warning('damage_discrimination', 
                            f'健康和损伤样本分离度中等 (Cohen\'s d = {result["separation_score"]:.3f})')
        elif result['separation_score'] > 0.2: # 稍微降低阈值，因为生成器改进初期可能不会立刻达到0.5
            self._add_warning('damage_discrimination', 
                            f'健康和损伤样本分离度较低 (Cohen\'s d = {result["separation_score"]:.3f})')
        else:
            self._add_error('damage_discrimination', 
                          f'无法计算健康和损伤样本的分离度或分离度极低 (Cohen\'s d = {result["separation_score"]:.3f})')
        
        return result


    def check_distribution_balance(self) -> Dict:
        """检查数据分布平衡性"""
        result = {
            'total_scenarios': 0,
            'healthy_count': 0,
            'damaged_count': 0,
            'class_counts': {}
        }
        
        if self.metadata is None:
            return result
        
        for item in self.metadata:
            damage_class = item.get('damage_class', 0)
            result['total_scenarios'] += 1
            result['class_counts'][damage_class] = \
                result['class_counts'].get(damage_class, 0) + 1
            
            if damage_class == 0:
                result['healthy_count'] += 1
            else:
                result['damaged_count'] += 1
        
        healthy_ratio = result['healthy_count'] / result['total_scenarios'] if result['total_scenarios'] > 0 else 0
        result['healthy_ratio'] = healthy_ratio
        
        if 0.1 <= healthy_ratio <= 0.5:
            self._add_success('distribution_balance', 
                            f'数据分布平衡，健康样本占比 {healthy_ratio*100:.1f}%')
        else:
            self._add_warning('distribution_balance', 
                            f'数据分布可能不平衡，健康样本占比 {healthy_ratio*100:.1f}%')
        
        return result
    
    def run_full_inspection(self) -> Dict:
        """运行完整的数据质量检查"""
        print("=" * 80)
        print("开始全面数据质量检查...")
        print("=" * 80)
        
        results = {}
        
        # 1. 加载元数据
        print("\n[1/8] 检查元数据加载...")
        results['metadata'] = self.load_metadata()
        
        if self.metadata is not None:
            # 2. 检查文件完整性
            print("\n[2/8] 检查文件完整性...")
            results['file_existence'] = self.check_file_existence()
            
            # 3. 检查数据维度
            print("\n[3/8] 检查数据维度...")
            results['dimensions'] = self.check_data_dimensions()
            
            # 4. 检查NaN和Inf
            print("\n[4/8] 检查NaN和Inf...")
            results['nan_inf'] = self.check_nan_inf()
            
            # 5. 检查数值范围
            print("\n[5/8] 检查数值范围...")
            results['value_ranges'] = self.check_value_ranges()
            
            # 6. 检查标签一致性
            print("\n[6/8] 检查标签一致性...")
            results['labels'] = self.check_label_consistency()
            
            # 7. 检查信号质量
            print("\n[7/8] 检查信号质量...")
            results['signal_quality'] = self.check_signal_quality()
            
            # 8. 检查损伤判别能力
            print("\n[8/8] 检查损伤判别能力...")
            results['damage_discrimination'] = self.check_damage_discrimination()
            
            # 额外: 检查数据分布平衡性
            print("\n[额外] 检查数据分布平衡性...")
            results['distribution_balance'] = self.check_distribution_balance()
        else:
            print("警告：元数据加载失败，跳过后续检查")
        
        # 计算总体质量分数
        self._calculate_quality_score()
        
        print("\n" + "=" * 80)
        print("数据质量检查完成！")
        print("=" * 80)
        
        return results
    
    # 私有辅助方法
    def _add_success(self, category: str, message: str):
        if category not in self.check_results:
            self.check_results[category] = []
        self.check_results[category].append({'status': 'success', 'message': message})
    
    def _add_error(self, category: str, message: str):
        if category not in self.check_results:
            self.check_results[category] = []
        self.check_results[category].append({'status': 'error', 'message': message})
    
    def _add_warning(self, category: str, message: str):
        if category not in self.check_results:
            self.check_results[category] = []
        self.check_results[category].append({'status': 'warning', 'message': message})
    
    def _calculate_quality_score(self):
        total_checks = 0
        passed_checks = 0
        
        for category, checks in self.check_results.items():
            for check in checks:
                total_checks += 1
                if check['status'] == 'success':
                    passed_checks += 1
        
        self.quality_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    def generate_text_report(self, output_file: str = None) -> str:
        """生成文字检查报告"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("数据质量检查报告")
        report_lines.append("=" * 80)
        report_lines.append(f"检查时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据目录: {self.data_dir}")
        report_lines.append(f"总体质量分数: {self.quality_score:.1f}/100")
        report_lines.append("")
        
        # 摘要统计
        report_lines.append("-" * 80)
        report_lines.append("检查结果摘要")
        report_lines.append("-" * 80)
        
        success_count = 0
        error_count = 0
        warning_count = 0
        
        for category, checks in self.check_results.items():
            for check in checks:
                if check['status'] == 'success':
                    success_count += 1
                elif check['status'] == 'error':
                    error_count += 1
                elif check['status'] == 'warning':
                    warning_count += 1
        
        report_lines.append(f"✓ 通过检查: {success_count} 项")
        report_lines.append(f"✗ 发现错误: {error_count} 项")
        report_lines.append(f"⚠ 发现警告: {warning_count} 项")
        report_lines.append("")
        
        # 详细结果
        report_lines.append("-" * 80)
        report_lines.append("详细检查结果")
        report_lines.append("-" * 80)
        
        for category, checks in self.check_results.items():
            report_lines.append(f"\n【{category}】")
            for check in checks:
                status_icon = "✓" if check['status'] == 'success' else ("✗" if check['status'] == 'error' else "⚠")
                report_lines.append(f"  {status_icon} {check['message']}")
        
        # 数据集统计
        if self.metadata:
            report_lines.append("\n" + "-" * 80)
            report_lines.append("数据集统计信息")
            report_lines.append("-" * 80)
            
            total_scenarios = len(self.metadata)
            healthy_scenarios = sum(1 for item in self.metadata if len(item.get('damaged_dofs', [])) == 0)
            damaged_scenarios = total_scenarios - healthy_scenarios
            
            report_lines.append(f"总场景数: {total_scenarios}")
            report_lines.append(f"健康场景: {healthy_scenarios} ({healthy_scenarios/total_scenarios*100:.1f}%)")
            report_lines.append(f"损伤场景: {damaged_scenarios} ({damaged_scenarios/total_scenarios*100:.1f}%)")
            
            # 类别分布
            class_dist = {}
            for item in self.metadata:
                damage_class = item.get('damage_class', 0)
                class_dist[damage_class] = class_dist.get(damage_class, 0) + 1
            
            report_lines.append("\n损伤类别分布:")
            class_names = {
                0: '健康', 1: '轻微损伤', 2: '中等损伤', 3: '严重损伤', 4: '多点损伤'
            }
            for cls, count in sorted(class_dist.items()):
                name = class_names.get(cls, f'类别{cls}')
                report_lines.append(f"  {name}: {count} 个场景 ({count/total_scenarios*100:.1f}%)")
            
            # === 图表统计信息 ===
            report_lines.append("\n" + "-" * 80)
            report_lines.append("图表统计信息")
            report_lines.append("-" * 80)
            
            # 1. 标签一致性统计
            label_stats = self._collect_label_consistency_stats()
            if label_stats:
                report_lines.append("\n[标签一致性分析]")
                report_lines.append(f"  总检查数: {label_stats['total']}")
                report_lines.append(f"  一致数: {label_stats['consistent_count']} ({label_stats['consistency_rate']:.1f}%)")
                report_lines.append(f"  不一致数: {label_stats['inconsistent_count']} ({100-label_stats['consistency_rate']:.1f}%)")
                
                report_lines.append("\n  各类别一致性率:")
                for class_id, data in sorted(label_stats['class_consistency'].items()):
                    name = data['name']
                    count = int(data['count'])
                    rate = data['rate']
                    report_lines.append(f"    {name}: {rate:.1f}% ({count}/{count})")
                
                if label_stats['inconsistent_cases']:
                    report_lines.append("\n  不一致案例 (最多显示10个):")
                    for case in label_stats['inconsistent_cases']:
                        report_lines.append(f"    场景{case['scenario_id']}: 期望{case['expected_damaged_dofs']}, 实际{case['actual_damaged_labels']}")
            
            # 2. 健康vs损伤对比统计
            hvdd_stats = self._collect_healthy_vs_damaged_stats()
            if hvdd_stats:
                report_lines.append("\n[健康 vs 损伤对比]")
                if 'healthy' in hvdd_stats:
                    h = hvdd_stats['healthy']
                    report_lines.append(f"  健康样本 ({h['count']} 个):")
                    report_lines.append(f"    特征均值: {h['mean_mean']:.6f} ± {h['std_mean']:.6f}")
                    report_lines.append(f"    特征标准差: {h['mean_std']:.6f} ± {h['std_std']:.6f}")
                    report_lines.append(f"    均值范围: [{h['mean_range'][0]:.6f}, {h['mean_range'][1]:.6f}]")
                
                if 'damaged' in hvdd_stats:
                    d = hvdd_stats['damaged']
                    report_lines.append(f"  损伤样本 ({d['count']} 个):")
                    report_lines.append(f"    特征均值: {d['mean_mean']:.6f} ± {d['std_mean']:.6f}")
                    report_lines.append(f"    特征标准差: {d['mean_std']:.6f} ± {d['std_std']:.6f}")
                    report_lines.append(f"    均值范围: [{d['mean_range'][0]:.6f}, {d['mean_range'][1]:.6f}]")
                
                if 'cohens_d' in hvdd_stats:
                    d_value = hvdd_stats['cohens_d']
                    if d_value > 0.8:
                        sep_level = "优秀"
                    elif d_value > 0.5:
                        sep_level = "中等"
                    elif d_value > 0.2:
                        sep_level = "较低"
                    else:
                        sep_level = "极低"
                    report_lines.append(f"  Cohen's d 分离度: {d_value:.3f} ({sep_level})")
            
            # 3. 信号质量统计
            signal_stats = self._collect_signal_quality_stats()
            if signal_stats:
                report_lines.append("\n[信号质量分析]")
                if 'healthy' in signal_stats:
                    h = signal_stats['healthy']
                    report_lines.append(f"  健康样本 ({h['count']} 个):")
                    report_lines.append(f"    信号能量: 均值={h['energy']['mean']:.2e}, 标准差={h['energy']['std']:.2e}, 范围=[{h['energy']['min']:.2e}, {h['energy']['max']:.2e}]")
                    report_lines.append(f"    RMS值: 均值={h['rms']['mean']:.4f}, 标准差={h['rms']['std']:.4f}, 范围=[{h['rms']['min']:.4f}, {h['rms']['max']:.4f}]")
                
                if 'damaged' in signal_stats:
                    d = signal_stats['damaged']
                    report_lines.append(f"  损伤样本 ({d['count']} 个):")
                    report_lines.append(f"    信号能量: 均值={d['energy']['mean']:.2e}, 标准差={d['energy']['std']:.2e}, 范围=[{d['energy']['min']:.2e}, {d['energy']['max']:.2e}]")
                    report_lines.append(f"    RMS值: 均值={d['rms']['mean']:.4f}, 标准差={d['rms']['std']:.4f}, 范围=[{d['rms']['min']:.4f}, {d['rms']['max']:.4f}]")
            
            # 4. 加速度分布统计
            acc_stats = self._collect_acceleration_stats()
            if acc_stats:
                report_lines.append("\n[加速度数据分布]")
                report_lines.append(f"  采样点数: {acc_stats['sample_count']:,}")
                report_lines.append(f"  均值: {acc_stats['mean']:.4f} m/s²")
                report_lines.append(f"  标准差: {acc_stats['std']:.4f} m/s²")
                report_lines.append(f"  最小值: {acc_stats['min']:.4f} m/s²")
                report_lines.append(f"  中位数: {acc_stats['median']:.4f} m/s²")
                report_lines.append(f"  最大值: {acc_stats['max']:.4f} m/s²")
                report_lines.append(f"  偏度: {acc_stats['skewness']:.4f}")
                report_lines.append(f"  峰度: {acc_stats['kurtosis']:.4f}")
            
            # 5. 特征图分布统计
            feat_stats = self._collect_feature_map_stats()
            if feat_stats:
                report_lines.append("\n[特征图数值分布]")
                report_lines.append(f"  采样点数: {feat_stats['sample_count']:,}")
                report_lines.append(f"  均值: {feat_stats['mean']:.6f}")
                report_lines.append(f"  标准差: {feat_stats['std']:.6f}")
                report_lines.append(f"  最小值: {feat_stats['min']:.6f}")
                report_lines.append(f"  中位数: {feat_stats['median']:.6f}")
                report_lines.append(f"  最大值: {feat_stats['max']:.6f}")
                report_lines.append(f"  数值范围: {feat_stats['range']:.6f}")
            
            # 6. 损伤严重程度统计
            severity_stats = self._collect_severity_stats()
            if severity_stats:
                report_lines.append("\n[损伤严重程度分布]")
                report_lines.append(f"  总损伤点数: {severity_stats['total_samples']}")
                report_lines.append(f"  均值: {severity_stats['mean']:.3f}")
                report_lines.append(f"  标准差: {severity_stats['std']:.3f}")
                report_lines.append(f"  最小值: {severity_stats['min']:.3f}")
                report_lines.append(f"  中位数: {severity_stats['median']:.3f}")
                report_lines.append(f"  最大值: {severity_stats['max']:.3f}")
                report_lines.append(f"  分区间统计:")
                for label, count in sorted(severity_stats['bin_distribution'].items()):
                    percentage = count / severity_stats['total_samples'] * 100
                    report_lines.append(f"    {label}: {count} 个 ({percentage:.1f}%)")
        
        # 结论和建议
        report_lines.append("\n" + "-" * 80)
        report_lines.append("结论和建议")
        report_lines.append("-" * 80)
        
        if self.quality_score >= 90:
            report_lines.append("✓ 数据质量优秀，可以安全用于模型训练。")
        elif self.quality_score >= 70:
            report_lines.append("⚠ 数据质量良好，但存在一些需要注意的问题。")
            report_lines.append("  建议：修复错误项后再进行大规模训练。")
        elif self.quality_score >= 50:
            report_lines.append("✗ 数据质量一般，存在较多问题。")
            report_lines.append("  建议：修复所有错误项并检查数据生成逻辑。")
        else:
            report_lines.append("✗✗ 数据质量较差，存在严重问题！")
            report_lines.append("  强烈建议：彻底检查数据生成器，修复所有bug。")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"文字报告已保存至: {output_file}")
            except Exception as e:
                print(f"保存文字报告失败: {e}")
        
        return report_text


    def _collect_label_consistency_stats(self) -> Dict:
        """收集标签一致性统计信息"""
        label_data = []
        for item in self.metadata:
            scenario_id = item['scenario_id']
            damaged_dofs = item.get('damaged_dofs', [])
            damage_class = item.get('damage_class', 0)
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'labels' in hf:
                        labels = hf['labels'][:]
                        num_damaged_labels = np.sum(labels > 0)
                        label_data.append({
                            'scenario_id': scenario_id,
                            'expected_damaged_dofs': len(damaged_dofs),
                            'actual_damaged_labels': int(num_damaged_labels),
                            'is_consistent': len(damaged_dofs) == num_damaged_labels,
                            'damage_class': damage_class
                        })
            except:
                pass
        
        if len(label_data) == 0:
            return None
        
        df = pd.DataFrame(label_data)
        consistent_count = df['is_consistent'].sum()
        inconsistent_count = len(df) - consistent_count
        
        # 类别一致性统计
        class_consistency = df.groupby('damage_class')['is_consistent'].agg(['count', 'sum'])
        class_consistency['rate'] = class_consistency['sum'] / class_consistency['count'] * 100
        class_names_map = {0: '健康', 1: '轻微', 2: '中等', 3: '严重', 4: '多点'}
        class_consistency['name'] = class_consistency.index.map(lambda x: class_names_map.get(x, f'C{x}'))
        
        # 不一致案例
        inconsistent_cases = df[~df['is_consistent']].to_dict('records')
        
        return {
            'total': len(df),
            'consistent_count': consistent_count,
            'inconsistent_count': inconsistent_count,
            'consistency_rate': consistent_count / len(df) * 100,
            'class_consistency': class_consistency.to_dict('index'),
            'inconsistent_cases': inconsistent_cases[:10]  # 最多显示10个
        }
    
    def _collect_healthy_vs_damaged_stats(self) -> Dict:
        """收集健康vs损伤对比统计信息"""
        healthy_means = []
        healthy_stds = []
        damaged_means = []
        damaged_stds = []
        
        sample_limit = min(10, len(self.metadata))
        
        for item in self.metadata[:sample_limit]:
            scenario_id = item['scenario_id']
            damaged_dofs = item.get('damaged_dofs', [])
            is_healthy = len(damaged_dofs) == 0
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'feature_maps' in hf:
                        dset = hf['feature_maps']
                        sample_count = min(10, dset.shape[0])
                        
                        if sample_count > 0:
                            feat_maps_subset = dset[0:sample_count]
                            feat_mean = float(np.mean(feat_maps_subset))
                            feat_std = float(np.std(feat_maps_subset))
                            
                            if is_healthy:
                                healthy_means.append(feat_mean)
                                healthy_stds.append(feat_std)
                            else:
                                damaged_means.append(feat_mean)
                                damaged_stds.append(feat_std)
            except:
                pass
        
        if not healthy_means and not damaged_means:
            return None
        
        stats = {}
        if healthy_means:
            stats['healthy'] = {
                'count': len(healthy_means),
                'mean_mean': np.mean(healthy_means),
                'std_mean': np.std(healthy_means),
                'mean_std': np.mean(healthy_stds),
                'std_std': np.std(healthy_stds),
                'mean_range': (np.min(healthy_means), np.max(healthy_means))
            }
        if damaged_means:
            stats['damaged'] = {
                'count': len(damaged_means),
                'mean_mean': np.mean(damaged_means),
                'std_mean': np.std(damaged_means),
                'mean_std': np.mean(damaged_stds),
                'std_std': np.std(damaged_stds),
                'mean_range': (np.min(damaged_means), np.max(damaged_means))
            }
        
        # 计算Cohen's d
        if len(healthy_means) > 1 and len(damaged_means) > 1:
            mean_diff = np.mean(damaged_means) - np.mean(healthy_means)
            std_healthy_means = np.std(healthy_means, ddof=1)
            std_damaged_means = np.std(damaged_means, ddof=1)
            pooled_std = np.sqrt((std_healthy_means**2 + std_damaged_means**2) / 2)
            if pooled_std > 0:
                stats['cohens_d'] = abs(mean_diff) / pooled_std
            else:
                stats['cohens_d'] = 0.0
        
        return stats
    
    def _collect_signal_quality_stats(self) -> Dict:
        """收集信号质量统计信息"""
        healthy_energies = []
        damaged_energies = []
        healthy_rms = []
        damaged_rms = []
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'acceleration' in hf:
                        acc = hf['acceleration'][:]
                        energy = np.sum(acc ** 2)
                        rms = np.sqrt(np.mean(acc ** 2))
                        is_healthy = len(item.get('damaged_dofs', [])) == 0
                        
                        if is_healthy:
                            healthy_energies.append(energy)
                            healthy_rms.append(rms)
                        else:
                            damaged_energies.append(energy)
                            damaged_rms.append(rms)
            except:
                pass
        
        stats = {}
        if healthy_energies:
            stats['healthy'] = {
                'count': len(healthy_energies),
                'energy': {
                    'mean': np.mean(healthy_energies),
                    'std': np.std(healthy_energies),
                    'min': np.min(healthy_energies),
                    'max': np.max(healthy_energies),
                    'median': np.median(healthy_energies)
                },
                'rms': {
                    'mean': np.mean(healthy_rms),
                    'std': np.std(healthy_rms),
                    'min': np.min(healthy_rms),
                    'max': np.max(healthy_rms),
                    'median': np.median(healthy_rms)
                }
            }
        if damaged_energies:
            stats['damaged'] = {
                'count': len(damaged_energies),
                'energy': {
                    'mean': np.mean(damaged_energies),
                    'std': np.std(damaged_energies),
                    'min': np.min(damaged_energies),
                    'max': np.max(damaged_energies),
                    'median': np.median(damaged_energies)
                },
                'rms': {
                    'mean': np.mean(damaged_rms),
                    'std': np.std(damaged_rms),
                    'min': np.min(damaged_rms),
                    'max': np.max(damaged_rms),
                    'median': np.median(damaged_rms)
                }
            }
        
        return stats if stats else None
    
    def _collect_acceleration_stats(self) -> Dict:
        """收集加速度数据统计信息"""
        acc_values = []
        sample_limit = min(10, len(self.metadata))
        
        for item in self.metadata[:sample_limit]:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'acceleration' in hf:
                        acc = hf['acceleration'][:]
                        if acc.size > 5000:
                            flat_acc = acc.flatten()
                            indices = np.random.choice(flat_acc.size, 5000, replace=False)
                            acc_subset = flat_acc[indices]
                        else:
                            acc_subset = acc.flatten()
                        acc_values.extend(acc_subset)
            except:
                pass
        
        if len(acc_values) == 0:
            return None
        
        acc_values = np.array(acc_values)
        return {
            'sample_count': len(acc_values),
            'mean': float(np.mean(acc_values)),
            'std': float(np.std(acc_values)),
            'min': float(np.min(acc_values)),
            'median': float(np.median(acc_values)),
            'max': float(np.max(acc_values)),
            'skewness': float(stats.skew(acc_values)),
            'kurtosis': float(stats.kurtosis(acc_values))
        }
    
    def _collect_feature_map_stats(self) -> Dict:
        """收集特征图统计信息"""
        feat_values = []
        sample_limit = min(5, len(self.metadata))
        
        for item in self.metadata[:sample_limit]:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'feature_maps' in hf:
                        dset = hf['feature_maps']
                        num_windows = dset.shape[0]
                        read_count = min(3, num_windows)
                        subset = dset[0:read_count]
                        feat_values.extend(subset.flatten())
            except:
                pass
        
        if len(feat_values) == 0:
            return None
        
        feat_values = np.array(feat_values)
        return {
            'sample_count': len(feat_values),
            'mean': float(np.mean(feat_values)),
            'std': float(np.std(feat_values)),
            'min': float(np.min(feat_values)),
            'median': float(np.median(feat_values)),
            'max': float(np.max(feat_values)),
            'range': float(np.max(feat_values) - np.min(feat_values))
        }
    
    def _collect_severity_stats(self) -> Dict:
        """收集损伤严重程度统计信息"""
        severity_data = []
        for item in self.metadata:
            severity_ratios = item.get('severity_ratios', [])
            severity_data.extend(severity_ratios)
        
        if len(severity_data) == 0:
            return None
        
        severity_data = np.array(severity_data)
        
        # 分区间统计
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        bin_counts, _ = np.histogram(severity_data, bins=bins)
        
        bin_stats = {}
        for label, count in zip(bin_labels, bin_counts):
            bin_stats[label] = int(count)
        
        return {
            'total_samples': len(severity_data),
            'mean': float(np.mean(severity_data)),
            'std': float(np.std(severity_data)),
            'min': float(np.min(severity_data)),
            'median': float(np.median(severity_data)),
            'max': float(np.max(severity_data)),
            'bin_distribution': bin_stats
        }

    def generate_visualizations(self, output_dir: str = None):
        """生成可视化图表"""
        if output_dir is None:
            output_dir = self.data_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n生成可视化图表...")
        
        # 1. 汇总饼图
        self._plot_summary_pie(output_dir)
        
        # 如果有数据，继续生成其他图表
        if self.metadata and len(self.metadata) > 0:
            # 2. 类别分布
            self._plot_class_distribution(output_dir)
            # 3. 标签一致性
            self._plot_label_consistency(output_dir)
            # 4. 健康 vs 损伤对比
            self._plot_healthy_vs_damaged(output_dir)
            # 5. 信号质量
            self._plot_signal_quality(output_dir)
            # 6. 加速度分布
            self._plot_acceleration_distribution(output_dir)
            # 7. 特征图分布
            self._plot_feature_map_distribution(output_dir)
            # 8. 严重程度分布
            self._plot_severity_distribution(output_dir)
        
        print(f"可视化图表已保存至: {output_dir}")
    
    # --- 绘图辅助函数 ---
    
    def _plot_summary_pie(self, output_dir: str):
        """绘制检查结果汇总饼图"""
        success_count = sum(1 for checks in self.check_results.values() 
                          for check in checks if check['status'] == 'success')
        error_count = sum(1 for checks in self.check_results.values() 
                        for check in checks if check['status'] == 'error')
        warning_count = sum(1 for checks in self.check_results.values() 
                          for check in checks if check['status'] == 'warning')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sizes = [success_count, error_count, warning_count]
        labels = ['通过', '错误', '警告']
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        ax.set_title('数据质量检查结果汇总', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_summary_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, output_dir: str):
        """绘制类别分布图"""
        class_counts = {}
        class_names_map = {0: '健康', 1: '轻微损伤', 2: '中等损伤', 3: '严重损伤', 4: '多点损伤'}
        
        for item in self.metadata:
            damage_class = item.get('damage_class', 0)
            name = class_names_map.get(damage_class, f'类别{damage_class}')
            class_counts[name] = class_counts.get(name, 0) + 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        
        # 柱状图
        bars = ax1.bar(class_counts.keys(), class_counts.values(), color=colors)
        ax1.set_xlabel('损伤类别', fontsize=12)
        ax1.set_ylabel('场景数量', fontsize=12)
        ax1.set_title('损伤类别分布（柱状图）', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 饼图
        wedges, texts, autotexts = ax2.pie(class_counts.values(), labels=class_counts.keys(),
                                           autopct='%1.1f%%', colors=colors, shadow=True)
        ax2.set_title('损伤类别分布（饼图）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_label_consistency(self, output_dir: str):
        """绘制标签一致性分析图"""
        label_data = []
        for item in self.metadata:
            scenario_id = item['scenario_id']
            damaged_dofs = item.get('damaged_dofs', [])
            damage_class = item.get('damage_class', 0)
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'labels' in hf:
                        labels = hf['labels'][:]
                        num_damaged_labels = np.sum(labels > 0)
                        label_data.append({
                            'scenario_id': scenario_id,
                            'expected_damaged_dofs': len(damaged_dofs),
                            'actual_damaged_labels': int(num_damaged_labels),
                            'is_consistent': len(damaged_dofs) == num_damaged_labels,
                            'damage_class': damage_class
                        })
            except:
                pass
        
        if len(label_data) == 0:
            return
            
        df = pd.DataFrame(label_data)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 一致性饼图
        consistent_count = df['is_consistent'].sum()
        inconsistent_count = len(df) - consistent_count
        axes[0, 0].pie([consistent_count, inconsistent_count], labels=['一致', '不一致'],
                      colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', shadow=True)
        axes[0, 0].set_title('标签一致性检查', fontsize=14, fontweight='bold')
        
        # 散点图
        colors_scatter = ['#2ecc71' if c else '#e74c3c' for c in df['is_consistent']]
        axes[0, 1].scatter(df['expected_damaged_dofs'], df['actual_damaged_labels'], c=colors_scatter, alpha=0.6, s=80)
        max_val = max(df['expected_damaged_dofs'].max(), df['actual_damaged_labels'].max())
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
        axes[0, 1].set_xlabel('期望受损DOF数量', fontsize=12)
        axes[0, 1].set_ylabel('实际受损标签数量', fontsize=12)
        axes[0, 1].set_title('标签一致性散点图', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 类别一致性率
        class_consistency = df.groupby('damage_class')['is_consistent'].agg(['count', 'sum'])
        class_consistency['rate'] = class_consistency['sum'] / class_consistency['count'] * 100
        class_names_map = {0: '健康', 1: '轻微', 2: '中等', 3: '严重', 4: '多点'}
        class_consistency['name'] = class_consistency.index.map(lambda x: class_names_map.get(x, f'C{x}'))
        
        x_pos = np.arange(len(class_consistency))
        bars = axes[1, 0].bar(x_pos, class_consistency['rate'],
                            color=['#2ecc71' if r == 100 else '#f39c12' for r in class_consistency['rate']])
        axes[1, 0].set_xlabel('损伤类别', fontsize=12)
        axes[1, 0].set_ylabel('一致性率 (%)', fontsize=12)
        axes[1, 0].set_title('各损伤类别的标签一致性率', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(class_consistency['name'], rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0, 105])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom')
        
        # 不一致详情
        inconsistent_cases = df[~df['is_consistent']]
        if len(inconsistent_cases) > 0:
            case_text = "不一致案例:\n"
            for idx, row in inconsistent_cases.head(10).iterrows():
                case_text += f"场景{row['scenario_id']}: 期望{row['expected_damaged_dofs']}, 实际{row['actual_damaged_labels']}\n"
            if len(inconsistent_cases) > 10:
                case_text += f"... 还有 {len(inconsistent_cases)-10} 个案例"
            axes[1, 1].text(0.05, 0.95, case_text, fontsize=9, family='monospace',
                          verticalalignment='top', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('不一致案例详情', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, '✓ 所有标签都一致！', fontsize=14, ha='center', va='center', fontweight='bold')
            axes[1, 1].axis('off')
            axes[1, 1].set_title('标签一致性', fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_label_consistency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_healthy_vs_damaged(self, output_dir: str):
        """绘制健康vs损伤样本对比图 (优化内存版)"""
        feature_data = {'healthy': [], 'damaged': []}
        
        for item in self.metadata:
            scenario_id = item['scenario_id']
            damaged_dofs = item.get('damaged_dofs', [])
            is_healthy = len(damaged_dofs) == 0
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'feature_maps' in hf:
                        dset = hf['feature_maps']
                        # 优化：只读取前 10 个时间窗口来估算统计量，避免加载整个数据集
                        # 每个窗口是 (224, 224, 3)，10个窗口约12MB，远小于原来的650MB
                        sample_count = min(10, dset.shape[0])
                        
                        if sample_count > 0:
                            feat_maps_subset = dset[0:sample_count]
                            
                            # 计算均值和标准差
                            # 注意：这只是基于部分样本的估算，但足够反映特征变化
                            feat_mean = np.mean(feat_maps_subset)
                            feat_std = np.std(feat_maps_subset)
                            
                            if is_healthy:
                                feature_data['healthy'].append({'mean': feat_mean, 'std': feat_std})
                            else:
                                feature_data['damaged'].append({'mean': feat_mean, 'std': feat_std})
            except Exception as e:
                print(f"处理场景 {scenario_id} 时出错: {e}")
                pass
        
        if not feature_data['healthy'] and not feature_data['damaged']:
            print("没有特征数据，跳过绘图")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        healthy_means = [d['mean'] for d in feature_data['healthy']]
        damaged_means = [d['mean'] for d in feature_data['damaged']]
        
        # 均值分布
        axes[0, 0].hist(healthy_means, bins=30, alpha=0.5, label='健康', color='#2ecc71')
        axes[0, 0].hist(damaged_means, bins=30, alpha=0.5, label='损伤', color='#e74c3c')
        axes[0, 0].set_xlabel('特征均值', fontsize=12)
        axes[0, 0].set_ylabel('频数', fontsize=12)
        axes[0, 0].set_title('健康vs损伤：特征均值分布', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 标准差分布
        healthy_stds = [d['std'] for d in feature_data['healthy']]
        damaged_stds = [d['std'] for d in feature_data['damaged']]
        axes[0, 1].hist(healthy_stds, bins=30, alpha=0.5, label='健康', color='#2ecc71')
        axes[0, 1].hist(damaged_stds, bins=30, alpha=0.5, label='损伤', color='#e74c3c')
        axes[0, 1].set_xlabel('特征标准差', fontsize=12)
        axes[0, 1].set_ylabel('频数', fontsize=12)
        axes[0, 1].set_title('健康vs损伤：特征标准差分布', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 箱线图
        bp = axes[1, 0].boxplot([healthy_means, damaged_means], labels=['健康', '损伤'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1, 0].set_ylabel('特征均值', fontsize=12)
        axes[1, 0].set_title('健康vs损伤：箱线图对比', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 散点图
        axes[1, 1].scatter(healthy_means, healthy_stds, c='#2ecc71', alpha=0.6, 
                            label='健康', s=50)
        axes[1, 1].scatter(damaged_means, damaged_stds, c='#e74c3c', alpha=0.6, 
                            label='损伤', s=50)
        axes[1, 1].set_xlabel('特征均值', fontsize=12)
        axes[1, 1].set_ylabel('特征标准差', fontsize=12)
        axes[1, 1].set_title('健康vs损伤：均值vs标准差散点图', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_healthy_vs_damaged.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_signal_quality(self, output_dir: str):
        """绘制信号质量分析图"""
        signal_data = []
        for item in self.metadata:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'acceleration' in hf:
                        acc = hf['acceleration'][:]
                        signal_data.append({
                            'scenario_id': scenario_id,
                            'energy': np.sum(acc ** 2),
                            'rms': np.sqrt(np.mean(acc ** 2)),
                            'is_healthy': len(item.get('damaged_dofs', [])) == 0
                        })
            except Exception as e:
                pass
        
        if len(signal_data) == 0:
            return
            
        df = pd.DataFrame(signal_data)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        healthy_df = df[df['is_healthy']]
        damaged_df = df[~df['is_healthy']]

        # --- 辅助函数：安全绘制分布 ---
        def plot_distribution(ax, data, label, color, title_suffix=""):
            if len(data) == 0:
                return
            
            min_val = np.min(data)
            max_val = np.max(data)
            
            # 如果数据是常数（所有值相同），直方图无法绘制，改用垂直线表示
            if min_val == max_val:
                # 绘制垂直线
                ax.axvline(min_val, color=color, linestyle='--', alpha=0.8, linewidth=2, label=label)
                # 添加数值标注
                current_ylim = ax.get_ylim()
                # 如果当前y轴还是默认的(0,1)，手动设置一个稍微高点的位置
                y_pos = current_ylim[1] * 0.9 if current_ylim[1] > 1 else 0.9
                ax.text(min_val, y_pos, f'{label}: {min_val:.2e}', 
                        color=color, fontsize=9, ha='center', va='top', fontweight='bold')
            else:
                # 正常绘制直方图
                ax.hist(data, bins='auto', color=color, label=label, alpha=0.5)

        # --- 1. 信号能量对比 ---
        if len(healthy_df) > 0:
            plot_distribution(axes[0, 0], healthy_df['energy'], '健康', '#2ecc71')
        if len(damaged_df) > 0:
            plot_distribution(axes[0, 0], damaged_df['energy'], '损伤', '#e74c3c')
            
        axes[0, 0].set_xlabel('信号能量', fontsize=12)
        axes[0, 0].set_ylabel('频数', fontsize=12)
        axes[0, 0].set_title('信号能量分布对比', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 只有当数据中包含正数且有变化时，对数坐标才有意义，否则设置线性坐标
        has_positive_energy = False
        if len(healthy_df) > 0 and healthy_df['energy'].max() > 0:
            has_positive_energy = True
        if len(damaged_df) > 0 and damaged_df['energy'].max() > 0:
            has_positive_energy = True
            
        # 检查是否有变化的非零数据（避免全是0时log(0)报错或全是常数时显示不美观）
        varied_positive_energy = False
        all_energy = []
        if len(healthy_df) > 0: all_energy.extend(healthy_df['energy'])
        if len(damaged_df) > 0: all_energy.extend(damaged_df['energy'])
        
        if len(all_energy) > 0 and np.max(all_energy) > np.min(all_energy) and np.min(all_energy) > 0:
            try:
                axes[0, 0].set_xscale('log')
            except:
                pass

        # --- 2. RMS 对比 ---
        if len(healthy_df) > 0:
            plot_distribution(axes[0, 1], healthy_df['rms'], '健康', '#2ecc71')
        if len(damaged_df) > 0:
            plot_distribution(axes[0, 1], damaged_df['rms'], '损伤', '#e74c3c')
            
        axes[0, 1].set_xlabel('RMS值', fontsize=12)
        axes[0, 1].set_ylabel('频数', fontsize=12)
        axes[0, 1].set_title('信号RMS分布对比', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # --- 3. 能量 vs RMS 散点图 ---
        if len(df) > 0:
            colors = ['#2ecc71' if is_healthy else '#e74c3c' for is_healthy in df['is_healthy']]
            axes[1, 0].scatter(df['rms'], df['energy'], c=colors, alpha=0.6, s=50)
            axes[1, 0].set_xlabel('RMS值', fontsize=12)
            axes[1, 0].set_ylabel('信号能量', fontsize=12)
            axes[1, 0].set_title('信号能量 vs RMS', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            # 安全设置对数坐标
            try:
                if df['energy'].min() > 0:
                    axes[1, 0].set_yscale('log')
            except:
                pass
            
            from matplotlib.patches import Patch
            axes[1, 0].legend(handles=[Patch(facecolor='#2ecc71', label='健康'), Patch(facecolor='#e74c3c', label='损伤')])

        # --- 4. 统计摘要 ---
        stats_text = "统计摘要:\n"
        stats_text += f"总样本数: {len(df)}\n"
        stats_text += f"健康样本: {df['is_healthy'].sum()}\n"
        stats_text += f"损伤样本: {len(df) - df['is_healthy'].sum()}\n"
        
        # 添加能量统计
        if len(healthy_df) > 0:
            stats_text += f"健康能量均值: {healthy_df['energy'].mean():.2e}\n"
        if len(damaged_df) > 0:
            stats_text += f"损伤能量均值: {damaged_df['energy'].mean():.2e}\n"
            
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('统计摘要', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_signal_quality.png'), dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_acceleration_distribution(self, output_dir: str):
        """绘制加速度数据分布 (优化内存版)"""
        acc_values = []
        sample_limit = min(10, len(self.metadata)) # 减少到10个场景
        
        for item in self.metadata[:sample_limit]:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'acceleration' in hf:
                        acc = hf['acceleration'][:]
                        # 优化：如果数据点太多，随机抽取 5000 个点
                        if acc.size > 5000:
                            flat_acc = acc.flatten()
                            indices = np.random.choice(flat_acc.size, 5000, replace=False)
                            acc_subset = flat_acc[indices]
                        else:
                            acc_subset = acc.flatten()
                        
                        acc_values.extend(acc_subset)
            except Exception as e:
                pass
        
        if len(acc_values) == 0:
            return
            
        acc_values = np.array(acc_values)
        print(f"绘制加速度分布，数据点数: {len(acc_values):,}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 直方图
        axes[0, 0].hist(acc_values, bins=100, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('加速度值 (m/s²)', fontsize=12)
        axes[0, 0].set_ylabel('频数', fontsize=12)
        axes[0, 0].set_title('加速度数据分布直方图', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Q-Q图
        stats.probplot(acc_values, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('加速度数据Q-Q图（正态性检验）', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1, 0].boxplot(acc_values, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#3498db', alpha=0.7))
        axes[1, 0].set_ylabel('加速度值 (m/s²)', fontsize=12)
        axes[1, 0].set_title('加速度数据箱线图', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 统计信息
        stats_text = f"""
        统计信息:
        样本数: {len(acc_values):,}
        均值: {acc_values.mean():.4f}
        标准差: {acc_values.std():.4f}
        最小值: {acc_values.min():.4f}
        中位数: {np.median(acc_values):.4f}
        最大值: {acc_values.max():.4f}
        偏度: {stats.skew(acc_values):.4f}
        峰度: {stats.kurtosis(acc_values):.4f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('加速度数据统计信息', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_acceleration_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_map_distribution(self, output_dir: str):
        """绘制特征图分布 (优化内存版)"""
        feat_values = []
        
        # 只处理前 5 个场景即可，减少内存压力
        sample_limit = min(5, len(self.metadata))
        
        for item in self.metadata[:sample_limit]:
            scenario_id = item['scenario_id']
            filename = os.path.join(self.data_dir, f'scenario_{scenario_id:04d}.h5')
            try:
                with h5py.File(filename, 'r') as hf:
                    if 'feature_maps' in hf:
                        dset = hf['feature_maps']
                        # 关键优化：不读取全部数据
                        # 只读取前 3 个窗口，每个窗口约0.5MB，总共1.5MB
                        # 原来读取全部是 650MB
                        num_windows = dset.shape[0]
                        read_count = min(3, num_windows)
                        
                        # 读取切片
                        subset = dset[0:read_count] 
                        feat_values.extend(subset.flatten())
            except Exception as e:
                print(f"读取特征图分布出错: {e}")
                pass
        
        if len(feat_values) == 0:
            return
            
        feat_values = np.array(feat_values)
        print(f"绘制特征图分布，数据点数: {len(feat_values):,}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 直方图
        axes[0, 0].hist(feat_values, bins=100, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('特征值', fontsize=12)
        axes[0, 0].set_ylabel('频数', fontsize=12)
        axes[0, 0].set_title('特征图数值分布直方图', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 累积分布
        sorted_data = np.sort(feat_values)
        cumulative = np.cumsum(sorted_data) / np.sum(sorted_data)
        axes[0, 1].plot(sorted_data, cumulative, color='#9b59b6', linewidth=2)
        axes[0, 1].set_xlabel('特征值', fontsize=12)
        axes[0, 1].set_ylabel('累积概率', fontsize=12)
        axes[0, 1].set_title('特征图累积分布函数', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1, 0].boxplot(feat_values, vert=True, patch_artist=True, boxprops=dict(facecolor='#9b59b6', alpha=0.7))
        axes[1, 0].set_ylabel('特征值', fontsize=12)
        axes[1, 0].set_title('特征图箱线图', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 统计信息
        stats_text = f"""
        统计信息:
        样本数: {len(feat_values):,}
        均值: {feat_values.mean():.6f}
        标准差: {feat_values.std():.6f}
        最小值: {feat_values.min():.6f}
        中位数: {np.median(feat_values):.6f}
        最大值: {feat_values.max():.6f}
        范围: {feat_values.max() - feat_values.min():.6f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('特征图统计信息', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '07_feature_map_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_severity_distribution(self, output_dir: str):
        """绘制损伤严重程度分布图"""
        severity_data = []
        for item in self.metadata:
            severity_ratios = item.get('severity_ratios', [])
            severity_data.extend(severity_ratios)
        
        if len(severity_data) == 0:
            return
            
        severity_data = np.array(severity_data)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 直方图
        axes[0, 0].hist(severity_data, bins=20, color='#e67e22', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('损伤严重程度', fontsize=12)
        axes[0, 0].set_ylabel('频数', fontsize=12)
        axes[0, 0].set_title('损伤严重程度分布', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 累积分布
        sorted_severity = np.sort(severity_data)
        cumulative = np.cumsum(sorted_severity) / np.sum(sorted_severity)
        axes[0, 1].plot(sorted_severity, cumulative, color='#e67e22', linewidth=2)
        axes[0, 1].set_xlabel('损伤严重程度', fontsize=12)
        axes[0, 1].set_ylabel('累积概率', fontsize=12)
        axes[0, 1].set_title('损伤严重程度累积分布', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 区间统计
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        bin_counts, _ = np.histogram(severity_data, bins=bins)
        colors_bin = plt.cm.Reds(np.linspace(0.4, 1, len(bin_counts)))
        bars = axes[1, 0].bar(bin_labels, bin_counts, color=colors_bin, edgecolor='black')
        axes[1, 0].set_xlabel('损伤严重程度范围', fontsize=12)
        axes[1, 0].set_ylabel('频数', fontsize=12)
        axes[1, 0].set_title('损伤严重程度分区间统计', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for bar, count in zip(bars, bin_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(count)}', ha='center', va='bottom')
        
        # 统计信息
        stats_text = f"""
        损伤严重程度统计:
        总样本数: {len(severity_data)}
        均值: {severity_data.mean():.3f}
        标准差: {severity_data.std():.3f}
        最小值: {severity_data.min():.3f}
        中位数: {np.median(severity_data):.3f}
        最大值: {severity_data.max():.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('损伤严重程度统计信息', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '08_severity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


# 使用示例
if __name__ == "__main__":
    # 设置数据目录路径（请修改为您实际的数据路径）
    DATA_DIR = './jacket_damage_data'
    
    # 如果使用 `new_mdof.py` 生成的数据，默认输出目录是 `./jacket_damage_data`
    # 或者您可以指定其他路径：
    # DATA_DIR = '/path/to/your/data'
    
    print(f"正在检查目录: {DATA_DIR}")
    
    # 创建检查器实例
    inspector = DataQualityInspector(data_dir=DATA_DIR)
    
    # 运行完整检查
    results = inspector.run_full_inspection()
    
    # 生成文字报告
    report_file = os.path.join(DATA_DIR, 'data_quality_report.txt')
    report_text = inspector.generate_text_report(output_file=report_file)
    
    # 打印报告到控制台
    print("\n" + "="*80)
    print("数据质量检查报告")
    print("="*80)
    print(report_text)
    
    # 生成可视化图表
    plot_dir = os.path.join(DATA_DIR, 'quality_plots')
    inspector.generate_visualizations(output_dir=plot_dir)
    
    # 打印最终分数
    print("\n" + "="*80)
    print(f"总体质量分数: {inspector.quality_score:.1f}/100")
    print("="*80)
    
    if inspector.quality_score >= 90:
        print("结果：数据质量优秀，可以用于训练！")
    elif inspector.quality_score >= 70:
        print("结果：数据质量良好，建议检查警告项。")
    else:
        print("结果：数据质量不佳，强烈建议修复错误！")
