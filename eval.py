"""
模型评估脚本
生成混淆矩阵、分类报告和多种可视化图表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve, 
    average_precision_score
)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd

# 导入模型和数据集类
from model1_mn_cnn_classifier import OffshoreDamageDetectionSystem
from h5_gvr_dataset import H5GVRDataset 


def main():
    # ================= 配置参数 =================
    DATA_DIR = './jacket_damage_data_ansys'
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'best_damage_detector.pth'
    
    # 类别名称
    CLASS_NAMES = ['健康/无损', '损伤']
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


    print("=" * 70)
    print("模型评估 - 生成混淆矩阵和统计信息")
    print("=" * 70)
    
    # ================= 1. 加载数据集 =================
    print("\n[步骤 1/6] 加载数据集...")
    full_dataset = H5GVRDataset(data_dir=DATA_DIR, window_length=3000, transform=None)
    print(f"  ✓ 总样本数: {len(full_dataset)}")
    
    # 划分数据集（与训练时保持一致）
    all_indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.4, random_state=42, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, shuffle=True
    )
    print(f"  ✓ 测试集样本数: {len(test_idx)}")
    
    # ================= 2. 初始化检测系统 =================
    print("\n[步骤 2/6] 初始化检测系统...")
    detection_system = OffshoreDamageDetectionSystem(num_classes=NUM_CLASSES, device=DEVICE)
    
    # 设置测试集的 transform
    test_dataset_base = H5GVRDataset(DATA_DIR, window_length=2000)
    test_dataset_base.transform = detection_system.valid_transform
    test_dataset = Subset(test_dataset_base, test_idx)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ================= 3. 加载训练好的模型 =================
    print(f"\n[步骤 3/6] 加载最佳模型: {MODEL_PATH}")
    detection_system.model.load_state_dict(torch.load(MODEL_PATH))
    detection_system.model.eval()
    print("  ✓ 模型加载成功")
    
    # ================= 4. 模型预测 =================
    print("\n[步骤 4/6] 执行模型预测...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for time_series, images, labels in test_loader:
            time_series = time_series.to(DEVICE)
            images = images.to(DEVICE)
            
            outputs = detection_system.model(time_series, images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    print("  ✓ 预测完成")
    
    # ================= 5. 计算评估指标 =================
    print("\n[步骤 5/6] 计算评估指标...")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 分类报告
    report = classification_report(
        all_labels, all_predictions, 
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    # 打印分类报告
    print("\n" + "=" * 70)
    print("分类报告")
    print("=" * 70)
    print(classification_report(
        all_labels, all_predictions, 
        target_names=CLASS_NAMES
    ))
    
    # 计算关键指标
    accuracy = report['accuracy']
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    
    # 计算 ROC 曲线和 AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算 Precision-Recall 曲线
    precision, recall, pr_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(
            all_labels == i, all_probabilities[:, i]
        )
        pr_auc[i] = average_precision_score(all_labels == i, all_probabilities[:, i])
    
    print(f"\n关键指标汇总:")
    print(f"  ✓ 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✓ 宏平均精确率: {precision_macro:.4f}")
    print(f"  ✓ 宏平均召回率: {recall_macro:.4f}")
    print(f"  ✓ 宏平均 F1 分数: {f1_macro:.4f}")
    
    # ================= 6. 生成可视化图表 =================
    print("\n[步骤 6/6] 生成可视化图表...")
    
    # 创建一个大的图，包含多个子图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ===== 图1: 混淆矩阵（热力图） =====
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={'label': '样本数量'}, ax=ax1
    )
    ax1.set_ylabel('真实标签', fontsize=11, fontweight='bold')
    ax1.set_xlabel('预测标签', fontsize=11, fontweight='bold')
    ax1.set_title('混淆矩阵', fontsize=13, fontweight='bold', pad=10)
    
    # 添加归一化标注
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j+0.5, i+0.7, f'\n({cm_normalized[i, j]:.2%})',
                    ha='center', va='center', fontsize=8, color='gray')
    
    # ===== 图2: 各类别性能指标 =====
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    class_metrics = []
    for class_name in CLASS_NAMES:
        class_metrics.append([
            report[class_name]['precision'],
            report[class_name]['recall'],
            report[class_name]['f1-score']
        ])
    
    class_metrics = np.array(class_metrics)
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        ax2.bar(x + i*width, class_metrics[:, i], width, 
                label=metric.capitalize(), alpha=0.8)
    
    ax2.set_xlabel('类别', fontsize=11, fontweight='bold')
    ax2.set_ylabel('分数', fontsize=11, fontweight='bold')
    ax2.set_title('各类别性能指标对比', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(CLASS_NAMES)
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值
    for i, class_name in enumerate(CLASS_NAMES):
        for j, metric in enumerate(metrics_to_plot):
            value = class_metrics[i, j]
            ax2.text(x[i] + j*width, value + 0.02, f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # ===== 图3: ROC 曲线 =====
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['#1f77b4', '#ff7f0e']
    for i, color in zip(range(NUM_CLASSES), colors):
        ax3.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', lw=1.5, label='随机分类器')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('假正率', fontsize=11, fontweight='bold')
    ax3.set_ylabel('真正率', fontsize=11, fontweight='bold')
    ax3.set_title('ROC 曲线', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc="lower right")
    ax3.grid(alpha=0.3)
    
    # ===== 图4: Precision-Recall 曲线 =====
    ax4 = fig.add_subplot(gs[1, 0])
    for i, color in zip(range(NUM_CLASSES), colors):
        ax4.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{CLASS_NAMES[i]} (AP = {pr_auc[i]:.3f})')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('召回率', fontsize=11, fontweight='bold')
    ax4.set_ylabel('精确率', fontsize=11, fontweight='bold')
    ax4.set_title('精确率-召回率曲线', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc="lower left")
    ax4.grid(alpha=0.3)
    
    # ===== 图5: 预测概率分布 =====
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        # 对于真实标签为 i 的样本，显示预测为各类别的概率
        mask = (all_labels == i)
        if np.sum(mask) > 0:
            probs = all_probabilities[mask, i]
            ax5.hist(probs, bins=30, alpha=0.6, color=color,
                    label=f'{CLASS_NAMES[i]}', edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('预测概率', fontsize=11, fontweight='bold')
    ax5.set_ylabel('频数', fontsize=11, fontweight='bold')
    ax5.set_title('预测概率分布（正确预测）', fontsize=13, fontweight='bold', pad=10)
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # ===== 图6: 分类错误分析 =====
    ax6 = fig.add_subplot(gs[1, 2])
    error_types = []
    error_counts = []
    
    # 正确分类
    correct_count = np.sum(all_predictions == all_labels)
    error_types.append('正确分类')
    error_counts.append(correct_count)
    
    # 错误分类（按类型）
    for true_class in range(NUM_CLASSES):
        for pred_class in range(NUM_CLASSES):
            if true_class != pred_class:
                count = np.sum((all_labels == true_class) & (all_predictions == pred_class))
                if count > 0:
                    error_types.append(f'{CLASS_NAMES[true_class]} →\n{CLASS_NAMES[pred_class]}')
                    error_counts.append(count)
    
    if len(error_types) > 1:  # 有错误分类
        colors_pie = ['#2ecc71'] + ['#e74c3c'] * (len(error_types) - 1)
        wedges, texts, autotexts = ax6.pie(
            error_counts, labels=error_types, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 9}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax6.text(0.5, 0.5, '100% 正确分类\n没有误分类！',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.5))
    
    ax6.set_title('分类结果分布', fontsize=13, fontweight='bold', pad=10)
    
    # ===== 图7: 宏平均指标雷达图 =====
    ax7 = fig.add_subplot(gs[2, 0], projection='polar')
    metrics_radar = ['精确率', '召回率', 'F1分数']
    values_radar = [precision_macro, recall_macro, f1_macro]
    values_radar += values_radar[:1]  # 闭合
    
    angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    ax7.plot(angles, values_radar, 'o-', linewidth=2, color='#3498db')
    ax7.fill(angles, values_radar, alpha=0.25, color='#3498db')
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(metrics_radar, fontsize=10)
    ax7.set_ylim(0, 1)
    ax7.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax7.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax7.set_title('宏平均性能雷达图', fontsize=13, fontweight='bold', pad=20, va='bottom')
    ax7.grid(True)
    
    # 在每个角上添加数值
    for angle, value in zip(angles[:-1], values_radar[:-1]):
        ax7.text(angle, value + 0.05, f'{value:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ===== 图8: 每个类别的样本分布 =====
    ax8 = fig.add_subplot(gs[2, 1])
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    
    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    bars1 = ax8.bar(x_pos - width/2, label_counts, width, 
                   label='真实分布', color='#3498db', alpha=0.8)
    bars2 = ax8.bar(x_pos + width/2, pred_counts, width,
                   label='预测分布', color='#e74c3c', alpha=0.8)
    
    ax8.set_xlabel('类别', fontsize=11, fontweight='bold')
    ax8.set_ylabel('样本数量', fontsize=11, fontweight='bold')
    ax8.set_title('测试集类别分布对比', fontsize=13, fontweight='bold', pad=10)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(CLASS_NAMES)
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # ===== 图9: 综合性能总结表 =====
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # 创建性能表格
    table_data = [
        ['指标', '数值'],
        ['-'*20, '-'*20],
        ['测试集准确率', f'{accuracy:.4f}'],
        ['宏平均精确率', f'{precision_macro:.4f}'],
        ['宏平均召回率', f'{recall_macro:.4f}'],
        ['宏平均 F1 分数', f'{f1_macro:.4f}'],
        ['-'*20, '-'*20],
        ['测试样本总数', f'{len(all_labels)}'],
        ['正确分类数', f'{correct_count}'],
        ['错误分类数', f'{len(all_labels) - correct_count}'],
    ]
    
    table = ax9.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            elif table_data[i][0] == '-'*20:  # 分隔线
                cell.set_facecolor('#ecf0f1')
            elif i == len(table_data) - 1:  # 最后一行
                cell.set_facecolor('#ecf0f1')
    
    ax9.set_title('性能总结', fontsize=13, fontweight='bold', pad=10)
    
    # 添加总体标题
    fig.suptitle(f'模型评估结果 - 海洋平台损伤检测\n'
                 f'测试准确率: {accuracy*100:.2f}% | '
                 f'F1分数: {f1_macro:.4f}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图表
    output_path = 'model_evaluation_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"  ✓ 评估图表已保存: {output_path}")
    
    plt.show()
    
    # ================= 保存详细报告 =================
    print("\n保存详细报告...")
    
    # 创建 DataFrame 保存每个样本的预测结果
    results_df = pd.DataFrame({
        'True_Label': [CLASS_NAMES[label] for label in all_labels],
        'Predicted_Label': [CLASS_NAMES[pred] for pred in all_predictions],
        'Correct': [label == pred for label, pred in zip(all_labels, all_predictions)],
        'Confidence_Healthy': all_probabilities[:, 0],
        'Confidence_Damaged': all_probabilities[:, 1],
    })
    
    results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    print("  ✓ 预测结果已保存: prediction_results.csv")
    
    # 保存混淆矩阵
    np.save('confusion_matrix.npy', cm)
    print("  ✓ 混淆矩阵已保存: confusion_matrix.npy")
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  1. model_evaluation_results.png - 综合评估图表")
    print(f"  2. prediction_results.csv - 详细预测结果")
    print(f"  3. confusion_matrix.npy - 混淆矩阵数据")
    print(f"\n关键指标:")
    print(f"  ✓ 准确率: {accuracy*100:.2f}%")
    print(f"  ✓ 宏平均 F1 分数: {f1_macro:.4f}")
    print(f"  ✓ 健康类别 AUC: {roc_auc[0]:.4f}")
    print(f"  ✓ 损伤类别 AUC: {roc_auc[1]:.4f}")

if __name__ == "__main__":
    main()
