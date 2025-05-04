import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path
import matplotlib.transforms as mtransforms

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def draw_arrow(ax, start, end, color='black', width=1.5, style='-', curved=False, rad=0.3, shrink=25):
    """Draw an arrow from start to end with optional shrinkage to avoid text overlap"""
    if curved:
        # Create curved connection with arrow and shrinkage
        connectionstyle = f'arc3,rad={rad}'
        arrowprops = dict(
            arrowstyle='->', 
            color=color,
            lw=width,
            ls=style,
            connectionstyle=connectionstyle,
            shrinkA=shrink,  # Shrink from the start point
            shrinkB=shrink   # Shrink from the end point
        )
        ax.annotate('', xy=end, xytext=start, arrowprops=arrowprops)
    else:
        # Create straight arrow with shrinkage
        arrowprops = dict(
            arrowstyle='->', 
            color=color, 
            linewidth=width, 
            linestyle=style,
            shrinkA=shrink,  # Shrink from the start point
            shrinkB=shrink   # Shrink from the end point
        )
        ax.annotate('', xy=end, xytext=start, arrowprops=arrowprops)

def create_block(ax, x, y, width, height, label, color='lightblue', fontsize=10, alpha=0.7):
    """Create a block with label"""
    rect = patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='black', alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize, wrap=True)
    return (x + width/2, y + height/2)  # Return center coordinates

def create_title(ax, title, subtitle=None):
    """Create a title for the flowchart"""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    if subtitle:
        ax.text(0.5, 0.97, subtitle, fontsize=12, ha='center', transform=ax.transAxes)

def create_transferlearning1_flowchart():
    """Create the flowchart for transferLearning.py"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    create_title(ax, "TransferLearning.py - ResNet50 Transfer Learning Flowchart", 
                "使用预训练ResNet50模型进行垃圾分类 (Full Fine-tuning)")
    
    # Starting point
    start_center = create_block(ax, 1, 9, 4, 0.8, "准备阶段", color='#FFD580', fontsize=12)
    
    # Data preparation blocks
    data_prep_center = create_block(ax, 1, 7.5, 4, 1.2, 
                               "数据准备\n加载垃圾分类数据集\n图像增强与预处理\nImageFolderDataset", 
                               color='#B5EAD7', fontsize=10)
    
    # Load pretrained model
    model_load_center = create_block(ax, 7, 7.5, 4, 1.2, 
                                "加载预训练ResNet50模型\npretrained=True\n替换最后的FC层和平均池化层", 
                                color='#C7CEEA', fontsize=10)
    
    # Training preparation 
    train_prep_center = create_block(ax, 1, 5.5, 4, 1.0, 
                               "训练准备\n损失函数：SoftmaxCrossEntropy\n优化器：Momentum\n学习率：0.001", 
                               color='#E2F0CB', fontsize=10)
    
    # Fine-tuning details
    finetune_center = create_block(ax, 7, 5.5, 4, 1.0, 
                              "全参数微调\n所有层参数均可训练\ntrainable_params=network.trainable_params()", 
                              color='#FFDAC1', fontsize=10)
    
    # Training loop
    training_center = create_block(ax, 4, 3.5, 4, 1.2, 
                              "训练循环 (10 epochs)\n前向传播与反向传播\n计算损失与梯度\n模型参数更新", 
                              color='#FF9AA2', fontsize=10)
    
    # Evaluation
    eval_center = create_block(ax, 4, 1.8, 4, 0.8, 
                          "验证与评估\n计算验证集准确率\n保存最佳模型", 
                          color='#C7CEEA', fontsize=10)
    
    # Visualization
    vis_center = create_block(ax, 11, 3.5, 4, 1.2, 
                         "模型可视化\n加载最佳模型\n显示预测结果\n蓝色=正确 红色=错误", 
                         color='#B5EAD7', fontsize=10)
    
    # Draw arrows with appropriate shrinkage
    draw_arrow(ax, start_center, data_prep_center, shrink=30)
    draw_arrow(ax, start_center, model_load_center, curved=True, rad=0.3, shrink=30)
    draw_arrow(ax, data_prep_center, train_prep_center, shrink=30)
    draw_arrow(ax, model_load_center, finetune_center, shrink=30)
    draw_arrow(ax, train_prep_center, training_center, curved=True, rad=-0.2, shrink=30)
    draw_arrow(ax, finetune_center, training_center, curved=True, rad=0.2, shrink=30)
    draw_arrow(ax, training_center, eval_center, shrink=30)
    draw_arrow(ax, eval_center, training_center, curved=True, rad=-0.5, shrink=30)
    draw_arrow(ax, eval_center, vis_center, curved=True, rad=0.3, shrink=30)
    
    # Add notes/legend
    legend_items = [
        ("全参数微调", "所有网络参数都参与训练更新"),
        ("批量大小", "batch_size=18"),
        ("图像大小", "image_size=224"),
        ("类别数量", "26种垃圾分类")
    ]
    
    for i, (title, desc) in enumerate(legend_items):
        y_pos = 0.8 - i*0.15
        ax.text(0.8, y_pos, f"{title}: {desc}", fontsize=10, ha='left', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('TransferLearning1_Flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_transferlearning2_flowchart():
    """Create the flowchart for transferLearning2.py"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    create_title(ax, "TransferLearning2.py - ResNet50 Transfer Learning Flowchart", 
                "使用预训练ResNet50模型进行垃圾分类 (特征提取/Feature Extraction)")
    
    # Starting point
    start_center = create_block(ax, 1, 9, 4, 0.8, "准备阶段", color='#FFD580', fontsize=12)
    
    # Data preparation blocks
    data_prep_center = create_block(ax, 1, 7.5, 4, 1.2, 
                               "数据准备\n加载垃圾分类数据集\n图像增强与预处理\nImageFolderDataset", 
                               color='#B5EAD7', fontsize=10)
    
    # Load pretrained model
    model_load_center = create_block(ax, 7, 7.5, 4, 1.2, 
                                "加载预训练ResNet50模型\npretrained=True\n替换最后的FC层和平均池化层", 
                                color='#C7CEEA', fontsize=10)
    
    # Training preparation 
    train_prep_center = create_block(ax, 1, 5.5, 4, 1.0, 
                               "训练准备\n损失函数：SoftmaxCrossEntropy\n优化器：Momentum\n学习率：0.001", 
                               color='#E2F0CB', fontsize=10)
    
    # Feature extraction details
    freeze_center = create_block(ax, 7, 5.5, 4, 1.0, 
                              "冻结特征提取器\n仅训练全连接层(FC)参数\ntrainable_params只包含FC层参数", 
                              color='#FFDAC1', fontsize=10)
    
    # Training loop
    training_center = create_block(ax, 4, 3.5, 4, 1.2, 
                              "训练循环 (10 epochs)\n前向传播与反向传播\n计算损失与梯度\n仅更新FC层参数", 
                              color='#FF9AA2', fontsize=10)
    
    # Evaluation
    eval_center = create_block(ax, 4, 1.8, 4, 0.8, 
                          "验证与评估\n计算验证集准确率\n保存最佳模型", 
                          color='#C7CEEA', fontsize=10)
    
    # Visualization
    vis_center = create_block(ax, 11, 3.5, 4, 1.2, 
                         "模型可视化\n加载最佳模型\n显示预测结果\n蓝色=正确 红色=错误", 
                         color='#B5EAD7', fontsize=10)
    
    # Draw arrows with appropriate shrinkage
    draw_arrow(ax, start_center, data_prep_center, shrink=30)
    draw_arrow(ax, start_center, model_load_center, curved=True, rad=0.3, shrink=30)
    draw_arrow(ax, data_prep_center, train_prep_center, shrink=30)
    draw_arrow(ax, model_load_center, freeze_center, shrink=30)
    draw_arrow(ax, train_prep_center, training_center, curved=True, rad=-0.2, shrink=30)
    draw_arrow(ax, freeze_center, training_center, curved=True, rad=0.2, shrink=30)
    draw_arrow(ax, training_center, eval_center, shrink=30)
    draw_arrow(ax, eval_center, training_center, curved=True, rad=-0.5, shrink=30)
    draw_arrow(ax, eval_center, vis_center, curved=True, rad=0.3, shrink=30)
    
    # Add notes/legend
    legend_items = [
        ("特征提取策略", "冻结卷积层，仅训练全连接层"),
        ("批量大小", "batch_size=18"),
        ("图像大小", "image_size=224"),
        ("并行线程数", "workers=8 (比第一个文件多)")
    ]
    
    for i, (title, desc) in enumerate(legend_items):
        y_pos = 0.8 - i*0.15
        ax.text(0.8, y_pos, f"{title}: {desc}", fontsize=10, ha='left', transform=ax.transAxes)
    
    # Highlight the main difference
    highlight = patches.Rectangle((6.5, 5.2), 5, 1.5, fill=False, edgecolor='red', linestyle='--', linewidth=2)
    ax.add_patch(highlight)
    ax.text(12, 5.1, "主要区别", color='red', fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig('TransferLearning2_Flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_flowchart():
    """Create a flowchart comparing both transfer learning approaches"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    create_title(ax, "迁移学习方法比较 - 全参数微调 vs 特征提取", 
                "ResNet50模型用于垃圾分类的两种迁移学习策略")
    
    # Left side - Full Fine-tuning
    ax.text(4, 11, "全参数微调 (transferLearning.py)", fontsize=14, fontweight='bold', ha='center')
    
    # Right side - Feature Extraction
    ax.text(12, 11, "特征提取 (transferLearning2.py)", fontsize=14, fontweight='bold', ha='center')
    
    # Create central dividing line
    ax.plot([8, 8], [0.5, 10.5], '--', color='gray', linewidth=1)
    
    # Common elements at the top
    common_top = create_block(ax, 6, 9.5, 4, 0.8, "共同点：预训练ResNet50模型", color='#C7CEEA', fontsize=12)
    
    # Create blocks for left side (Full Fine-tuning)
    ft_data_center = create_block(ax, 2, 8.5, 4, 0.7, 
                               "数据加载与预处理\nbatch_size=18, image_size=224", 
                               color='#B5EAD7', fontsize=10)
    
    ft_model_center = create_block(ax, 2, 7.5, 4, 0.7, 
                               "加载预训练ResNet50\n替换FC层和平均池化层", 
                               color='#E2F0CB', fontsize=10)
    
    ft_params_center = create_block(ax, 2, 6.5, 4, 0.7, 
                                "优化器设置：Momentum\n学习率：0.001, momentum=0.9", 
                                color='#FFDAC1', fontsize=10)
    
    ft_train_center = create_block(ax, 2, 5.5, 4, 0.7, 
                               "参数设置:\nnetwork.trainable_params()\n所有网络参数均参与训练", 
                               color='#FF9AA2', fontsize=10, alpha=0.9)
    
    ft_update_center = create_block(ax, 2, 4.5, 4, 0.7, 
                                "训练循环中更新所有层参数\n包括特征提取层和分类层", 
                                color='#FF9AA2', fontsize=10, alpha=0.9)
    
    ft_mem_center = create_block(ax, 2, 3.5, 4, 0.7, 
                              "内存和计算消耗\n较大的存储器需求\n更新所有网络参数", 
                              color='#FFD580', fontsize=10)
    
    ft_time_center = create_block(ax, 2, 2.5, 4, 0.7, 
                               "训练速度\n速度较慢\n需要计算所有参数的梯度", 
                               color='#FFD580', fontsize=10)
    
    ft_advantage_center = create_block(ax, 2, 1.5, 4, 0.7, 
                                   "优势\n目标任务差异大时表现更好\n更高的模型适应性", 
                                   color='#C7CEEA', fontsize=10)
    
    # Create blocks for right side (Feature Extraction)
    fe_data_center = create_block(ax, 10, 8.5, 4, 0.7, 
                               "数据加载与预处理\nbatch_size=18, image_size=224", 
                               color='#B5EAD7', fontsize=10)
    
    fe_model_center = create_block(ax, 10, 7.5, 4, 0.7, 
                               "加载预训练ResNet50\n替换FC层和平均池化层", 
                               color='#E2F0CB', fontsize=10)
    
    fe_params_center = create_block(ax, 10, 6.5, 4, 0.7, 
                                "优化器设置：Momentum\n学习率：0.001, momentum=0.9", 
                                color='#FFDAC1', fontsize=10)
    
    fe_train_center = create_block(ax, 10, 5.5, 4, 0.7, 
                               "参数冻结:\ntrainable_params仅包含FC层\n仅全连接层参数参与训练", 
                               color='#FF9AA2', fontsize=10, alpha=0.9)
    
    fe_update_center = create_block(ax, 10, 4.5, 4, 0.7, 
                                "训练循环中仅更新FC层参数\n冻结所有特征提取层", 
                                color='#FF9AA2', fontsize=10, alpha=0.9)
    
    fe_mem_center = create_block(ax, 10, 3.5, 4, 0.7, 
                              "内存和计算消耗\n较小的存储器需求\n更少的参数需要更新", 
                              color='#FFD580', fontsize=10)
    
    fe_time_center = create_block(ax, 10, 2.5, 4, 0.7, 
                               "训练速度\n速度较快\n只需计算少量参数的梯度", 
                               color='#FFD580', fontsize=10)
    
    fe_advantage_center = create_block(ax, 10, 1.5, 4, 0.7, 
                                   "优势\n目标任务相似时表现良好\n防止过拟合风险小", 
                                   color='#C7CEEA', fontsize=10)
    
    # Add conclusion at the bottom
    conclusion = "结论：特征提取适合数据量小或与预训练任务相似的情况；全参数微调适合数据量大或任务差异大的情况"
    ax.text(8, 0.7, conclusion, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add a highlight to the key differences
    highlight1 = patches.Rectangle((1.5, 5.2), 5, 1.3, fill=False, edgecolor='red', linestyle='--', linewidth=2)
    highlight2 = patches.Rectangle((9.5, 5.2), 5, 1.3, fill=False, edgecolor='red', linestyle='--', linewidth=2)
    ax.add_patch(highlight1)
    ax.add_patch(highlight2)
    
    # Draw arrows with appropriate shrinkage
    # Left side
    draw_arrow(ax, common_top, ft_data_center, curved=True, rad=-0.2, shrink=30)
    draw_arrow(ax, ft_data_center, ft_model_center, shrink=25)
    draw_arrow(ax, ft_model_center, ft_params_center, shrink=25)
    draw_arrow(ax, ft_params_center, ft_train_center, shrink=25)
    draw_arrow(ax, ft_train_center, ft_update_center, shrink=25)
    draw_arrow(ax, ft_update_center, ft_mem_center, shrink=25)
    draw_arrow(ax, ft_mem_center, ft_time_center, shrink=25)
    draw_arrow(ax, ft_time_center, ft_advantage_center, shrink=25)
    
    # Right side
    draw_arrow(ax, common_top, fe_data_center, curved=True, rad=0.2, shrink=30)
    draw_arrow(ax, fe_data_center, fe_model_center, shrink=25)
    draw_arrow(ax, fe_model_center, fe_params_center, shrink=25)
    draw_arrow(ax, fe_params_center, fe_train_center, shrink=25)
    draw_arrow(ax, fe_train_center, fe_update_center, shrink=25)
    draw_arrow(ax, fe_update_center, fe_mem_center, shrink=25)
    draw_arrow(ax, fe_mem_center, fe_time_center, shrink=25)
    draw_arrow(ax, fe_time_center, fe_advantage_center, shrink=25)
    
    plt.tight_layout()
    plt.savefig('TransferLearning_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create all three flowcharts
create_transferlearning1_flowchart()
create_transferlearning2_flowchart()
create_comparison_flowchart()

print("三个流程图已生成:")
print("1. TransferLearning1_Flowchart.png - 全参数微调迁移学习")
print("2. TransferLearning2_Flowchart.png - 特征提取迁移学习")
print("3. TransferLearning_Comparison.png - 两种方法比较") 