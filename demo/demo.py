import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(16, 12))

# 设置背景颜色
ax.set_facecolor('#f8f8f8')

# 定义组件位置和大小
components = [
    {"name": "输入\n[N, 3, 88, 88]", "pos": (0, 0), "width": 2.5, "height": 1.5, "color": "#d6eaf8"},
    
    # 第一个卷积块
    {"name": "卷积层1 (Conv2d)\n输入: 3, 输出: 32\n核大小: 5×5, 步长: 1\npadding: same", 
     "pos": (3.5, 0), "width": 4, "height": 1.5, "color": "#d5f5e3"},
    {"name": "批归一化1\nBatchNorm2d", "pos": (8.5, 0), "width": 2, "height": 1.5, "color": "#e8daef"},
    {"name": "ReLU", "pos": (11.5, 0), "width": 1.5, "height": 1.5, "color": "#fadbd8"},
    {"name": "最大池化层1\n核大小: 2×2\n步长: 2", "pos": (14, 0), "width": 2.5, "height": 1.5, "color": "#fdebd0"},
    
    # 第二个卷积块
    {"name": "卷积层2 (Conv2d)\n输入: 32, 输出: 64\n核大小: 3×3, 步长: 1\npadding: same", 
     "pos": (3.5, -3), "width": 4, "height": 1.5, "color": "#d5f5e3"},
    {"name": "批归一化2\nBatchNorm2d", "pos": (8.5, -3), "width": 2, "height": 1.5, "color": "#e8daef"},
    {"name": "ReLU", "pos": (11.5, -3), "width": 1.5, "height": 1.5, "color": "#fadbd8"},
    {"name": "最大池化层2\n核大小: 2×2\n步长: 2", "pos": (14, -3), "width": 2.5, "height": 1.5, "color": "#fdebd0"},
    
    # 第三个卷积块
    {"name": "卷积层3 (Conv2d)\n输入: 64, 输出: 128\n核大小: 3×3, 步长: 1\npadding: same", 
     "pos": (3.5, -6), "width": 4, "height": 1.5, "color": "#d5f5e3"},
    {"name": "批归一化3\nBatchNorm2d", "pos": (8.5, -6), "width": 2, "height": 1.5, "color": "#e8daef"},
    {"name": "ReLU", "pos": (11.5, -6), "width": 1.5, "height": 1.5, "color": "#fadbd8"},
    {"name": "最大池化层3\n核大小: 2×2\n步长: 2", "pos": (14, -6), "width": 2.5, "height": 1.5, "color": "#fdebd0"},
    
    # 第四个卷积块
    {"name": "卷积层4 (Conv2d)\n输入: 128, 输出: 128\n核大小: 3×3, 步长: 1\npadding: same", 
     "pos": (3.5, -9), "width": 4, "height": 1.5, "color": "#d5f5e3"},
    {"name": "批归一化4\nBatchNorm2d", "pos": (8.5, -9), "width": 2, "height": 1.5, "color": "#e8daef"},
    {"name": "ReLU", "pos": (11.5, -9), "width": 1.5, "height": 1.5, "color": "#fadbd8"},
    {"name": "最大池化层4\n核大小: 2×2\n步长: 2", "pos": (14, -9), "width": 2.5, "height": 1.5, "color": "#fdebd0"},
    
    # 全连接层部分
    {"name": "展平层\nFlatten", "pos": (3.5, -12), "width": 2.5, "height": 1.5, "color": "#d7bde2"},
    {"name": "全连接层1 (Dense)\n输入: 5×5×128=3200\n输出: 256", 
     "pos": (7, -12), "width": 4, "height": 1.5, "color": "#aed6f1"},
    {"name": "ReLU", "pos": (12, -12), "width": 1.5, "height": 1.5, "color": "#fadbd8"},
    {"name": "Dropout\n(rate=0.3)", "pos": (14.5, -12), "width": 2, "height": 1.5, "color": "#f9e79f"},
    
    # 输出层
    {"name": "全连接层2 (Dense)\n输入: 256, 输出: 26", 
     "pos": (7, -15), "width": 4, "height": 1.5, "color": "#aed6f1"},
    {"name": "输出\n[N, 26]", "pos": (12, -15), "width": 2.5, "height": 1.5, "color": "#d6eaf8"}
]

# 先绘制所有组件
for comp in components:
    rect = Rectangle(comp["pos"], comp["width"], comp["height"], 
                     facecolor=comp["color"], edgecolor='black', alpha=0.8,
                     linewidth=1.5, zorder=1, joinstyle='round')
    ax.add_patch(rect)
    ax.text(comp["pos"][0] + comp["width"]/2, comp["pos"][1] + comp["height"]/2, 
            comp["name"], ha='center', va='center', fontsize=10, weight='bold')

# 添加尺寸注释(张量维度标签)
tensor_dims = [
    {"text": "[N, 3, 88, 88]", "pos": (2.7, 0.9), "ha": "right", "va": "bottom"},
    {"text": "[N, 32, 88, 88]", "pos": (7.7, 0.9), "ha": "right", "va": "bottom"},
    {"text": "[N, 32, 44, 44]", "pos": (16.7, 0.9), "ha": "right", "va": "bottom"},
    {"text": "[N, 64, 44, 44]", "pos": (7.7, -2.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 64, 22, 22]", "pos": (16.7, -2.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 128, 22, 22]", "pos": (7.7, -5.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 128, 11, 11]", "pos": (16.7, -5.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 128, 11, 11]", "pos": (7.7, -8.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 128, 5, 5]", "pos": (16.7, -8.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 3200]", "pos": (6.2, -11.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 256]", "pos": (11.2, -11.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 256]", "pos": (16.7, -11.1), "ha": "right", "va": "bottom"},
    {"text": "[N, 26]", "pos": (11.2, -14.1), "ha": "right", "va": "bottom"}
]

for dim in tensor_dims:
    ax.text(dim["pos"][0], dim["pos"][1], dim["text"], 
            ha=dim["ha"], va=dim["va"], fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#eeeeee", ec="gray", alpha=0.7))

# 绘制数据流向箭头
connections = [
    # 第一个卷积块内部连接
    {"start": (2.5, 0.75), "end": (3.5, 0.75)},
    {"start": (7.5, 0.75), "end": (8.5, 0.75)},
    {"start": (10.5, 0.75), "end": (11.5, 0.75)},
    {"start": (13, 0.75), "end": (14, 0.75)},
    
    # 第一块到第二块
    {"start": (16.5, 0.75), "end": (17.5, 0.75), "curved": True},
    {"start": (17.5, 0.75), "end": (17.5, -2.25), "curved": True},
    {"start": (17.5, -2.25), "end": (3.5, -2.25), "curved": True},
    
    # 第二个卷积块内部连接
    {"start": (7.5, -2.25), "end": (8.5, -2.25)},
    {"start": (10.5, -2.25), "end": (11.5, -2.25)},
    {"start": (13, -2.25), "end": (14, -2.25)},
    
    # 第二块到第三块
    {"start": (16.5, -2.25), "end": (17.5, -2.25), "curved": True},
    {"start": (17.5, -2.25), "end": (17.5, -5.25), "curved": True},
    {"start": (17.5, -5.25), "end": (3.5, -5.25), "curved": True},
    
    # 第三个卷积块内部连接
    {"start": (7.5, -5.25), "end": (8.5, -5.25)},
    {"start": (10.5, -5.25), "end": (11.5, -5.25)},
    {"start": (13, -5.25), "end": (14, -5.25)},
    
    # 第三块到第四块
    {"start": (16.5, -5.25), "end": (17.5, -5.25), "curved": True},
    {"start": (17.5, -5.25), "end": (17.5, -8.25), "curved": True},
    {"start": (17.5, -8.25), "end": (3.5, -8.25), "curved": True},
    
    # 第四个卷积块内部连接
    {"start": (7.5, -8.25), "end": (8.5, -8.25)},
    {"start": (10.5, -8.25), "end": (11.5, -8.25)},
    {"start": (13, -8.25), "end": (14, -8.25)},
    
    # 第四块到全连接层
    {"start": (16.5, -8.25), "end": (17.5, -8.25), "curved": True},
    {"start": (17.5, -8.25), "end": (17.5, -11.25), "curved": True},
    {"start": (17.5, -11.25), "end": (3.5, -11.25), "curved": True},
    
    # 全连接部分内部连接
    {"start": (6, -11.25), "end": (7, -11.25)},
    {"start": (11, -11.25), "end": (12, -11.25)},
    {"start": (13.5, -11.25), "end": (14.5, -11.25)},
    
    # 全连接层到输出层
    {"start": (16.5, -11.25), "end": (17.5, -11.25), "curved": True},
    {"start": (17.5, -11.25), "end": (17.5, -14.25), "curved": True},
    {"start": (17.5, -14.25), "end": (7, -14.25), "curved": True},
    
    # 输出层连接
    {"start": (11, -14.25), "end": (12, -14.25)},
]

# 绘制连接箭头
for conn in connections:
    if conn.get("curved", False):
        connectionstyle = "arc3,rad=0.2"
    else:
        connectionstyle = "arc3,rad=0"
        
    arrow = FancyArrowPatch(
        conn["start"], conn["end"],
        connectionstyle=connectionstyle,
        arrowstyle='-|>', 
        mutation_scale=15,
        lw=1.5,
        color="black",
        zorder=0
    )
    ax.add_patch(arrow)

# 设置图表标题和边界
ax.set_title('平衡版 LeNet5 网络架构流程图', fontsize=18, pad=20)
ax.set_xlim(-1, 18.5)
ax.set_ylim(-16.5, 2)
ax.axis('off')

# 添加图例
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor="#d6eaf8", edgecolor='black', label='输入/输出层'),
    Rectangle((0, 0), 1, 1, facecolor="#d5f5e3", edgecolor='black', label='卷积层'),
    Rectangle((0, 0), 1, 1, facecolor="#e8daef", edgecolor='black', label='批归一化层'),
    Rectangle((0, 0), 1, 1, facecolor="#fdebd0", edgecolor='black', label='池化层'),
    Rectangle((0, 0), 1, 1, facecolor="#fadbd8", edgecolor='black', label='激活函数'),
    Rectangle((0, 0), 1, 1, facecolor="#d7bde2", edgecolor='black', label='展平层'),
    Rectangle((0, 0), 1, 1, facecolor="#aed6f1", edgecolor='black', label='全连接层'),
    Rectangle((0, 0), 1, 1, facecolor="#f9e79f", edgecolor='black', label='Dropout层')
]
ax.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True, fontsize=10)

# 添加改进点注释
improvements = [
    "1. 更深的网络结构(4个卷积层)",
    "2. 增大输入尺寸(88×88)保留更多细节",
    "3. 加入批归一化稳定训练",
    "4. 更宽的特征通道(32→64→128→128)",
    "5. 合理的Dropout率(0.3)减轻过拟合",
    "6. 学习率预热和阶梯式衰减",
    "7. 使用Adam优化器提高稳定性"
]

plt.figtext(0.5, -0.06, "平衡版LeNet5改进特点", ha="center", fontsize=12, weight="bold")

# 分两列显示改进点
for i, imp in enumerate(improvements[:4]):
    plt.figtext(0.25, -0.09 - i*0.02, imp, ha="left", fontsize=10)

for i, imp in enumerate(improvements[4:]):
    plt.figtext(0.65, -0.09 - i*0.02, imp, ha="left", fontsize=10)

# 添加底部注释
plt.figtext(0.5, -0.18, '面向垃圾分类任务的平衡版LeNet5卷积神经网络', ha='center', fontsize=12)

# 保存和显示图表
plt.tight_layout()
plt.subplots_adjust(bottom=-0.15)
plt.savefig('BalancedLeNet5_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

print("平衡版LeNet5网络架构流程图已保存为'BalancedLeNet5_architecture.png'")