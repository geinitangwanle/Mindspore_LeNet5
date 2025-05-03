import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore
from mindspore import nn
from mindspore import context, dataset
from mindspore.dataset import vision, transforms
from mindspore import save_checkpoint
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial']
# 设置运行模式为图模式，提高性能
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def download_mnist():
    """下载垃圾分类数据集"""
    if not os.path.exists('data_en'):
        from download import download

        url = "https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/MindStudio-pc/data_en.zip"
        path = download(url, "./", kind="zip", replace=True)
        print("数据集下载完成")

def get_class_names():
    """获取数据集的类别名称"""
    class_names = os.listdir('data_en/train')
    class_names.sort()  # 确保类别顺序一致
    return class_names

def prepare_dataset():
    """准备垃圾分类数据集，返回处理好的训练集和测试集"""
    # 获取类别名称
    class_names = get_class_names()
    num_classes = len(class_names)
    print(f"数据集包含 {num_classes} 个类别: {class_names}")
    
    # 创建类别到索引的映射
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    # 使用ImageFolderDataset加载数据
    train_dataset = dataset.ImageFolderDataset('data_en/train', class_indexing=class_to_idx, 
                                               shuffle=True, decode=True)
    test_dataset = dataset.ImageFolderDataset('data_en/test', class_indexing=class_to_idx, 
                                              shuffle=False, decode=True)
    
    print(f"训练集大小: {train_dataset.get_dataset_size()}")
    print(f"测试集大小: {test_dataset.get_dataset_size()}")
    print(f"数据集列名: {train_dataset.get_col_names()}")

    # 数据预处理和批处理
    def datapipe(ds, batch_size):
        # 图像预处理: 调整大小、缩放、归一化和维度变换
        image_transforms = [
            vision.Resize((32, 32)),  # 调整图像大小为32x32
            vision.Rescale(1.0 / 255.0, 0),  # 将像素值缩放到[0, 1]范围
            vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 标准化处理
            vision.HWC2CHW()  # 将图像格式从HWC转换为CHW
        ]
        # 标签转换为int32类型
        label_transform = transforms.TypeCast(mindspore.int32)

        # 应用数据转换
        ds = ds.map(image_transforms, 'image')
        ds = ds.map(label_transform, 'label')
        # 设置批处理大小
        ds = ds.batch(batch_size)
        return ds

    # 设置批处理大小为32
    batch_size = 32
    train_dataset = datapipe(train_dataset, batch_size)
    test_dataset = datapipe(test_dataset, batch_size)
    
    # 查看处理后的数据格式
    for image, label in test_dataset.create_tuple_iterator():
        print(f"图像形状 [N, C, H, W]: {image.shape} {image.dtype}")
        print(f"标签形状: {label.shape} {label.dtype}")
        break
        
    return train_dataset, test_dataset, num_classes

class LeNet5(nn.Cell):
    """
    修改版LeNet5模型，适用于垃圾分类
    """
    def __init__(self, num_classes=26):
        super(LeNet5, self).__init__()
        # 卷积层1，输入通道3（彩色图像），输出通道6，卷积核大小5x5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, pad_mode='valid')
        # 最大池化层1，核大小2x2，步长2
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2，输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid')
        # 最大池化层2，核大小2x2，步长2
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 展平层
        self.flatten = nn.Flatten()
        # 全连接层1，输入4x4x16=256，输出120
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        # 激活函数
        self.relu = nn.ReLU()
        # 全连接层2，输入120，输出84
        self.fc2 = nn.Dense(120, 84)
        # 输出层，输入84，输出num_classes（垃圾分类类别数）
        self.fc3 = nn.Dense(84, num_classes)

    def construct(self, x):
        # 第一个卷积-池化模块
        x = self.conv1(x)              # 输出: [batch_size, 6, 28, 28]
        x = self.relu(x)
        x = self.max_pool1(x)          # 输出: [batch_size, 6, 14, 14]
        
        # 第二个卷积-池化模块
        x = self.conv2(x)              # 输出: [batch_size, 16, 10, 10]
        x = self.relu(x)
        x = self.max_pool2(x)          # 输出: [batch_size, 16, 5, 5]
        
        # 展平操作
        x = self.flatten(x)            # 输出: [batch_size, 16*5*5]
        
        # 全连接层
        x = self.fc1(x)                # 输出: [batch_size, 120]
        x = self.relu(x)
        x = self.fc2(x)                # 输出: [batch_size, 84]
        x = self.relu(x)
        x = self.fc3(x)                # 输出: [batch_size, num_classes]
        
        return x

def train_model(model, train_dataset, test_dataset, epochs=10):
    """训练和评估模型"""
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

    # 定义前向计算函数
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # 定义梯度计算函数
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # 定义单步训练函数
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    # 定义训练循环
    def train_loop(model, dataset, epochs):
        model.set_train()  # 设置模型为训练模式
        steps = dataset.get_dataset_size()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            loss_total = 0
            
            for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data, label)
                loss_total += loss.asnumpy()
                
                if batch % 20 == 0:
                    loss_val = loss.asnumpy()
                    print(f"    batch {batch}/{steps}: loss={loss_val:.4f}")
                    
            avg_loss = loss_total / steps
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

    # 定义评估函数
    def test_loop(model, dataset):
        model.set_train(False)  # 设置模型为评估模式
        
        correct = 0
        total_samples = 0
        
        for data, label in dataset.create_tuple_iterator():
            # 前向计算
            logits = model(data)
            # 计算预测结果
            pred = logits.argmax(1)
            # 统计正确预测的样本数
            correct += (pred == label).asnumpy().sum()
            total_samples += label.shape[0]
        
        # 计算准确率
        accuracy = correct / total_samples
        print(f"测试集准确率: {accuracy*100:.2f}%")
        return accuracy

    # 执行训练和评估
    print("开始训练...")
    train_loop(model, train_dataset, epochs)
    print("训练完成，开始评估...")
    accuracy = test_loop(model, test_dataset)
    
    return accuracy

def save_model(model, filename="lenet5_garbage.ckpt"):
    """保存模型"""
    save_checkpoint(model, filename)
    print(f"模型已保存到 {filename}")

def visualize_results(model, test_dataset, class_names):
    """可视化预测结果"""
    # 获取一批测试数据
    for images, labels in test_dataset.create_tuple_iterator():
        break

    # 获取预测结果
    model.set_train(False)
    output = model(images)
    predictions = output.argmax(1).asnumpy()
    images = images.asnumpy()
    labels = labels.asnumpy()

    # 还原图像用于显示
    def denormalize(image):
        image = image.transpose(1, 2, 0)  # CHW to HWC
        image = image * 0.5 + 0.5  # 反归一化
        return np.clip(image, 0, 1)

    # 显示部分样本的预测结果
    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # 显示图像
        plt.imshow(denormalize(images[i]))
        
        # 预测正确显示绿色，错误显示红色
        if predictions[i] == labels[i]:
            color = 'green'
            plt.xlabel(f"{class_names[predictions[i]]}", color=color)
        else:
            color = 'red'
            plt.xlabel(f"预: {class_names[predictions[i]]}\n实: {class_names[labels[i]]}", color=color)
            
    plt.tight_layout()
    plt.savefig('./lenet5_garbage_predictions.png')
    plt.show()

def main():
    """主函数"""
    # 下载数据集
    download_mnist()
    
    # 准备数据集
    train_dataset, test_dataset, num_classes = prepare_dataset()
    
    # 获取类别名称
    class_names = get_class_names()
    
    # 创建模型
    model = LeNet5(num_classes=num_classes)
    print("LeNet5模型结构:")
    print(model)
    
    # 训练并评估模型
    accuracy = train_model(model, train_dataset, test_dataset, epochs=10)
    
    # 保存模型
    save_model(model)
    
    # 可视化结果
    visualize_results(model, test_dataset, class_names)
    
    print(f"模型在测试集上的最终准确率: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main() 