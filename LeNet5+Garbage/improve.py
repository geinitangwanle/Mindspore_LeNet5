import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore
from mindspore import nn
from mindspore import context, dataset
from mindspore.dataset import vision, transforms
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net

# 设置随机种子以确保结果可复现
np.random.seed(42)
mindspore.set_seed(42)

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial']

# 设置运行模式为图模式，提高性能
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

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
    
    # 数据增强和预处理
    def datapipe(ds, batch_size, is_training=True):
        # 针对训练集的数据增强
        if is_training:
            image_transforms = [
                vision.Resize((96, 96)),                          # 调整大小到96x96
                vision.RandomCrop((88, 88)),                      # 随机裁剪到88x88以保留更多细节
                vision.RandomHorizontalFlip(prob=0.5),            # 随机水平翻转
                vision.RandomColorAdjust(brightness=0.1, contrast=0.1),  # 轻微颜色调整
                vision.Rescale(1.0 / 255.0, 0),                   # 将像素值缩放到[0, 1]范围
                vision.Normalize(mean=[0.485, 0.456, 0.406],      # 标准化处理
                               std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()                                  # 转换通道顺序
            ]
        else:
            # 测试集只需要调整大小和标准化
            image_transforms = [
                vision.Resize((96, 96)),                          # 调整大小
                vision.CenterCrop((88, 88)),                      # 中心裁剪
                vision.Rescale(1.0 / 255.0, 0),
                vision.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]
            
        # 标签转换为int32类型
        label_transform = transforms.TypeCast(mindspore.int32)

        # 应用数据转换
        ds = ds.map(image_transforms, 'image')
        ds = ds.map(label_transform, 'label')
        # 设置批处理大小
        ds = ds.batch(batch_size, drop_remainder=False)
        
        # 对训练数据进行混洗
        if is_training:
            ds = ds.shuffle(buffer_size=batch_size * 10)
            
        return ds

    # 设置批处理大小
    batch_size = 32
    train_dataset = datapipe(train_dataset, batch_size, is_training=True)
    test_dataset = datapipe(test_dataset, batch_size, is_training=False)
    
    # 查看处理后的数据格式
    for image, label in test_dataset.create_tuple_iterator():
        print(f"图像形状 [N, C, H, W]: {image.shape} {image.dtype}")
        print(f"标签形状: {label.shape} {label.dtype}")
        break
        
    return train_dataset, test_dataset, num_classes

class BalancedLeNet5(nn.Cell):
    """平衡的LeNet5模型，适度增加深度和宽度"""
    def __init__(self, num_classes=26):
        super(BalancedLeNet5, self).__init__()
        
        # 第一个卷积块 - 增加通道数
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 展平层
        self.flatten = nn.Flatten()
        
        # 计算展平后的特征尺寸
        # 输入88x88经过4次池化后变为5x5
        flattened_size = 5 * 5 * 128
        
        # 全连接层
        self.fc1 = nn.Dense(flattened_size, 256)
        self.fc_relu1 = nn.ReLU()
        self.fc_dropout1 = nn.Dropout(0.3)  # 降低dropout率
        
        # 输出层
        self.fc2 = nn.Dense(256, num_classes)
        
    def construct(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 展平
        x = self.flatten(x)
        
        # 全连接层
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc_dropout1(x)
        
        # 输出层
        x = self.fc2(x)
        
        return x

def train_model(model, train_dataset, test_dataset, epochs=25):
    """训练和评估模型"""
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 学习率预热和阶梯式衰减
    lr_init = 0.0001  # 较低的初始学习率
    lr_max = 0.001    # 预热后的最大学习率
    warmup_epochs = 3  # 预热阶段的epoch数
    
    # 分段学习率调度，先预热再阶梯式衰减
    def lr_scheduler(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return lr_init + (lr_max - lr_init) * epoch / warmup_epochs
        elif epoch < 15:
            return lr_max
        elif epoch < 20:
            return lr_max * 0.1
        else:
            return lr_max * 0.01
    
    # 定义优化器 - 使用Adam优化器，较小的权重衰减
    optimizer = nn.Adam(
        params=model.trainable_params(),
        learning_rate=lr_init,  # 初始使用较小的学习率
        weight_decay=1e-5       # 降低权重衰减
    )
    
    # 定义前向计算和梯度计算函数
    def forward_fn(data, label):
        output = model(data)
        loss = loss_fn(output, label)
        return loss, output

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    
    # 定义训练步骤
    def train_step(data, label, lr):
        # 更新学习率
        for param in optimizer.parameters:
            if param.name == 'learning_rate':
                param.set_data(mindspore.Tensor(lr, mindspore.float32))
        
        (loss, output), grads = grad_fn(data, label)
        optimizer(grads)
        pred = output.argmax(1)
        acc = (pred == label).asnumpy().mean()
        return loss, acc
    
    # 用于早停的变量
    best_acc = 0
    patience = 5
    patience_counter = 0
    
    # 训练循环
    print("开始训练平衡版LeNet5模型...")
    for epoch in range(epochs):
        # 设置当前学习率
        current_lr = lr_scheduler(epoch)
        print(f"Epoch {epoch+1} - 学习率: {current_lr:.6f}")
        
        # 训练阶段
        model.set_train(True)
        train_loss = 0
        train_acc = 0
        train_steps = 0
        
        for data, label in train_dataset.create_tuple_iterator():
            loss, acc = train_step(data, label, current_lr)
            train_loss += loss.asnumpy()
            train_acc += acc
            train_steps += 1
            
            if train_steps % 20 == 0:
                print(f"  Batch {train_steps}: loss={loss.asnumpy():.4f}, acc={acc:.4f}")
            
        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / train_steps
        avg_train_acc = train_acc / train_steps
        
        # 评估阶段
        model.set_train(False)
        test_loss = 0
        correct = 0
        total = 0
        
        # 创建混淆矩阵
        num_classes = test_dataset.output_shapes()[1][0]  # 获取类别数
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        for data, label in test_dataset.create_tuple_iterator():
            output = model(data)
            loss = loss_fn(output, label)
            test_loss += loss.asnumpy()
            
            pred = output.argmax(1)
            correct += (pred == label).asnumpy().sum()
            total += label.shape[0]
            
            # 更新混淆矩阵
            for i in range(len(label)):
                confusion_matrix[label[i], pred[i]] += 1
        
        avg_test_loss = test_loss / test_dataset.get_dataset_size()
        test_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_acc:.4f}")
        print(f"  测试损失: {avg_test_loss:.4f}, 测试准确率: {test_acc:.4f}")
        
        # 计算每个类别的准确率
        per_class_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        # 筛选出表现最差的5个类别
        worst_classes = np.argsort(per_class_acc)[:5]
        print("  表现最差的5个类别:")
        class_names = get_class_names()
        for idx in worst_classes:
            if np.sum(confusion_matrix[idx]) > 0:  # 确保该类有样本
                acc = per_class_acc[idx]
                print(f"    {class_names[idx]}: {acc:.4f}")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, f"balanced_lenet5_epoch_{epoch+1}.ckpt")
            print(f"  新的最佳模型已保存! 准确率: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 每5个epoch保存一次模型，便于后续分析
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, f"balanced_lenet5_epoch_{epoch+1}_checkpoint.ckpt")
            
        # 早停检查
        if patience_counter >= patience:
            print(f"早停：连续{patience}个epoch没有改进，停止训练")
            break
    
    return best_acc

def visualize_results(model, test_dataset, class_names):
    """可视化模型预测结果"""
    model.set_train(False)
    
    # 获取一批测试数据
    for images, labels in test_dataset.create_tuple_iterator():
        break
        
    # 获取预测结果
    output = model(images)
    predictions = output.argmax(1).asnumpy()
    labels = labels.asnumpy()
    images = images.asnumpy()
    
    # 计算每个类别的准确率
    class_correct = {}
    class_total = {}
    
    for image, label, pred in zip(images, labels, predictions):
        class_name = class_names[label]
        if class_name not in class_total:
            class_total[class_name] = 0
            class_correct[class_name] = 0
            
        class_total[class_name] += 1
        if label == pred:
            class_correct[class_name] += 1
    
    # 打印每个类别的准确率
    print("\n每个类别的准确率:")
    for class_name in class_names:
        if class_name in class_total and class_total[class_name] > 0:
            accuracy = class_correct.get(class_name, 0) / class_total[class_name]
            print(f"{class_name}: {accuracy:.2f} ({class_correct.get(class_name, 0)}/{class_total[class_name]})")
    
    # 绘制图像和预测结果
    # 图像反归一化函数
    def denormalize(image):
        image = image.transpose(1, 2, 0)  # CHW to HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean  # 反归一化
        return np.clip(image, 0, 1)
    
    # 显示图像
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
    plt.savefig('./balanced_lenet5_predictions.png')
    plt.show()

def main():
    """主函数"""
    # 准备数据集
    train_dataset, test_dataset, num_classes = prepare_dataset()
    
    # 获取类别名称
    class_names = get_class_names()
    
    # 创建平衡版LeNet5模型
    model = BalancedLeNet5(num_classes=num_classes)
    print("平衡版LeNet5模型结构:")
    print(model)
    
    # 训练并评估模型
    best_acc = train_model(model, train_dataset, test_dataset, epochs=25)
    
    # 加载最佳模型
    best_model_files = [f for f in os.listdir('./') if f.startswith('balanced_lenet5_epoch_') and f.endswith('.ckpt') and not f.endswith('_checkpoint.ckpt')]
    if best_model_files:
        # 按修改时间排序，获取最新保存的最佳模型
        best_model_file = sorted(best_model_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
        model_param = load_checkpoint(best_model_file)
        load_param_into_net(model, model_param)
        print(f"已加载最佳模型: {best_model_file}")
    
    # 可视化结果
    visualize_results(model, test_dataset, class_names)
    
    print(f"模型在测试集上的最终准确率: {best_acc*100:.2f}%")
    print("\n平衡版LeNet5改进点总结:")
    print("1. 使用更适中的网络深度(4个卷积层)")
    print("2. 采用更大的输入图像尺寸(88x88)以保留更多细节")
    print("3. 降低正则化强度，更合理的Dropout率(0.3)")
    print("4. 学习率预热和阶梯式衰减策略")
    print("5. 降低权重衰减系数，减轻过拟合")
    print("6. 使用Adam优化器以稳定训练")
    print("7. 更长的训练周期(25个epoch)以达到收敛")
    print("8. 每5个epoch保存检查点以跟踪模型进展")

if __name__ == "__main__":
    main() 