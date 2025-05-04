import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_augmented_images(original_image, augmented_images, titles):
    """绘制原始图像和增强后的图像"""
    n_images = len(augmented_images) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(20, 4))
    
    # 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 显示增强后的图像
    for i, (img, title) in enumerate(zip(augmented_images, titles)):
        axes[i+1].imshow(img)
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./data_augmentation_demo.png', dpi=300)
    plt.show()

def main():
    """主函数"""
    # 获取数据集中的一个样本图像
    try:
        # 从训练集中加载一张图像
        train_dir = 'data_en/train'
        categories = os.listdir(train_dir)
        
        # 选择第一个非隐藏文件的类别
        categories = [c for c in categories if not c.startswith('.')]
        if not categories:
            print("未找到有效的类别目录")
            return
            
        category = categories[2]
        category_path = os.path.join(train_dir, category)
        
        # 获取该类别下的图像文件
        images = os.listdir(category_path)
        images = [img for img in images if img.endswith(('.jpg', '.jpeg', '.png')) and not img.startswith('.')]
        
        if not images:
            print(f"在 {category} 类别下未找到有效的图像文件")
            return
            
        # 选择第一张图像
        image_path = os.path.join(category_path, images[0])
        print(f"使用图像: {image_path} (类别: {category})")
    except Exception as e:
        print(f"加载训练集图像时出错: {e}")
        # 如果无法加载训练集图像，使用默认图像路径
        print("尝试使用默认图像路径...")
        # 请替换为你的实际图像路径
        image_path = 'data_en/train/Plastic Bottle/1.jpg'  # 假设路径
    
    # 使用PIL直接加载图像
    try:
        original_image = Image.open(image_path)
        print(f"成功加载图像: {image_path}, 大小: {original_image.size}")
        
        # 将PIL图像转换为numpy数组以便于处理
        original_array = np.array(original_image)
    except Exception as e:
        print(f"加载图像时出错: {e}")
        print("生成随机测试图像...")
        # 生成随机彩色图像
        original_array = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    
    # 手动实现数据增强
    # 1. 调整大小
    pil_image = Image.fromarray(original_array)
    resized_image = pil_image.resize((96, 96), Image.BILINEAR)
    resized_array = np.array(resized_image)
    
    # 2. 随机裁剪 (简化版)
    def random_crop(image, crop_size):
        h, w = image.shape[:2]
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])
        return image[top:top+crop_size[0], left:left+crop_size[1]]
    
    # 3. 中心裁剪
    def center_crop(image, crop_size):
        h, w = image.shape[:2]
        top = (h - crop_size[0]) // 2
        left = (w - crop_size[1]) // 2
        return image[top:top+crop_size[0], left:left+crop_size[1]]
    
    # 4. 水平翻转
    def horizontal_flip(image):
        return image[:, ::-1]
    
    # 5. 调整亮度
    def adjust_brightness(image, factor):
        # 确保是float32类型，避免溢出
        img_float = image.astype(np.float32) / 255.0
        img_bright = img_float * factor
        return np.clip(img_bright * 255.0, 0, 255).astype(np.uint8)

    # 6. 调整对比度
    def adjust_contrast(image, factor):
        img_float = image.astype(np.float32) / 255.0
        gray = np.mean(img_float, axis=2, keepdims=True)
        img_contrast = (img_float - gray) * factor + gray
        return np.clip(img_contrast * 255.0, 0, 255).astype(np.uint8)
    
    # 应用数据增强
    cropped_random = random_crop(resized_array, (88, 88))
    cropped_center = center_crop(resized_array, (88, 88))
    flipped = horizontal_flip(resized_array)
    brightened = adjust_brightness(resized_array, 1.2)  # 增加亮度20%
    contrasted = adjust_contrast(resized_array, 1.2)    # 增加对比度20%
    
    # 组合增强 - 随机组合多种增强
    combined = random_crop(resized_array, (88, 88))
    if np.random.rand() > 0.5:
        combined = horizontal_flip(combined)
    combined = adjust_brightness(combined, np.random.uniform(0.8, 1.2))
    combined = adjust_contrast(combined, np.random.uniform(0.8, 1.2))
    
    # 准备显示
    images_to_show = [
        resized_array,
        cropped_random,
        cropped_center,
        flipped,
        brightened,
        contrasted,
        combined
    ]
    
    titles = [
        "调整大小 (96x96)",
        "随机裁剪 (88x88)",
        "中心裁剪 (88x88)",
        "水平翻转",
        "增加亮度",
        "增加对比度",
        "组合增强"
    ]
    
    # 绘制图像
    plot_augmented_images(original_array, images_to_show, titles)
    
    # 创建多种增强效果示例
    plt.figure(figsize=(15, 12))
    augmented_samples = []
    
    for i in range(20):
        # 从原始图像开始
        img = resized_array.copy()
        
        # 随机裁剪
        img = random_crop(img, (88, 88))
        
        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = horizontal_flip(img)
            
        # 随机亮度调整
        img = adjust_brightness(img, np.random.uniform(0.8, 1.2))
        
        # 随机对比度调整
        img = adjust_contrast(img, np.random.uniform(0.8, 1.2))
        
        augmented_samples.append(img)
    
    # 显示多个增强后的样本
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, img in enumerate(augmented_samples):
        axes[i].imshow(img)
        axes[i].set_title(f"增强样本 #{i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./data_augmentation_samples.png', dpi=300)
    plt.show()
    
    print("数据增强演示完成。结果已保存为 data_augmentation_demo.png 和 data_augmentation_samples.png")

if __name__ == "__main__":
    main() 