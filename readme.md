# 项目环境配置指南

## 必要条件

在运行本项目之前，请确保您的系统满足以下条件：

1. Python 3.7 或更高版本
2. pip 包管理器

## 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/username/project.git
   cd project
   ```

2. 创建并激活虚拟环境（推荐）：
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. 安装依赖包：
   ```bash
   pip install matplotlib
   pip install mindspore
   ```

## 项目环境配置
本地： Apple M4 16G
远程： intel linux_64 8核cpu
注意！运行项目代码可能需要调整数据集的目录位置