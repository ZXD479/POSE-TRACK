# POSE-TRACK

**POSE-TRACK** 是一个用于 [简要描述项目功能或目标] 的 Python 项目。本 README 提供了详细的安装和运行指南。

## 🏁 目录

- [简介](#简介)
- [环境要求](#环境要求)
- [安装依赖](#安装依赖)
- [运行项目](#运行项目)


## 🌟 简介

[在这里详细描述你的项目是什么，解决什么问题，以及它的核心功能。]

## 💾 环境要求

为了确保项目正常运行，请确保满足以下环境要求：

- **Python 版本**: 3.8 或更高版本
- **Conda**: 推荐使用 Conda 管理虚拟环境。
- **CUDA**: 如果需要 GPU 加速，请确保安装了兼容的 CUDA 版本（例如 CUDA 11.6）。
- **PyTorch**: 需要安装与 CUDA 版本兼容的 PyTorch。

## 🛠️ 安装依赖

以下是安装项目依赖的步骤：

1. **克隆仓库**：
   ```bash
   
   git clone https://github.com/ZXD479/Pose-Track.git 
   cd Pose-Track
   conda activate -n posetrack python=3.8
   conda activate posetrack
   pip install -r requirements.txt
   python setup.py develop
   
# Reproduce on SportsMOT Dataset

本部分介绍如何在 **SportsMOT 数据集** 上复现项目的结果。

## 📂 1. Data Preparation for Reproduction on SportsMOT Dataset

为了在 SportsMOT 数据集上复现结果，你需要下载检测（detection）和嵌入（embedding）文件，并将其放置到相应的文件夹中。

### 下载数据

请从以下链接下载检测和嵌入文件：
- [Google Drive 链接](https://drive.google.com/your-link-here )

### 文件结构

将下载的文件按照以下目录结构放置：
