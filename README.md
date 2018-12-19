# Siamese Network

* 孪生网络实现，损失函数为对比损失。

## Pre-requisite

* python 3.6
* torch
* torchvision
* PIL

## Data

数据集为应为一对图像。若为一类图像，那么标签为1，若不是一类图像，那么标签为0。

具体实现见：/utils/siamese_image_floder.py

## 损失函数

损失函数为对比损失。若标签为1，那么损失值为欧式距离，若不是一类图像，那么损失值为(margin - 欧式距离)。

具体实现见：/utils/contrastive.py

## How to use

1. 训练：
    * `python train_siamese.py`
2. 生成正负样本特征向量：
    * `python generate_vector_file.py`
3. 测试：
    * `python testing_siamese.py`  
