# Siamese Network

* 孪生网络实现，损失函数为对比损失。

## Pre-requisite

* python 3.6
* torch
* torchvision
* PIL

## Data

* 对比损失 ：数据集为应为一对图像。
* Triplet Loss ：需要输入为3个样本，img1为样本图像、img2为same图像、img3为different图像。

具体实现见：/utils/siamese_image_floder.py

## How to use

1. 训练：
    * `python train_siamese.py`
2. 生成正负样本特征向量：
    * `python generate_vector_file.py`
3. 测试：
    * `python testing_siamese.py`  
