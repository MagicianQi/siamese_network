# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import copy

from utils.siamese_image_floder import SiameseImageFloder, SiameseImageTripletFloder
from utils.contrastive import ContrastiveLoss

# --------------------路径参数--------------------

# train_data_path = "/home/datasets/qishuo/siamese/train/"
# val_data_path = "/home/datasets/qishuo/siamese/val/"
# negtive_path = "/home/datasets/qishuo/siamese/negtive/"
train_data_path = "/Users/qs/PycharmProjects/siamese/imgs/"
val_data_path = "/Users/qs/PycharmProjects/siamese/val/"
negtive_path = "/Users/qs/PycharmProjects/siamese/selfie/"
train_batch_size = 10
val_batch_size = 2
num_epochs = 1
GPU_id = "cuda:3"
lr_init = 0.01

# --------------------加载数据--------------------

print("Getting data...")

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets_train = SiameseImageTripletFloder(p_root=train_data_path,
                                           n_root=negtive_path,
                                           transform=transform_train)

datasets_val = SiameseImageTripletFloder(p_root=val_data_path,
                                         n_root=negtive_path,
                                         transform=transform_train)

dataLoader_train = torch.utils.data.DataLoader(datasets_train,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=2)

dataLoader_val = torch.utils.data.DataLoader(datasets_val,
                                             batch_size=val_batch_size,
                                             shuffle=True,
                                             num_workers=2)

# --------------------全局参数--------------------

device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
train_size = len(datasets_train)
val_size = len(datasets_val)

# --------------------模型--------------------

print("Getting models...")

model = torchvision.models.resnet18(pretrained=True)
# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 100)
model = model.to(device)
model.train(mode=True)

# --------------------损失函数及优化算法--------------------

# criterion = ContrastiveLoss()
criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# --------------------训练--------------------

print("Training...")

# 临时保存最佳参数
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 20)
    # 统计训练集loss值
    running_loss = 0.0
    # 学习率衰减
    exp_lr_scheduler.step()
    # 用于打印iter轮数
    i = 0
    for inputs_1, inputs_2, inputs_3 in dataLoader_train:
        i += 1
        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)
        inputs_3 = inputs_3.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits_1 = model(inputs_1)
        logits_2 = model(inputs_2)
        logits_3 = model(inputs_3)
        # 计算损失值
        loss = criterion(logits_1, logits_2, logits_3)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        running_loss += loss.item()
        print("Epoch : " + str(epoch) + "    Iter : " + str(i) + "    Loss : " + str(loss.item()))

    train_loss = float(running_loss) / float(i)
    print("train acc  : " + str(train_loss))

    # 统计训练集loss值
    running_loss_val = 0.0
    # 用于打印iter轮数
    j = 0
    for inputs_val_1, inputs_val_2, inputs_val_3 in dataLoader_val:
        j += 1
        inputs_val_1 = inputs_val_1.to(device)
        inputs_val_2 = inputs_val_2.to(device)
        inputs_val_3 = inputs_val_3.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits_val_1 = model(inputs_val_1)
        logits_val_2 = model(inputs_val_2)
        logits_val_3 = model(inputs_val_3)
        # 计算损失值
        loss_val = criterion(logits_val_1, logits_val_2, logits_val_3)

        running_loss_val += loss_val.item()
        print("Val Epoch : " + str(epoch) + "    Iter : " + str(j) + "    Loss : " + str(loss_val.item()))

    if running_loss_val < best_loss:
        best_loss = running_loss_val
        best_model_wts = copy.deepcopy(model.state_dict())

# 加载最佳模型参数
model.load_state_dict(best_model_wts)
print(best_loss)
# 保存模型
torch.save(model.state_dict(),
           "./models/Siamese_Epoch_" + str(num_epochs) + "_loss_" + str(best_loss) + '.pkl')
