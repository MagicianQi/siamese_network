# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import os
import PIL

# --------------------参数--------------------

GPU_id = "cuda:3"
model_path = "./models/Siamese_Epoch_50_loss_0.0.pkl"
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
pos_root = "./imgs/"
neg_root = "./val_selfie/"

# --------------------模型--------------------

print("Getting models...")

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.to(device)
model.eval()

# --------------------生成正样本与负样本的向量文件--------------------

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def cal_vector_and_write_file(root, filename):
    name_list = []

    for name in os.listdir(root):
        if name != '.DS_Store':
            name_list.append(root + name)

    with open(filename, "w") as file:
        for filepath in name_list:
            # 加载图像
            image = PIL.Image.open(filepath)
            image = transform(image).float()
            # 升为4维，否则会报错
            image = image.unsqueeze(0)

            image = image.to(device)
            logits = model(image)

            result = logits.cpu().detach().numpy()[0]
            print(filepath)
            for value in result:
                file.write(str(value) + ",")
            file.write("\n")


cal_vector_and_write_file(pos_root, "pos.txt")
cal_vector_and_write_file(neg_root, "neg.txt")
