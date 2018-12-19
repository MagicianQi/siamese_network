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
root = "./test/"

# --------------------模型--------------------

print("Getting models...")

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.to(device)
model.eval()

# --------------------测试--------------------

print("Testing...")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

name_list = []

for name in os.listdir(root):
    if name != '.DS_Store':
        name_list.append(root + name)


def get_vector(filepath):
    result = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            s_data = line.strip().split(",")[:-1]
            s_data = [float(x) for x in s_data]
            result.append(s_data)
    return result


def dot(K, L):
    if len(K) != len(L):
        return 0

    return sum(i[0] * i[1] for i in zip(K, L))


def cal_similarity(vector, vec_list):
    num = len(vec_list)
    total_sim = 0.0
    for item in vec_list:
        total_sim += dot(vector, item)
    return total_sim / float(num)


for filepath in name_list:
    # 加载图像
    image = PIL.Image.open(filepath)
    image = transform(image).float()
    # 升为4维，否则会报错
    image = image.unsqueeze(0)

    image = image.to(device)
    logits = model(image)

    result = logits.cpu().detach().numpy()[0]
    pos = get_vector("pos.txt")
    neg = get_vector("neg.txt")

    pos_sim = cal_similarity(result, pos)
    neg_sim = cal_similarity(result, neg)
    print(filepath + "    " + str(i) + "    pos_sim : " + str(pos_sim) + "    neg_sim : " + str(neg_sim))
