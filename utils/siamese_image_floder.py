# -*- coding: utf-8 -*-


import os
import torch
import torch.utils.data as data
from PIL import Image
import random


def default_loader(path):
    return Image.open(path).convert('RGB')


class SiameseImageFloder(data.Dataset):
    def __init__(self, p_root, n_root, transform=None, loader=default_loader):
        p_img_list = []
        n_img_list = []
        for p_name in os.listdir(p_root):
            p_img_list.append(os.path.join(p_root, p_name))
        for n_name in os.listdir(n_root):
            n_img_list.append(os.path.join(n_root, n_name))
        img_pairs_list = []
        for i in range(len(p_img_list)):
            for j in range(len(p_img_list[i:])):
                img_pairs_list.append([p_img_list[i], p_img_list[i+j], 1])
        n_positive = len(img_pairs_list)
        if len(n_img_list) > int(n_positive / len(p_img_list)):
            n_img_list = random.sample(n_img_list, int(n_positive / len(p_img_list)))
        for img_1 in p_img_list:
            for img_2 in n_img_list:
                img_pairs_list.append([img_1, img_2, 0])

        self.img_pairs_list = img_pairs_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path_1, img_path_2, label = self.img_pairs_list[index]
        img_1 = self.loader(img_path_1)
        img_2 = self.loader(img_path_2)
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        return img_1, img_2, torch.Tensor([label])

    def __len__(self):
        return len(self.img_pairs_list)


class SiameseImageTripletFloder(data.Dataset):
    def __init__(self, p_root, n_root, transform=None, loader=default_loader):
        p_img_list = []
        n_img_list = []
        for p_name in os.listdir(p_root):
            p_img_list.append(os.path.join(p_root, p_name))
        for n_name in os.listdir(n_root):
            n_img_list.append(os.path.join(n_root, n_name))

        img_pairs_list = []
        for i in range(len(p_img_list)):
            for j in range(len(p_img_list[i:])):
                img_pairs_list.append([p_img_list[i], p_img_list[i+j]])
        
        while len(n_img_list) < len(img_pairs_list):
            n_img_list = n_img_list + n_img_list
        n_img_list = random.sample(n_img_list, len(img_pairs_list))

        for k in range(len(img_pairs_list)):
            img_pairs_list[k].append(n_img_list[k])

        self.img_pairs_list = img_pairs_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, img_path_p, img_path_n = self.img_pairs_list[index]
        img = self.loader(img_path)
        img_p = self.loader(img_path_p)
        img_n = self.loader(img_path_n)
        if self.transform is not None:
            img = self.transform(img)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)
        return img, img_p, img_n

    def __len__(self):
        return len(self.img_pairs_list)
