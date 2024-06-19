# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import argparse
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import re
import random
import numpy as np
import torchvision.transforms.functional as F

def Training_DataReader(root_path):
    source_imgA = glob.glob(root_path + '/train/sourceA/*.jpg')#10544
    source_imgB = glob.glob(root_path + '/train/sourceB/*.jpg')
    gt_img = glob.glob(root_path + '/train/decisionmap/*.png')
    return source_imgA,source_imgB,gt_img
def Test_DataReader(root_path):
    source_imgA = glob.glob(root_path + '/test/sourceA/*.jpg')#5008
    source_imgB = glob.glob(root_path + '/test/sourceB/*.jpg')
    gt = glob.glob(root_path + '/test/decisionmap/*.png')
    return source_imgA,source_imgB,gt
def DataReader(root_path):
    source_imgA = glob.glob(root_path + '/train/sourceA/*.jpg')
    source_imgB = glob.glob(root_path + '/train/sourceB/*.jpg')
    gt = glob.glob(root_path + '/train/decisionmap/*.png')
    clear_img = glob.glob(root_path + '/train/FCFB/FC/*.bmp')[0:480]
    blur_img=glob.glob(root_path + '/train/FCFB/FB/*.bmp')[0:480]
    return source_imgA,source_imgB,gt,clear_img,blur_img

def Predict_DataReader(root_path):
    source_imgA = glob.glob(root_path + '/A/*.jpg')
    source_imgB = glob.glob(root_path + '/B/*.jpg')

    source_imgA.sort(
        key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file
    source_imgB.sort(
        key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file name

    return source_imgA,source_imgB

def Predict_DataLoader(args,root_path):
    sourceA_img,sourceB_img=Predict_DataReader(root_path)
    source_A_B_ImgLoader = torch.utils.data.DataLoader(
        ImageDataset_pair_predict(args,sourceA_img,sourceB_img),
        batch_size=1, shuffle=False, pin_memory=True)
    return source_A_B_ImgLoader

def Sup_DataLoader(args,sourceA_img,sourceB_img,gt_img):
    source_A_B_ImgLoader = torch.utils.data.DataLoader(
        ImageDataset_pair(args,sourceA_img,sourceB_img,gt_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return source_A_B_ImgLoader

class ImageDataset_pair_predict(Dataset):
    def __init__(self, args,imgA_list,imgB_list):
        transforms_ = [transforms.Resize((args.input_size, args.input_size)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # transforms_ = [transforms.Resize((args.input_size, args.input_size)),
        #                transforms.ToTensor(),
        #                transforms.Normalize((0.5,), (0.5,))]

        self.transform = transforms.Compose(transforms_)
        self.imgA_list=imgA_list
        self.imgB_list=imgB_list
        self.imgA_list.sort(
        key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file name
        self.imgB_list.sort(
        key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file name

    def __getitem__(self, index):
        imgA = Image.open(self.imgA_list[index]).convert('RGB')
        imgB = Image.open(self.imgB_list[index]).convert('RGB')
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        return (imgA,imgB)

    def __len__(self):
        return len(self.imgA_list)

class RandomFlipPair(object):
    def __init__(self, p=0.33):
        self.p = p

    def __call__(self, imgA, imgB, img_gt):
        if random.random() < self.p:
            imgA = F.hflip(imgA)
            imgB = F.hflip(imgB)
            img_gt = F.hflip(img_gt)
        if random.random() < self.p:
            imgA = F.vflip(imgA)
            imgB = F.vflip(imgB)
            img_gt = F.vflip(img_gt)
        return imgA, imgB, img_gt
class ImageDataset_pair(Dataset):
    def __init__(self, args,imgA_list,imgB_list,gt_list):
        transforms_ = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        transforms_mask = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()]

        self.transform = transforms.Compose(transforms_)
        self.transforms_mask=transforms.Compose(transforms_mask)
        self.transforms_flip = RandomFlipPair(p=0.33)
        self.imgA_list=imgA_list
        self.imgB_list=imgB_list
        self.gt_list=gt_list

        self.imgA_list=sorted(self.imgA_list)
        self.imgB_list=sorted(self.imgB_list)
        self.gt_list=sorted(self.gt_list)

        #保证是16的倍数
        max_len_multiple_16 = (len(self.imgA_list) // 16) * 16
        self.imgA_list = self.imgA_list[:max_len_multiple_16]
        self.imgB_list = self.imgB_list[:max_len_multiple_16]
        self.gt_list = self.gt_list[:max_len_multiple_16]

    def __getitem__(self, index):
        imgA = Image.open(self.imgA_list[index]).convert('RGB')
        imgB = Image.open(self.imgB_list[index]).convert('RGB')
        img_gt=Image.open(self.gt_list[index]).convert('L')

        # imgA, imgB, img_gt = self.transforms_flip(imgA, imgB, img_gt)

        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        img_gt= self.transforms_mask(img_gt)
        #


        return (imgA,imgB,img_gt)

    def __len__(self):
        return len(self.imgA_list)

class ImageDataset(Dataset):
    def __init__(self, args,data_list):
        transforms_ = [
                    transforms.Resize((args.input_size, args.input_size)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        # transforms_ = [
        #     transforms.Resize((args.input_size, args.input_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5), (0.5))]

        self.transform = transforms.Compose(transforms_)
        self.img_list = sorted(data_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        # img = self.swap_patches(img,num_swaps=1,patch_size=32)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)

    def swap_patches(self,img, num_swaps=1,patch_size=32):
        img = img.resize((256, 256))
        w, h = img.size
        num_patches_w = w // patch_size
        num_patches_h = h // patch_size

        for _ in range(num_swaps):
            # 随机选择两个不重叠的小块
            while True:
                patch1_x = random.randint(0, num_patches_w - 1)
                patch1_y = random.randint(0, num_patches_h - 1)
                patch2_x = random.randint(0, num_patches_w - 1)
                patch2_y = random.randint(0, num_patches_h - 1)
                if (patch1_x != patch2_x) or (patch1_y != patch2_y):
                    break

            # 计算小块在原始图像中的坐标
            x1 = patch1_x * patch_size
            y1 = patch1_y * patch_size
            x2 = patch2_x * patch_size
            y2 = patch2_y * patch_size

            # 获取两个小块的图像数据
            patch1 = np.array(img.crop((x1, y1, x1 + patch_size, y1 + patch_size)))
            patch2 = np.array(img.crop((x2, y2, x2 + patch_size, y2 + patch_size)))

            # 将两个小块交换
            img.paste(Image.fromarray(patch2), (x1, y1))
            img.paste(Image.fromarray(patch1), (x2, y2))

        return img
def My_DataLoader(args,sourceA_img,sourceB_img,gt_img,clear_img,blur_img):
    source_A_B_ImgLoader = torch.utils.data.DataLoader(
        ImageDataset_pair(args,sourceA_img,sourceB_img,gt_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    clear_ImgLoader1 = torch.utils.data.DataLoader(
        ImageDataset(args,clear_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    clear_ImgLoader2 = torch.utils.data.DataLoader(
        ImageDataset(args,clear_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    blur_ImgLoader1 = torch.utils.data.DataLoader(
        ImageDataset(args,blur_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    blur_ImgLoader2 = torch.utils.data.DataLoader(
        ImageDataset(args,blur_img),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return source_A_B_ImgLoader,clear_ImgLoader1,clear_ImgLoader2,blur_ImgLoader1,blur_ImgLoader2





import random
import numpy as np
from PIL import Image

def swap_patches(img, num_swaps=1):
    w, h = img.size
    patch_size = 128  # 32*32的小块
    num_patches_w = w // patch_size
    num_patches_h = h // patch_size

    for _ in range(num_swaps):
        # 随机选择两个不重叠的小块
        while True:
            patch1_x = random.randint(0, num_patches_w - 1)
            patch1_y = random.randint(0, num_patches_h - 1)
            patch2_x = random.randint(0, num_patches_w - 1)
            patch2_y = random.randint(0, num_patches_h - 1)
            if (patch1_x != patch2_x) or (patch1_y != patch2_y):
                break

        # 计算小块在原始图像中的坐标
        x1 = patch1_x * patch_size
        y1 = patch1_y * patch_size
        x2 = patch2_x * patch_size
        y2 = patch2_y * patch_size

        # 获取两个小块的图像数据
        patch1 = np.array(img.crop((x1, y1, x1 + patch_size, y1 + patch_size)))
        patch2 = np.array(img.crop((x2, y2, x2 + patch_size, y2 + patch_size)))

        # 将两个小块交换
        img.paste(Image.fromarray(patch2), (x1, y1))
        img.paste(Image.fromarray(patch1), (x2, y2))

    return img

# # 加载测试图像
# img_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/SG_Dataset/dataset/test_data/DUT/dut500-source/3.bmp"
# img = Image.open(img_path).resize((256,256))
#
# # 显示原始图像
# print("原始图像：")
# img.show()
#
# # 测试交换小块功能
#
# swapped_img = swap_patches(img,2)
#
# # 显示交换小块后的图像
# print("交换小块后的图像：")
# swapped_img.show()
