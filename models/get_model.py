# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch.nn as nn
import torch

from models.ultraVmUnet import UltraLight_VM_UNet_G,UltraLight_VM_UNet_D
# from models.MSFIN import MODEL
# from models.DRPL import Generator
# from models.fcn_compare import FullConvSegmentationNet
#第一步训练所要用的网络
#FCN
# def get_G(args):
#     generator = FullConvSegmentationNet(input_channels=6, output_channels=1, layers=32)#8,16,32,64
#     # generator=Generator(input_channels=1)
#     if torch.cuda.is_available():
#         generator = generator.cuda()
#     if args.GPU_parallelism:
#         generator = nn.DataParallel(generator)
#     return generator

#BridegeMFF
def get_G(args):
    generator = UltraLight_VM_UNet_G(num_classes=1,
                                   input_channels=3,
                                   c_list=[8,16,24,32,48,64],  # [8,16,24,32,48,64]
                                   split_att='fc',
                                   bridge=True)
    # generator=Generator(input_channels=1)
    if torch.cuda.is_available():
        generator = generator.cuda()
    if args.GPU_parallelism:
        generator = nn.DataParallel(generator)
    return generator

#MSFIN
# def get_G(args):
#     generator = MODEL()
#     if torch.cuda.is_available():
#         generator = generator.cuda()
#     if args.GPU_parallelism:
#         generator = nn.DataParallel(generator)
#     return generator

#DRPL
# def get_G(args):
#     generator = Generator(12)
#     if torch.cuda.is_available():
#         generator = generator.cuda()
#     if args.GPU_parallelism:
#         generator = nn.DataParallel(generator)
#     return generator

def get_D(args):
    D = UltraLight_VM_UNet_D(num_classes=1,
                           input_channels=3,
                           c_list=[8, 16, 24, 32, 48, 64],  # [8,16,24,32,48,64]
                           split_att='fc',
                           bridge=True)
    if torch.cuda.is_available():
        D = D.cuda()
    if args.GPU_parallelism:
        D = nn.DataParallel(D)
    return D

