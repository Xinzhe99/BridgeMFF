# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg16 import VGG16
def Get_GAN_discriminator_loss():
    criterion_GAN_discriminator = torch.nn.BCELoss()
    # criterion_pixelwise = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion_GAN_discriminator.cuda()
        # criterion_pixelwise.cuda()
    return criterion_GAN_discriminator

class Get_GAN_generator_loss(nn.Module):
    def __init__(self,min_mask_coverage, mask_alpha, binarization_alpha):
        super().__init__()
        self.min_mask_coverage = min_mask_coverage
        self.mask_alpha = mask_alpha
        self.bin_alpha = binarization_alpha
    def binarization_loss(self,mask):
        return torch.min(1 - mask, mask).mean()

    def min_mask_loss(self,mask):
        '''
        One object mask per channel in this case
        '''
        return F.relu(self.min_mask_coverage - mask.mean(dim=(2, 3))).mean()

    def forward(self,mask):
        binarization_loss = self.binarization_loss(mask)#诱导二值化
        min_loss = self.min_mask_loss(mask)#loss定义最小检测出的区域覆盖全图的比例
        return self.mask_alpha * min_loss + self.bin_alpha * binarization_loss, dict(min_mask_loss=min_loss, bin_loss=binarization_loss)

def feather_loss(model_vgg,syn_celar,clear,syn_blur,blur):
    model_vgg.eval()
    feather_syn_c, output_c = model_vgg(syn_celar)
    feather_c,output_cc = model_vgg(clear)
    # print(feather_syn_c.size())
    similarity=torch.cosine_similarity(feather_c,feather_syn_c,dim=1)
    loss_c=torch.mean(1-similarity,dim=0)
    # loss_c=l1loss(feather_syn_c,feather_c)

    feather_syn_b, output_b = model_vgg(syn_blur)
    feather_b, output_bb = model_vgg(blur)

    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
    loss_b = torch.mean(1 - similarity2, dim=0)


    similarity_diff1=torch.cosine_similarity(feather_c,feather_syn_b,dim=1)
    similarity_diff2 = torch.cosine_similarity(feather_b, feather_syn_c, dim=1)
    similarity_diff=torch.mean(similarity_diff1**2 + similarity_diff2**2)
    #
    # _, predictionb = torch.max(output_b.data, 1)#0
    # _, predictionbb = torch.max(output_bb.data, 1)  #
    # _, predictionc = torch.max(output_c.data, 1)#2
    # nb=0
    # for i in range(0,len(predictionb)):
    #     if predictionb[i]==0:
    #         nb+=1
    # nc = 0
    # for i in range(0, len(predictionc)):
    #     if predictionc[i] == 2:
    #         nc += 1
    nc=0
    nb=1
    return loss_c,loss_b,similarity_diff,nc,nb




class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class BinaryFocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input
        # 如果模型没有做sigmoid的话，这里需要加上
        # logits = torch.sigmoid(logits)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()

#
# # 创建模拟数据
# predict = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
# target = torch.tensor([[1, 0], [1, 1]])
#
# # 计算BinaryDiceLoss
# dice_loss = BinaryDiceLoss()
# dice_loss_value = dice_loss(predict, target)
#
# # 计算BinaryFocalLoss
# focal_loss = BinaryFocalLoss()
# focal_loss_value = focal_loss(predict, target)
#
# print("Binary Dice Loss:", dice_loss_value.item())
# print("Binary Focal Loss:", focal_loss_value.item())