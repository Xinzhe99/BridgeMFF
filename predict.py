# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import torch
import argparse
from Dataloader import Predict_DataLoader
from tools.config_dir import config_dir
import os.path
import cv2
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from models.get_model import get_G
import Dataloader


def optimize_mask(mask):
    mask_opt = np.zeros_like(mask)
    mask_opt[mask >= 0.5] = 1
    mask_opt[mask < 0.5] = 0
    return mask_opt
def predict(args,image_path,stict_path,mask_save_path,opt_mask_save_path,fusion_save_path):
    os.makedirs(mask_save_path, exist_ok=True)
    os.makedirs(fusion_save_path, exist_ok=True)
    os.makedirs(opt_mask_save_path, exist_ok=True)
    generator = get_G(args)
    generator.load_state_dict(torch.load(stict_path))
    generator.eval()
    # image_path.sort(
    #     key=lambda x: int(str(re.findall("\d+", x.split('/')[-1])[-1])))  # Sort by the number in the file name
    dataloder = Predict_DataLoader(args,image_path)
    sourceA_img_list, sourceB_img_list = Dataloader.Predict_DataReader(image_path)
    # print(sourceA_img_list)

    for i,(imgA,imgB) in tqdm(enumerate(dataloder)):
        imgA=Variable(imgA).cuda()
        imgB = Variable(imgB).cuda()
        mask = generator(imgA,imgB).detach().cpu().numpy()[0, 0, :, :]#numpy[0-1]


        mask_opt = optimize_mask(mask)

        img1 = cv2.imread(sourceA_img_list[i]).astype(np.float32) / 255.0
        img2 = cv2.imread(sourceB_img_list[i]).astype(np.float32) / 255.0
        target_size = (img2.shape[1], img2.shape[0])  # (宽, 高)

        # 调整 mask 的大小为与 img2 相同
        mask = cv2.resize(mask, target_size)
        mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        mask_opt = cv2.resize(mask_opt, target_size)
        mask_opt = np.repeat(mask_opt[:, :, np.newaxis], 3, axis=2)

        fusion_result=img1*mask_opt+img2*(1-mask_opt)
        fusion_result=(fusion_result * 255).astype(np.uint8)

        mask = (mask * 255).astype(np.uint8)
        mask_opt = (mask_opt * 255).astype(np.uint8)
        # 保存图像
        cv2.imwrite(os.path.join(mask_save_path,'{}.png'.format(i+1)), mask)
        cv2.imwrite(os.path.join(opt_mask_save_path, '{}.png'.format(i + 1)), mask_opt)
        cv2.imwrite(os.path.join(fusion_save_path, '{}.jpg'.format(i + 1)), fusion_result)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--input_size', type=tuple, default=256, help='number of epochs to train')
    parser.add_argument('--model_path', default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_pair_fusion/gan_mff_my/train_runs/train_runs131/Epoch_21/models/generator.pth')
    parser.add_argument('--batch_size', type=int, default=32, help='32 default')
    parser.add_argument('--GPU_parallelism', type=bool, default=True, help='multi gpu')
    parser.add_argument('--Generator_only', type=bool, default=True, help='multi gpu')
    parser.add_argument('--test_dataset_path', default=r'./xxx')
    args = parser.parse_args()
    predict_save_path = config_dir(resume=False, subdir_name='predict_run')
    mask_path = os.path.join(predict_save_path, 'mask')
    fusion_result_path = os.path.join(predict_save_path, 'fusion')
    opt_mask_save_path = os.path.join(predict_save_path, 'opt_mask')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)
    predict(args,image_path=args.test_dataset_path,
         stict_path=args.model_path,
         mask_save_path=mask_path,
        opt_mask_save_path=opt_mask_save_path,
         fusion_save_path=fusion_result_path)