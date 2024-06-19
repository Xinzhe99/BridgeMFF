# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import os
from optimizer import *
import Dataloader
from models.get_model import get_G
from tools.config_dir import config_dir
from tools.utils import to_image, to_image_mask, save_parameters, write_loss_logs
import warnings
from itertools import cycle
from predict import predict
from loss import BinaryDiceLoss,BinaryFocalLoss
# 忽略所有警告
warnings.simplefilter("ignore")

if __name__=='__main__':
    time_begion=time.time()
    #训练参数配置
    parser = argparse.ArgumentParser(description='GAN_for_image_pair_FUSION')
    parser.add_argument('--save_name', default='train_runs')
    parser.add_argument('--exp_descriptions', default='Baseline test')
    parser.add_argument('--dataset_path', default='/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/DUTS_MFF_NEW',help='dataset_path')
    parser.add_argument('--epochs', type=int, default=100,help='number of epochs to train')
    parser.add_argument('--GPU_parallelism', type=bool, default=True, help='multi gpu')
    parser.add_argument('--input_size', type=int, default=256,help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,help='32 default')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--min_mask_coverage', default=0.1, type=float)
    parser.add_argument('--mask_alpha', default=1.0, type=float)
    parser.add_argument('--binarization_alpha', default=1.0, type=float)
    parser.add_argument('--optimizer', type=str, default='ADAMW', help='SGD/ADAM/ADAMW')
    parser.add_argument('--lr', type=float, default=1e-3,help='1e-3 default')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='0.9 default')
    parser.add_argument('--b1', type=float, default=0.9, help='5e-4 default')
    parser.add_argument('--b2', type=float, default=0.999, help='5e-4 default')
    #模型保存与恢复训练相关设置
    parser.add_argument('--model_save_fre', type=int, default=1,help='models save frequence (default: 5)')
    parser.add_argument('--resume_mode', type=bool, default=False)
    parser.add_argument('--resume_times', type=int, default=50)
    parser.add_argument('--resume_epoch', type=int, default=400)
    parser.add_argument('--resume_root_path', type=str,
                        default='/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_defocus_detection/GAN_for_dbd/train_runs/train_runs9')
    args = parser.parse_args()
    print('cuda_available:', torch.cuda.is_available())
    #恢复训练
    if args.resume_mode:
        print(f'开始恢复训练，恢复epoch为{args.resume_epoch}')
        save_root_path = args.resume_root_path#溯源至老文件夹
    #从头开始训练
    else:
        save_root_path = config_dir(resume=False, subdir_name=args.save_name)#新创建文件夹
    save_parameters(args, save_path=save_root_path,resume_mode=args.resume_mode)  # 保存训练参数

    #固定随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #定义网络
    generator =get_G(args)

    if args.resume_mode:
        generator.load_state_dict(torch.load(os.path.join(save_root_path,'Epoch_{}'.format(str(args.resume_epoch)),'models','generator.pth')))
        print('成功读取到checkpoint模型')

    #定义优化器
    optimizer = Get_optimizers(args, generator)
    # 定义学习率衰减策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    #读取训练数据集
    Train_sourceA_img,Train_sourceB_img,Train_gt_img= Dataloader.Training_DataReader(root_path=args.dataset_path)
    Train_ImgLoader=Dataloader.Sup_DataLoader(args, Train_sourceA_img, Train_sourceB_img,Train_gt_img)

    # 读取验证数据集
    Test_sourceA_img, Test_sourceB_img,Test_gt_img = Dataloader.Test_DataReader(root_path=args.dataset_path)
    Test_ImgLoader=Dataloader.Sup_DataLoader(args, Test_sourceA_img, Test_sourceB_img,Test_gt_img)

    #定义监督训练的损失函数
    criterion1=BinaryFocalLoss()
    criterion2=BinaryDiceLoss()

    #训练
    min_test_loss = np.inf
    best_epoch = 0

    start_epoch = args.resume_epoch + 1 if args.resume_mode else 1
    print('开始训练')
    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        total_train_loss = 0
        total_test_loss=0
        total_train_focus_loss = 0
        total_train_dice_loss = 0
        total_test_focus_loss = 0
        total_test_dice_loss = 0
        generator.train()
        for (source_imgA,source_imgB,gt) in tqdm(Train_ImgLoader):
            source_imgA,source_imgB,gt = map(lambda x: x.cuda(), (source_imgA,source_imgB,gt))

            optimizer.zero_grad()
            #推理mask
            mask = generator(source_imgA,source_imgB)
            #监督学习
            train_focus_loss=criterion1(mask,gt)
            train_dice_loss = criterion2(mask, gt)
            train_loss = 15*train_focus_loss+train_dice_loss

            total_train_focus_loss+=train_focus_loss
            total_train_dice_loss += train_dice_loss
            total_train_loss+=train_loss

            train_loss.backward()
            optimizer.step()
        print("\r[Epoch%d-train]-[focus_loss:%f]-[dice_loss:%f]" % (epoch,total_train_focus_loss/len(Train_ImgLoader),total_train_dice_loss/len(Train_ImgLoader)))

        #验证
        generator.eval()
        with torch.no_grad():
            for (source_imgA,source_imgB,gt) in tqdm(Test_ImgLoader):
                source_imgA,source_imgB,gt = map(lambda x: x.cuda(), (source_imgA,source_imgB,gt))
                #推理mask
                mask = generator(source_imgA,source_imgB)
                fusion=mask*source_imgA+(1-mask)*source_imgB
                #监督学习
                # 推理mask
                mask = generator(source_imgA, source_imgB)
                # 监督学习
                test_focus_loss = criterion1(mask, gt)
                test_dice_loss = criterion2(mask, gt)
                test_loss = 15*test_focus_loss +test_dice_loss

                total_test_focus_loss += test_focus_loss
                total_test_dice_loss += test_dice_loss
                total_test_loss += test_loss

            print("\r[Epoch%d-test]-[focus_loss:%f]-[dice_loss:%f]" % (
            epoch, total_test_focus_loss / len(Test_ImgLoader), total_test_dice_loss / len(Test_ImgLoader)))

        losses = {'train_focus_loss': total_train_focus_loss / len(Train_ImgLoader),
                  'train_dice_loss': total_train_dice_loss / len(Train_ImgLoader),
                  'test_focus_loss': total_test_focus_loss / len(Test_ImgLoader),
                  'test_dice_loss': total_test_dice_loss / len(Test_ImgLoader),}
        write_loss_logs(epoch, losses, os.path.join(save_root_path, 'loss_log.txt'))

        #记录最小的epoch
        best_save_path = os.path.join(save_root_path, 'Save_Best')
        os.makedirs(best_save_path, exist_ok=True)
        if (total_test_loss / len(Test_ImgLoader)) < min_test_loss:
            min_test_loss = total_test_loss / len(Test_ImgLoader)
            best_epoch = epoch
            best_model=generator.state_dict()
            # 保存最好模型
            # 保存验证集上loss最小的模型
            torch.save(best_model, os.path.join(best_save_path, 'best.pth'))
            with open(os.path.join(best_save_path, 'best_epoch_{}.txt'.format(str(best_epoch))), 'w') as f:
                f.write("Best epoch: %d, Min validation loss: %f" % (best_epoch, min_test_loss))

        #保存图片
        if epoch%args.model_save_fre==0:
            #每N个epoch保存一次结果
            checkpoint_save_path=os.path.join(save_root_path,'Epoch_{}'.format(str(epoch)))
            os.makedirs(checkpoint_save_path,exist_ok=True)
            #保存图片
            image_path=os.path.join(checkpoint_save_path, 'imgs')
            to_image(source_imgA, i=epoch, tag='inputA', path=image_path)
            to_image(source_imgB, i=epoch, tag='inputB', path=image_path)
            to_image(gt, i=epoch, tag='gt', path=image_path)
            to_image_mask(mask, i=epoch, tag='mask', path=image_path)
            #创建不同epoch的结果保存文件夹
            os.makedirs(os.path.join(checkpoint_save_path, 'models'),exist_ok=True)
            torch.save(generator.state_dict(),os.path.join(checkpoint_save_path,'models','generator.pth'))

            #评测
            generator_pth_path = os.path.join(checkpoint_save_path,'models','generator.pth')#训练好的模型文件

            #Lytro
            dataset_path = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/Lytro'#用于评测的数据集路径
            test_mask_save_path = os.path.join(checkpoint_save_path, 'eval','Lytro','mask')
            opt_mask_save_path = os.path.join(checkpoint_save_path, 'eval','Lytro', 'opt_mask')
            fusion_save_path= os.path.join(checkpoint_save_path, 'eval','Lytro','fusion')
            os.makedirs(test_mask_save_path,exist_ok=True)#网络输出结果保存的路径
            predict(args,stict_path=generator_pth_path, mask_save_path=test_mask_save_path, opt_mask_save_path=opt_mask_save_path,image_path=dataset_path,fusion_save_path=fusion_save_path)#保存网络推理结果到test_mask_save_path

            #MFFW
            dataset_path = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFFW'  # 用于评测的数据集路径
            test_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFFW', 'mask')
            opt_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFFW', 'opt_mask')
            fusion_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFFW', 'fusion')
            os.makedirs(test_mask_save_path, exist_ok=True)  # 网络输出结果保存的路径
            predict(args, stict_path=generator_pth_path, mask_save_path=test_mask_save_path,
                    opt_mask_save_path=opt_mask_save_path, image_path=dataset_path,
                    fusion_save_path=fusion_save_path)  # 保存网络推理结果到test_mask_save_path

            # MFI-WHU
            dataset_path = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/three_datasets_MFF/MFI-WHU'  # 用于评测的数据集路径
            test_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFI-WHU', 'mask')
            opt_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFI-WHU', 'opt_mask')
            fusion_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFI-WHU', 'fusion')
            os.makedirs(test_mask_save_path, exist_ok=True)  # 网络输出结果保存的路径
            predict(args, stict_path=generator_pth_path, mask_save_path=test_mask_save_path,
                    opt_mask_save_path=opt_mask_save_path, image_path=dataset_path,
                    fusion_save_path=fusion_save_path)  # 保存网络推理结果到test_mask_save_path

        scheduler.step()
    print("完成训练，耗时:", (time.time()-time_begion) / 3600,' h')
