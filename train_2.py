# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import numpy as np
import argparse
from tqdm import tqdm
import os
from optimizer import *
import Dataloader
import time
from models.get_model import get_G,get_D
from loss import Get_GAN_discriminator_loss
from tools.config_dir import config_dir
from tools.utils import to_image, to_image_mask, load_best_eval_log, save_parameters, write_loss_logs
import warnings
from itertools import cycle
from predict import predict
from loss import BinaryDiceLoss,BinaryFocalLoss
# 忽略所有警告
warnings.simplefilter("ignore")

if __name__=='__main__':
    time_begion=time.time()
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--save_name', default='train_runs')
    parser.add_argument('--exp_descriptions', default='demo')
    parser.add_argument('--dataset_path', default='/xxx/DUTS_MFF',help='can be replaced by your own dataset')
    parser.add_argument('--Visualization_datasets', default='/xxx/three_datasets_MFF', help='can be replaced by your own dataset')
    parser.add_argument('--pretrained_model', default='/xxx/xxx.pth', help='can be replaced by your own model')
    parser.add_argument('--epochs', type=int, default=50,help='number of epochs to train')
    parser.add_argument('--GPU_parallelism', type=bool, default=True, help='multi gpu')
    parser.add_argument('--input_size', type=int, default=256,help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,help='32 default')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--min_mask_coverage', default=0.1, type=float)
    parser.add_argument('--mask_alpha', default=1.0, type=float)
    parser.add_argument('--binarization_alpha', default=1.0, type=float)
    parser.add_argument('--optimizer', type=str, default='ADAMW', help='SGD/ADAM/ADAMW')
    parser.add_argument('--lr', type=float, default=6e-4,help='1e-4 default')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='0.9 default')
    parser.add_argument('--b1', type=float, default=0.9, help='5e-4 default')
    parser.add_argument('--b2', type=float, default=0.999, help='5e-4 default')
    parser.add_argument('--model_save_fre', type=int, default=1,help='models save frequence (default: 5)')

    args = parser.parse_args()
    print('cuda_available:', torch.cuda.is_available())

    save_root_path = config_dir(resume=False, subdir_name=args.save_name)#新创建文件夹

    best_eval_log = load_best_eval_log(save_root_path)  # 评测log读取
    save_parameters(args, save_path=save_root_path,resume_mode=args.resume_mode)  # 保存训练参数
    #固定随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #定义网络
    generator=get_G(args)
    discriminator1=get_D(args)
    discriminator2 =get_D(args)

    #读取预训练模型
    generator.load_state_dict(
        torch.load(args.pretrained_model))
    if args.resume_mode:
        generator.load_state_dict(torch.load(os.path.join(save_root_path,'Epoch_{}'.format(str(args.resume_epoch)),'models','generator.pth')))
        discriminator1.load_state_dict(torch.load(os.path.join(save_root_path,'Epoch_{}'.format(str(args.resume_epoch)),'models','discriminator1.pth')))
        discriminator2.load_state_dict(torch.load(os.path.join(save_root_path,'Epoch_{}'.format(str(args.resume_epoch)),'models','discriminator2.pth')))
        print('成功读取到checkpoint模型')

    #定义三个优化器
    optimizer_G=Get_optimizers(args,generator)
    optimizer_D1=Get_optimizers(args,discriminator1)
    optimizer_D2 = Get_optimizers(args,discriminator2)

    # #定义学习率衰减策略
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,T_max=args.epochs)
    scheduler_D1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D1,T_max=args.epochs)
    scheduler_D2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D2,T_max=args.epochs)

    # 定义训练所需要的loss
    #鉴别器loss
    criterion_GAN_discriminator = Get_GAN_discriminator_loss()
    # 定义监督训练的
    criterion1 = BinaryFocalLoss()
    criterion2 = BinaryDiceLoss()
    #生成器loss
    # criterion_GAN_generator=Get_GAN_generator_loss(args.min_mask_coverage, args.mask_alpha, args.binarization_alpha)
    #读取训练数据集
    sourceA_img,sourceB_img,gt_img,clear_img,blur_img = Dataloader.DataReader(root_path=args.dataset_path)
    source_A_B_ImgLoader, clear_img_loader1,clear_img_loader2, blur_img_loader1,blur_img_loader2=(
        Dataloader.My_DataLoader(args,sourceA_img,sourceB_img,gt_img,clear_img,blur_img))

    # 读取验证数据集
    Test_sourceA_img, Test_sourceB_img, Test_gt_img = Dataloader.Test_DataReader(root_path=args.dataset_path)
    Test_ImgLoader = Dataloader.Sup_DataLoader(args, Test_sourceA_img, Test_sourceB_img, Test_gt_img)

    # 训练
    min_test_loss = np.inf
    best_epoch = 0
    #训练
    print('开始训练')
    start_epoch = args.resume_epoch + 1 if args.resume_mode else 1
    for epoch in tqdm(range(start_epoch,args.epochs+1)):
        train_total_loss_focal=0
        train_total_loss_dice = 0
        train_total_loss_GAN1_fromA=0
        train_total_loss_GAN1_fromB=0
        train_total_loss_GAN2_fromA=0
        train_total_loss_GAN2_fromB=0


        test_total_loss_focal = 0
        test_total_loss_dice = 0
        test_total_loss=0

        total_loss_D1 = 0
        total_loss_D2 = 0

        for ((source_imgA,source_imgB,gt), clear_imgs1, clear_imgs2, blur_imgs1, blur_imgs2) in tqdm(zip(source_A_B_ImgLoader,
                                                                                        cycle(clear_img_loader1),
                                                                                        cycle(clear_img_loader2),
                                                                                        cycle(blur_img_loader1),
                                                                                        cycle(blur_img_loader2))):
            source_imgA,source_imgB,gt, clear_imgs1, clear_imgs2, blur_imgs1, blur_imgs2 = map(lambda x: x.cuda(), (source_imgA,source_imgB,gt, clear_imgs1, clear_imgs2, blur_imgs1, blur_imgs2))

            # ------------------
            #  Train Generators
            # ------------------
            patch = (1, 256, 256)
            valid = torch.ones((source_imgA.size(0), *patch), device='cuda', requires_grad=False)
            fake = torch.zeros((source_imgA.size(0), *patch), device='cuda', requires_grad=False)

            #优化器参数控制，生成器可以优化，鉴别器此时不优化
            optimizer_G.zero_grad()
            requires_grad(generator, True)
            requires_grad(discriminator1, False)
            requires_grad(discriminator2, False)

            #推理mask
            mask = generator(source_imgA,source_imgB)
            #合成图像
            Mask1 = mask

            syn_image_clear1 = Mask1 * source_imgA + (1 - Mask1) * clear_imgs1
            syn_image_blur1 = (1-Mask1) * source_imgA + Mask1 * blur_imgs1

            syn_image_clear2 = (1 - Mask1) * source_imgB + Mask1 * clear_imgs1
            syn_image_blur2 = Mask1 * source_imgB + (1-Mask1) * blur_imgs1

            fusion_image_clear = Mask1 * source_imgA + (1 - Mask1) * source_imgB
            fusion_image_blur = Mask1 * source_imgB + (1 - Mask1) * source_imgA
            #生成器需要产生让鉴别器难以区分的图像，因此优化生成器需要鉴别器的结果，而鉴别器判断真实的图片为全1。

            loss_focal=criterion1(Mask1,gt)
            loss_dice=criterion2(Mask1,gt)

            # 鉴别器1判断合成的全清晰图像
            pred_fake1_fromA = discriminator1(syn_image_clear1)
            pred_fake1_fromB = discriminator1(syn_image_clear2)

            # 鉴别器1推理合成的全清晰图像与真实的全清晰图像的差异，差异大，loss大，差异小，loss小，因此要最小化这个值，缩小合成的和真实的区别。
            loss_GAN1_fromA = criterion_GAN_discriminator(pred_fake1_fromA, valid)
            loss_GAN1_fromB = criterion_GAN_discriminator(pred_fake1_fromB, valid)

            # 鉴别器2判断合成的全模糊图像
            pred_fake2_fromA = discriminator2(syn_image_blur1)
            pred_fake2_fromB = discriminator2(syn_image_blur2)


            # 鉴别器2推理合成的全模糊图像与真实的全模糊图像的差异，差异大，loss大，差异小，loss小，因此要最小化这个值，缩小合成的和真实的区别。
            loss_GAN2_fromA = criterion_GAN_discriminator(pred_fake2_fromA, valid)
            loss_GAN2_fromB = criterion_GAN_discriminator(pred_fake2_fromB, valid)

            loss_G=20*loss_focal+loss_dice+0.0001*(loss_GAN1_fromA + loss_GAN1_fromB + loss_GAN2_fromA + loss_GAN2_fromB)

            train_total_loss_focal += loss_focal.item()
            train_total_loss_dice +=loss_dice.item()
            train_total_loss_GAN1_fromA +=loss_GAN1_fromA.item()
            train_total_loss_GAN1_fromB += loss_GAN1_fromB.item()
            train_total_loss_GAN2_fromA += loss_GAN2_fromA.item()
            train_total_loss_GAN2_fromB += loss_GAN2_fromB.item()


            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # # ---------------------

            #鉴别器1的需要区分真实的清晰图像和合成的清晰图像的区别，真实的为全1，合成的为全0，即->图片越真实，不管是清晰的还是模糊的,应该越接近1
            #优化判别器1
            optimizer_D1.zero_grad()
            requires_grad(generator, False)
            requires_grad(discriminator1, True)
            requires_grad(discriminator2, False)

            # Real loss
            pred_real1 = discriminator1(clear_imgs2)
            loss_real1 = criterion_GAN_discriminator(pred_real1, valid)

            # Fake loss
            pred_fake1_fromA = discriminator1(syn_image_clear1.detach())
            pred_fake1_fromB = discriminator1(syn_image_clear2.detach())
            loss_fake1_fromA = criterion_GAN_discriminator(pred_fake1_fromA, fake)
            loss_fake1_fromB = criterion_GAN_discriminator(pred_fake1_fromB, fake)
            # Total loss
            loss_D1 = loss_real1 + loss_fake1_fromA + loss_fake1_fromB
            total_loss_D1 += loss_D1

            loss_D1.backward()
            optimizer_D1.step()

            # 鉴别器2的需要区分真实的模糊图像和合成的模糊图像的区别，真实的为全1，合成的为全0,即->图片越真实，不管是清晰的还是模糊的,应该越接近1
            optimizer_D2.zero_grad()
            requires_grad(generator, False)
            requires_grad(discriminator1, False)
            requires_grad(discriminator2, True)

            # Real loss
            pred_real2 = discriminator2(blur_imgs2)
            loss_real2 = criterion_GAN_discriminator(pred_real2, valid)

            # Fake loss
            pred_fake2_fromA = discriminator2(syn_image_blur1.detach())
            pred_fake2_fromB = discriminator2(syn_image_blur2.detach())
            loss_fake2_fromA = criterion_GAN_discriminator(pred_fake2_fromA, fake)
            loss_fake2_fromB = criterion_GAN_discriminator(pred_fake2_fromB, fake)
            # Total loss
            loss_D2 = loss_real2 + loss_fake2_fromA + loss_fake2_fromB
            total_loss_D2 += loss_D2
            loss_D2.backward()
            optimizer_D2.step()

        # 每个epoch都输出一次loss
        nums = len(source_A_B_ImgLoader)
        print(
            "\r[Epoch%d-Train]-[D1loss:%f,D2loss:%f]-[loss_focal:%f,loss_dice:%f]-[loss_GAN1_fromA:%f,loss_GAN1_fromB:%f,loss_GAN2_fromA:%f,loss_GAN2_fromB:%f]" %
            (epoch,
             total_loss_D1 / nums,
             total_loss_D2 / nums,
             train_total_loss_focal / nums,
             train_total_loss_dice/ nums,
             train_total_loss_GAN1_fromA / nums,
             train_total_loss_GAN1_fromB / nums,
             train_total_loss_GAN2_fromA / nums,
             train_total_loss_GAN2_fromB / nums))
        losses = {
            'loss_D1': total_loss_D1 / nums,
            'loss_D2': total_loss_D2 / nums,
            'train_loss_focal': train_total_loss_focal / nums,
            'train_loss_dice':train_total_loss_dice / nums,
            'train_loss_GAN1_fromA': train_total_loss_GAN1_fromA / nums,
            'train_loss_GAN1_fromB': train_total_loss_GAN1_fromB / nums,
            'train_loss_GAN2_fromA': train_total_loss_GAN2_fromA / nums,
            'train_loss_GAN2_fromB': train_total_loss_GAN2_fromB / nums,
        }

        write_loss_logs(epoch, losses, os.path.join(save_root_path, 'loss_log.txt'))

        if epoch%args.model_save_fre==0:
            #每N个epoch保存一次结果
            checkpoint_save_path=os.path.join(save_root_path,'Epoch_{}'.format(str(epoch)))
            os.makedirs(checkpoint_save_path,exist_ok=True)
            #保存图片
            image_path=os.path.join(checkpoint_save_path, 'imgs')
            to_image(source_imgA, i=epoch, tag='inputA', path=image_path)
            to_image(source_imgB, i=epoch, tag='inputB', path=image_path)
            to_image(syn_image_clear1, i=epoch, tag='syn_clear1', path=image_path)
            to_image(syn_image_blur1, i=epoch, tag='syn_blur1', path=image_path)
            to_image(syn_image_clear2, i=epoch, tag='syn_clear2', path=image_path)
            to_image(syn_image_blur2, i=epoch, tag='syn_blur2', path=image_path)
            to_image(fusion_image_clear, i=epoch, tag='fusion_image_clear', path=image_path)
            to_image(fusion_image_blur, i=epoch, tag='fusion_image_blur', path=image_path)
            to_image_mask(mask, i=epoch, tag='mask', path=image_path)

        #验证
        generator.eval()
        with torch.no_grad():
            for (source_imgA, source_imgB, gt) in tqdm(Test_ImgLoader):
                source_imgA, source_imgB, gt = map(lambda x: x.cuda(), (source_imgA, source_imgB, gt))
                # 推理mask
                mask = generator(source_imgA, source_imgB)
                Mask1=mask

                test_focal_loss = criterion1(Mask1, gt)
                test_dice_loss=criterion2(Mask1, gt)

                test_total_loss_focal += test_focal_loss
                test_total_loss_dice +=test_dice_loss
                test_total_loss+=20*test_focal_loss+test_dice_loss
        # 每个epoch都输出一次loss
        nums = len(Test_ImgLoader)
        print("\r[Epoch%d-Test]-[loss_focal:%f,loss_dice:%f]" %(epoch,test_total_loss_focal / nums,test_total_loss_dice/ nums))
        losses = { 'test_loss_focal': test_total_loss_focal / nums,'test_loss_dice':test_total_loss_dice/ nums}
        write_loss_logs(epoch, losses, os.path.join(save_root_path, 'loss_log.txt'))

        # 记录最小的epoch
        best_save_path = os.path.join(save_root_path, 'Save_Best')
        os.makedirs(best_save_path, exist_ok=True)
        if (test_total_loss / len(Test_ImgLoader)) < min_test_loss:
            min_test_loss = test_total_loss / len(Test_ImgLoader)
            best_epoch = epoch
            best_model = generator.state_dict()
            # 保存最好模型
            # 保存验证集上loss最小的模型
            torch.save(best_model, os.path.join(best_save_path, 'best.pth'))
            with open(os.path.join(best_save_path, 'best_epoch_{}.txt'.format(str(best_epoch))), 'w') as f:
                f.write("Best epoch: %d, Min validation loss: %f" % (best_epoch, min_test_loss))


        if epoch % args.model_save_fre == 0:
            # 每N个epoch保存一次结果

            #创建不同epoch的结果保存文件夹
            os.makedirs(os.path.join(checkpoint_save_path, 'models'),exist_ok=True)
            torch.save(generator.state_dict(),os.path.join(checkpoint_save_path,'models','generator.pth'))
            torch.save(discriminator1.state_dict(),os.path.join(checkpoint_save_path,'models','discriminator1.pth'))
            torch.save(discriminator2.state_dict(),os.path.join(checkpoint_save_path,'models','discriminator2.pth'))

            # 评测
            generator_pth_path = os.path.join(checkpoint_save_path, 'models', 'generator.pth')  # 训练好的模型文件
            # Lytro
            dataset_path = os.path.join(argparse.Visualization_datasets,'Lytro')  # 用于评测的数据集路径
            test_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'Lytro', 'mask')
            opt_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'Lytro', 'opt_mask')
            fusion_save_path = os.path.join(checkpoint_save_path, 'eval', 'Lytro', 'fusion')
            os.makedirs(test_mask_save_path, exist_ok=True)  # 网络输出结果保存的路径
            predict(args, stict_path=generator_pth_path, mask_save_path=test_mask_save_path,
                    opt_mask_save_path=opt_mask_save_path, image_path=dataset_path,
                    fusion_save_path=fusion_save_path)  # 保存网络推理结果到test_mask_save_path

            # MFFW
            dataset_path = os.path.join(argparse.Visualization_datasets,'MFFW')  # 用于评测的数据集路径
            test_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFFW', 'mask')
            opt_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFFW', 'opt_mask')
            fusion_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFFW', 'fusion')
            os.makedirs(test_mask_save_path, exist_ok=True)  # 网络输出结果保存的路径
            predict(args, stict_path=generator_pth_path, mask_save_path=test_mask_save_path,
                    opt_mask_save_path=opt_mask_save_path, image_path=dataset_path,
                    fusion_save_path=fusion_save_path)  # 保存网络推理结果到test_mask_save_path

            # MFI-WHU
            dataset_path = os.path.join(argparse.Visualization_datasets,'MFI-WHU')  # 用于评测的数据集路径
            test_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFI-WHU', 'mask')
            opt_mask_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFI-WHU', 'opt_mask')
            fusion_save_path = os.path.join(checkpoint_save_path, 'eval', 'MFI-WHU', 'fusion')
            os.makedirs(test_mask_save_path, exist_ok=True)  # 网络输出结果保存的路径
            predict(args, stict_path=generator_pth_path, mask_save_path=test_mask_save_path,
                    opt_mask_save_path=opt_mask_save_path, image_path=dataset_path,
                    fusion_save_path=fusion_save_path)  # 保存网络推理结果到test_mask_save_path
        scheduler_G.step()
        scheduler_D1.step()
        scheduler_D2.step()


    print("完成训练，耗时:", (time.time()-time_begion) / 3600,' h')