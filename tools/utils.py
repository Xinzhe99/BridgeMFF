# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

from torchvision.utils import save_image
import os

def to_image(tensor,i,tag,path):
    #for i in range(32):
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path+'/{}.png'.format(str(i)+'_'+tag)
    save_image(tensor.detach(),
               fake_samples_file,
               normalize=True,
               range=(-1.,1.),
               nrow=4)

#保留mask
def to_image_mask(tensor, i,tag, path):
    image = tensor  # [i].cpu().clone()
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + '/{}.png'.format(str(i)+'_'+tag)
    save_image(image.detach(),
               fake_samples_file,
               normalize=True,
               range=(0., 1.),
               nrow=4)

def load_best_eval_log(path):
    file_path=os.path.join(path, 'best_eval_log.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            # 从文件中读取日志内容
            # 这里假设日志内容是以键值对的形式存储的，每行一个键值对，以':'分隔键和值
            log_content = f.readlines()
            log = {}
            for line in log_content:
                key, value = line.strip().split(':')
                log[key.strip()] = float(value.strip())  # 假设值是浮点数类型
        return log
    else:
        # 如果文件不存在，返回默认的日志内容
        return {'bestmae_epoch': 0, 'best_mae': 10, 'fm': 0, 'bestfm_epoch': 0, 'best_fm': 0, 'mae': 0}

def save_parameters(args,save_path,resume_mode):
    if resume_mode:
        # 构建参数文件路径
        parameters_file_path = os.path.join(save_path, f'parameters_resume_{args.resume_times}.txt')
    else:
        parameters_file_path = os.path.join(save_path, 'parameters.txt')

    # 写入参数到文件
    with open(parameters_file_path, 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

def write_loss_logs(epoch, losses, file_path):
    """
    将损失值写入到文本文件
    :param epoch: 当前 epoch
    :param total_loss_D1: loss_D1 的值
    :param total_loss_D2: loss_D2 的值
    :param total_loss_mask: loss_mask 的值
    :param total_loss_GAN1: loss_GAN1 的值
    :param total_loss_GAN2: loss_GAN2 的值
    :param file_path: 文本文件路径
    """
    # 打开文本文件以写入数据
    # 打开文本文件以写入数据
    with open(file_path, 'a') as f:
        # 写入标量数据
        f.write(f'Epoch: {epoch}, ')
        for loss_name, loss_value in losses.items():
            f.write(f'{loss_name}: {loss_value:.5f}, ')
        f.write('\n')


def write_eval_logs(epoch, mae, fmeasure, recall, precesion, file_path):
    """
    将标量值写入到文本文件
    :param epoch: 当前 epoch
    :param mae: mae 的值
    :param fmeasure: fmeasure 的值
    :param recall: recall 的值
    :param precesion: precesion 的值
    :param file_path: 文本文件路径
    """
    # 打开文本文件以写入数据
    with open(file_path, 'a') as f:
        # 写入标量数据
        f.write(f'Epoch: {epoch}, ')
        f.write(f'mae: {mae}, ')
        f.write(f'fmeasure: {fmeasure}, ')
        f.write(f'recall: {recall}, ')
        f.write(f'precesion: {precesion}\n')
