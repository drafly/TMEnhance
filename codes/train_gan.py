### GAN框架下同时训练AGCM和LE
import os
import math
import argparse
import random
import logging
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import numpy as np


def main():
    #### options
    parser = argparse.ArgumentParser()
    # parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True) 

    #### mkdir and loggers
    util.mkdir_and_rename(
        opt['path']['experiments_root'])  # rename experiment folder if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                    and 'pretrain_model' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                        screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                        screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = False      # 程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    torch.backends.cudnn.deterministic = True   # flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    current_step = 0
    start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    first_time = True
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            if first_time:
                start_time = time.time()
                first_time = False
            current_step += 1
            if current_step > total_iters:
                break
            
            #### training
            model.feed_data(train_data, 'train')
            model.optimize_parameters(current_step)
            
            #### update learning rate 设置warmup的初始学习率，默认不warmup
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log 按输出频率记录日志
            if current_step % opt['logger']['print_freq'] == 0:
                end_time = time.time()
                logs = model.get_current_log() # current_log即self.log_dict存的是损失
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, , time:{:.3f}> '.format(
                    epoch, current_step, model.get_current_learning_rate(), end_time-start_time)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)

                logger.info(message)
                start_time = time.time()

            # validation 按验证频率验证
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['HDR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data, 'val')
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['Fake_SDR'], out_type=np.uint8)  # uint16x  uint8
                    gt_img = util.tensor2img(visuals['SDR'], out_type=np.uint8)  # uint16x  uint8

                    # Save SR images for reference
                    if opt['datasets']['val']['save_img']:
                        save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    gt_img = gt_img / 255.  #65535x 255
                    sr_img = sr_img / 255.  #65535x 255
                    avg_psnr += util.calculate_psnr(sr_img, gt_img)

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()