import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss
import models.util as util

logger = logging.getLogger('base')

class GenerationModel(BaseModel):
    def __init__(self, opt):
        super(GenerationModel, self).__init__(opt)

        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()
        
        # define losses, optimizer and scheduler
        if self.is_train:
            # G region color loss
            if train_opt['region_color_weight'] > 0:
                l_rc_type = train_opt['region_color_criterion']
                if l_rc_type == 'l1':
                    self.cri_rc = nn.L1Loss().to(self.device)
                elif l_rc_type == 'l2':
                    self.cri_rc = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_rc_type))
                self.l_rc_w = train_opt['region_color_weight']
            else:
                logger.info('Remove rc loss.')
                self.cri_rc = None


            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            
            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        # print network
        self.print_network()
        self.load()

    def feed_data(self, data, mode='train'):
        self.var_HDR = data['HDR'].to(self.device)  # HDR
        self.var_cond = data['cond'].to(self.device) # condition

        if mode == 'train':
            self.var_SDR = data['SDR'].to(self.device)  # SDR

            self.L_Ref = data['L_Ref'].to(self.device)  # 先验知识
            self.M_Ref = data['M_Ref'].to(self.device)
            self.H_Ref = data['H_Ref'].to(self.device)

            self.quantile1 = data['quantile1'].to(self.device)
            self.quantile2 = data['quantile2'].to(self.device)

            self.L_Mask = data['L_Mask'].to(self.device)
            self.M_Mask = data['M_Mask'].to(self.device)
            self.H_Mask = data['H_Mask'].to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_SDR = self.netG((self.var_HDR, self.var_cond))

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_rc: # region color loss
                l_g_low = self.cri_rc(torch.mul(self.fake_SDR, self.L_Mask), torch.mul(self.L_Ref, self.L_Mask))
                l_g_high = self.cri_rc(torch.mul(self.fake_SDR, self.H_Mask), torch.mul(self.H_Ref, self.H_Mask))
                
                
                self.quantile1 = self.quantile1.view(-1, 1, 1, 1)
                self.quantile2 = self.quantile2.view(-1, 1, 1, 1)

                ### 由于图像的特异性，可能出现分位点相减为0
                # var_HDR_norm = (self.var_HDR - self.quantile1) / (self.quantile2 - self.quantile1)
                var_HDR_norm = (self.var_HDR - self.quantile1)
                var_HDR_norm = torch.clamp(var_HDR_norm, min=1e-8)    # 设置的一个极小常值

                W = util.u_law_inverse(var_HDR_norm)
                mixed_M_Ref = torch.mul(util.u_law(self.var_HDR), W) + torch.mul(self.M_Ref, torch.ones_like(W) - W)
                
                l_g_mid = self.cri_rc(torch.mul(self.fake_SDR, self.M_Mask), torch.mul(mixed_M_Ref, self.M_Mask))

                l_g_rc = self.l_rc_w * (l_g_low + l_g_mid + l_g_high)
                l_g_total += l_g_rc
            
            pred_g_fake = self.netD(self.fake_SDR)
            l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()

        # need to forward and backward separately, since batch norm statistics differ
        # real
        pred_d_real = self.netD(self.var_SDR)
        l_d_real = self.cri_gan(pred_d_real, True)
        l_d_real.backward()
        # fake
        pred_d_fake = self.netD(self.fake_SDR.detach())  # detach to avoid BP to G
        l_d_fake = self.cri_gan(pred_d_fake, False)
        l_d_fake.backward()

        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_rc:
                self.log_dict['l_g_rc'] = l_g_rc.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        # self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        # self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
        # Kep1er
        if isinstance(pred_d_real, list):
            D_real = 0
            for item in pred_d_real:
                D_real += torch.mean(item.detach())
            D_real /= len(pred_d_real)
            self.log_dict['D_real'] = D_real
        else:
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())

        if isinstance(pred_d_fake, list):
            D_fake = 0
            for item in pred_d_fake:
                D_fake += torch.mean(item.detach())
            D_fake /= len(pred_d_fake)
            self.log_dict['D_fake'] = D_fake
        else:
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_SDR = self.netG((self.var_HDR, self.var_cond))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_SDR=True):
        out_dict = OrderedDict()
        out_dict['HDR'] = self.var_HDR.detach()[0].float().cpu()
        out_dict['Fake_SDR'] = self.fake_SDR.detach()[0].float().cpu()
        
        if need_SDR:
            out_dict['SDR'] = self.var_SDR.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
            
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)


    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
