import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp

# Kepler
class LQGT_dataset(data.Dataset):
    '''
        读取HDR和SDR图像对,其中HDR为输入图像,SDR为GT图像.
        对于输入HDR图像,额外输入预处理图像作为低光、中光和高光区域的区域颜色损失的先验参考。
    '''

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']  # data_type = img
        self.paths_HDR, self.paths_SDR, self.paths_L_Ref, self.paths_M_Ref, self.paths_H_Ref = None, None, None, None, None

        self.sizes_HDR, self.paths_HDR = util.get_image_paths(self.data_type, opt['dataroot_HDR'])
        self.sizes_SDR, self.paths_SDR = util.get_image_paths(self.data_type, opt['dataroot_SDR'])
        self.sizes_L_Ref, self.paths_L_Ref = util.get_image_paths(self.data_type, opt['dataroot_L_Ref'])
        self.sizes_M_Ref, self.paths_M_Ref = util.get_image_paths(self.data_type, opt['dataroot_M_Ref'])
        self.sizes_H_Ref, self.paths_H_Ref = util.get_image_paths(self.data_type, opt['dataroot_H_Ref'])

        assert self.paths_HDR and self.paths_SDR, 'Error: HDR or SDR path is empty.'
        assert self.paths_L_Ref and self.paths_M_Ref and self.paths_H_Ref, 'Error: Ref path is empty.'

        assert len(self.paths_HDR) == len(self.paths_SDR) and \
        len(self.paths_HDR) == len(self.paths_L_Ref) and \
        len(self.paths_HDR) == len(self.paths_M_Ref) and \
        len(self.paths_HDR) == len(self.paths_H_Ref), 'Error: Datasets have different number of images.'

        self.cond_folder = opt['dataroot_cond']

    def __getitem__(self, index):
        HDR_path, SDR_path, L_Ref_path, M_Ref_path, H_Ref_path = None, None, None, None, None 
        GT_size = self.opt['GT_size']

        # get HDR image
        HDR_path = self.paths_HDR[index]
        im_HDR = util.read_img(None, HDR_path)

        # get SDR image
        SDR_path = self.paths_SDR[index]
        im_SDR = util.read_img(None, SDR_path)

        # get Ref image
        L_Ref_path = self.paths_L_Ref[index]
        M_Ref_path = self.paths_M_Ref[index]
        H_Ref_path = self.paths_H_Ref[index]

        im_L_Ref = util.read_img(None, L_Ref_path) # 读取图片并做最简单的归一化
        im_M_Ref = util.read_img(None, M_Ref_path) # 读取图片并做最简单的归一化
        im_H_Ref = util.read_img(None, H_Ref_path) # 读取图片并做最简单的归一化

        # # get condition
        cond_scale = self.opt['cond_scale']     # 4
        if self.cond_folder is not None:
            if '_' in osp.basename(HDR_path):
                cond_name = '_'.join(osp.basename(HDR_path).split('_')[:-1])+'_bicx'+str(cond_scale)+'.png'
            else: 
                cond_name = osp.basename(HDR_path).split('.')[0]+'_bicx'+str(cond_scale)+'.png'
            cond_path = osp.join(self.cond_folder, cond_name)
            cond_img = util.read_img(None, cond_path)
        else:
            cond_img = util.imresize_np(im_HDR, 1/cond_scale)

        if self.opt['phase'] == 'train':

            H_hdr, W_hdr, C = im_HDR.shape
            H_sdr, W_sdr, C = im_SDR.shape
            H_l_ref, W_l_ref, C = im_L_Ref.shape
            H_m_ref, W_m_ref, C = im_M_Ref.shape
            H_h_ref, W_h_ref, C = im_H_Ref.shape

            if H_hdr != H_sdr or H_hdr != H_l_ref or H_hdr != H_m_ref or H_hdr != H_h_ref:
                print('*******wrong image size*******:{}'.format(HDR_path))

            # randomly crop 随机裁剪
            if GT_size is not None:
                rnd_h = random.randint(0, max(0, H_hdr - GT_size))
                rnd_w = random.randint(0, max(0, W_hdr - GT_size))
                im_HDR = im_HDR[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
                im_SDR = im_SDR[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
                im_L_Ref = im_L_Ref[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
                im_M_Ref = im_M_Ref[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
                im_H_Ref = im_H_Ref[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            im_HDR, im_SDR, im_L_Ref, im_M_Ref, im_H_Ref = util.augment([im_HDR, im_SDR, im_L_Ref, im_M_Ref, im_H_Ref], 
                                                                        self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, Masks, HWC to CHW, numpy to tensor
        if im_HDR.shape[2] == 3:
            im_HDR = im_HDR[:, :, [2, 1, 0]]
            im_SDR = im_SDR[:, :, [2, 1, 0]]
            cond_img = cond_img[:, :, [2, 1, 0]]
            im_L_Ref = im_L_Ref[:, :, [2, 1, 0]]
            im_M_Ref = im_M_Ref[:, :, [2, 1, 0]]
            im_H_Ref = im_H_Ref[:, :, [2, 1, 0]]
            
        
        quantile1, quantile2, L_Mask, M_Mask, H_Mask = util.get_mask_ch3(im_HDR)

        im_HDR = torch.from_numpy(np.ascontiguousarray(np.transpose(im_HDR, (2, 0, 1)))).float()
        im_SDR = torch.from_numpy(np.ascontiguousarray(np.transpose(im_SDR, (2, 0, 1)))).float()
        cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond_img, (2, 0, 1)))).float()
        im_L_Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(im_L_Ref, (2, 0, 1)))).float()
        im_M_Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(im_M_Ref, (2, 0, 1)))).float()
        im_H_Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(im_H_Ref, (2, 0, 1)))).float()
        L_Mask = torch.from_numpy(np.ascontiguousarray(np.transpose(L_Mask, (2, 0, 1)))).float()
        M_Mask = torch.from_numpy(np.ascontiguousarray(np.transpose(M_Mask, (2, 0, 1)))).float()
        H_Mask = torch.from_numpy(np.ascontiguousarray(np.transpose(H_Mask, (2, 0, 1)))).float()


        return {'HDR':im_HDR, 'SDR':im_SDR,  'cond': cond,'L_Ref':im_L_Ref, 
                'quantile1': quantile1, 'quantile2': quantile2, 'M_Ref':im_M_Ref, 'H_Ref':im_H_Ref, 'L_Mask':L_Mask, 'M_Mask':M_Mask, 'H_Mask':H_Mask, 
                'HDR_path': HDR_path, 'SDR_path': SDR_path, 'L_Ref_path': L_Ref_path, 'M_Ref_path': M_Ref_path, 'H_Ref_path': H_Ref_path}


    def __len__(self):
        return len(self.paths_HDR)
