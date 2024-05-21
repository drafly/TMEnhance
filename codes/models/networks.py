import torch
import models.modules.Base_arch as Base_arch
import models.modules.Condition_arch as Condition_arch
import models.modules.Condition_and_Base_arch as Condition_and_Base_arch
import models.modules.Hallucination_arch as Hallucination_arch
import models.modules.discriminator_vgg_arch as DNet_arch
import models.modules.discriminator_arch as Simple_DNet_arch
import logging

logger = logging.getLogger('base')

####################
# define network
####################

#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'ConditionNet':
        netG = Condition_arch.ConditionNet(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'SRResNet':
        netG = Base_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], 
                                  nb=opt_net['nb'], act_type=opt_net['act_type'])
    elif which_model == 'Condition_and_SRResNet':
        netG = Condition_and_Base_arch.Condition_and_SRResNet(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'],
                                                              in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], 
                                                              nb=opt_net['nb'], act_type=opt_net['act_type'])
    elif which_model == 'Hallucination_Generator':
        netG = Hallucination_arch.Hallucination_Generator(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = DNet_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'multiLayerD_simpleD':
        netD = Simple_DNet_arch.MultiscaleDiscriminator(input_size=opt_net['GT_size'], d_model=opt_net['which_model_D'], 
                                                        input_nc=opt_net['in_nc'],ndf=opt_net['nf'], n_layers=opt_net['d_nlayers'], 
                                                        norm_layer=opt_net['d_norm'], last_activation=opt_net['d_last_activation'],
                                                        num_D=opt_net['num_D'], d_fully_connected=opt_net['d_fully_connected'], 
                                                        simpleD_maxpool=opt_net['simpleD_maxpool'], padding=opt_net['d_padding'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = DNet_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
