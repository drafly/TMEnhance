import torch.nn as nn
import models.modules.Base_arch as Base_arch
import models.modules.Condition_arch as Condition_arch

class Condition_and_SRResNet(nn.Module):
    def __init__(self, classifier='color_condition', cond_c=3, 
                in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu'):
        super(Condition_and_SRResNet, self).__init__()

        self.globalNet = Condition_arch.ConditionNet(classifier=classifier, cond_c=cond_c)
        self.localNet = Base_arch.SRResNet(in_nc=in_nc, out_nc=out_nc, nf=nf, 
                                  nb=nb, act_type=act_type)
    
    def forward(self, x):
        out = self.globalNet(x)
        out = self.localNet(out)
        return out
