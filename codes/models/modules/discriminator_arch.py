from torch import nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation="none"):
        super(Conv2dBlock, self).__init__()
        self.use_bias = False
        norm_dim = output_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

        if norm == 'none':
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.InstanceNorm2d(norm_dim)
        else:
            self.norm = nn.BatchNorm2d(norm_dim)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
            # assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        y = x.float()
        out = self.conv(y)
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_size, input_dim, dim, norm, last_activation, d_fully_connected, d_nlayers):
        super(Discriminator, self).__init__()
        self.model = []
        # 尺寸减半下采样
        self.model += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation="leakyReLU")]   
        # downsampling blocks
        if d_fully_connected:
            n_downsample = d_nlayers
        else:
            n_downsample = 0
            while input_size > 8:
                input_size = int(input_size / 2)
                n_downsample += 1
        for i in range(n_downsample):
            next_dim = min(dim * 2, 512)
            # 尺寸减半，通道数翻倍，最高为512
            self.model += [Conv2dBlock(dim, next_dim, 4, 2, 1, norm=norm, activation="leakyReLU")]
            dim = next_dim
            # dim *= 2
        # self.model += [nn.Conv2d(dim, 1, 4, 1, 0, bias=False)]
        self.model += [Conv2dBlock(dim, 1, 4, 1, 0, norm='none', activation='none')]
        if d_fully_connected:
            last_layer_size = input_size / 2 ** (n_downsample + 1) - 3
            print("last_layer_size",last_layer_size)
            self.model += [Flatten(),
            nn.Linear(int(last_layer_size ** 2), 1, bias=False)]
        if last_activation == "sigmoid":
            self.model += [nn.Sigmoid()]
        # self.model += [nn.Conv2d(dim, 1, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)

        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self, input_size, input_dim, dim, norm, last_activation, simpleD_maxpool, padding):
        super(SimpleDiscriminator, self).__init__()
        if padding:
            sub = 0
        else:
            sub = 2
        self.model = []
        self.model += [nn.Conv2d(input_dim, dim, 4, 2, padding=padding, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, 4, 2, padding=padding, bias=True)]
        if simpleD_maxpool:
            last_dim = dim * 2
            self.model += [nn.AdaptiveMaxPool2d((1))]
        else:
            last_dim = ((input_size // 2 - 1) // 2 - 1) ** 2
            # last_dim = ((int(input_size / 4) - sub) ** 2)
            print("last_dim",last_dim, padding)
            self.model += [nn.LeakyReLU(0.2, inplace=True),
                           nn.Conv2d(dim * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)]
        self.model += [Flatten(),
            nn.Linear(last_dim, 1, bias=False),
            nn.Sigmoid()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer="batch_norm", last_activation="none"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                   norm=norm_layer, activation="leakyReLU")]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                               norm=norm_layer, activation="leakyReLU")
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        if last_activation == "sigmoid":
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_size, d_model, input_nc, ndf=64, n_layers=3, norm_layer="batch_norm",
                 last_activation="none", num_D=3, getIntermFeat=False, d_fully_connected=False,
                 simpleD_maxpool=True, padding=1):
        '''
        多尺度鉴别器:D1,D2,D3. 三个鉴别器具有相同的架构,D2的输入是D1的输入降采样2x, D3的输入是D1的输入降采样4x.
        一方面, D2, D3引导生成器在粗尺度上生成全局伪真实图像。另一方面, D1以良好的比例引导生成器. 
        
        input_size: 输入图像的尺寸. 默认为268x268
        d_model: 鉴别器的网络模型. 默认为multiLayerD_simpleD, 即由simpleD组成的多尺度鉴别器
        input_nc: 即G_net输出的尺度, 1.
        ndf: d_down_dim(numoffilter). 默认为16.
        n_layers: 多层鉴别器的层数
        norm_layer: d_norm. 默认为none.
        last_activation: d_last_activation. 默认为Sigmoid.
        num_D: 鉴别器的数量
        simpleD_maxpool: 默认为False.
        padding: d_padding. 默认为0.
        '''
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            if "dcgan" in d_model:
                netD = Discriminator(input_size, input_nc, ndf, norm_layer, last_activation, d_fully_connected, n_layers)
                input_size = input_size // 2
            elif "patchD" in d_model:
                netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, last_activation)
            elif "simpleD" in d_model:
                netD = SimpleDiscriminator(input_size, input_nc,
                                            ndf, norm_layer, last_activation, simpleD_maxpool, padding)
                input_size = input_size // 2
            setattr(self, 'layer' + str(i), netD.model) # 给对象设置属性，如果对象不存在，则创建对象

        # self.downsample = nn.AvgPool2d(3, stride=2)#, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            # model = getattr(self, 'layer' + str(num_D - 1 - i))
            model = getattr(self, 'layer' + str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                # input_downsampled = self.downsample(input_downsampled)
                input_downsampled = F.interpolate(input_downsampled, scale_factor=0.5, mode='bicubic', align_corners=False)
        return result
