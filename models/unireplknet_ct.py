import torch
import torch.nn as nn
from .utils import _trunc_normal_, DropPath, LayerNorm, GRNwithNHWC, to_2tuple, NCHWtoNHWC, NHWCtoNCHW, CoordAttBlock



def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=False)

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        self.origin_bn = nn.BatchNorm2d(channels)
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                       padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                       bias=False))
            self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(channels))

    def forward(self, x):
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out


class UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 ffn_factor=4):
        super().__init__()

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size)
            self.norm = nn.BatchNorm2d(dim)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=False)
            self.norm = nn.BatchNorm2d(dim)

        self.se = CoordAttBlock(dim, dim)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=True))
        self.pwconv2 = nn.Sequential(
            nn.Linear(ffn_dim, dim, bias=False),
            NHWCtoNCHW(),
            nn.BatchNorm2d(dim))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, x):
        return x + self.compute_residual(x)


default_UniRepLKNet_A_F_P_kernel_sizes = ((3, 3),
                                          (13, 13),
                                          (13, 13, 13, 13, 13, 13),
                                          (13, 13))
default_UniRepLKNet_N_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_T_kernel_sizes = ((3, 3, 3),
                                      (13, 13, 13),
                                      (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                                      (13, 13, 13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = ((3, 3, 3),
                                             (13, 13, 13),
                                             (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                             (13, 13, 13))
UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes
}


class UniRepLKNet(nn.Module):
    r""" UniRepLKNet
        A PyTorch impl of UniRepLKNet

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 27, 3)
        dims (int): Feature dimension at each stage. Default: (96, 192, 384, 768)
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        kernel_sizes (tuple(tuple(int))): Kernel size for each block. None means using the default settings. Default: None.
    """
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=(3, 3, 27, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 kernel_sizes=None,
                 **kwargs
                 ):
        super().__init__()

        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError('no default kernel size settings for the given depths, '
                                 'please specify kernel sizes for each block, e.g., '
                                 '((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))')
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")))

        self.stages = nn.ModuleList()

        cur = 0
        for i in range(4):
            main_stage = nn.Sequential(
                *[UniRepLKNetBlock(dim=dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                                   layer_scale_init_value=layer_scale_init_value) for j in
                  range(depths[i])])
            self.stages.append(main_stage)
            cur += depths[i]

        last_channels = dims[-1]
        self.norm = nn.LayerNorm(last_channels, eps=1e-6)  # final norm layer
        self.head = nn.Linear(last_channels, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x


# ===================== Factory functions =====================
def unireplknet_atto(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(40, 80, 160, 320), **kwargs)
    return model

def unireplknet_femto(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(48, 96, 192, 384), **kwargs)
    return model

def unireplknet_pico(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(64, 128, 256, 512), **kwargs)
    return model

def unireplknet_nano(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_N_depths, dims=(80, 160, 320, 640), **kwargs)
    return model

def unireplknet_tiny(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_T_depths, dims=(80, 160, 320, 640), **kwargs)
    return model

def unireplknet_small(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(96, 192, 384, 768), **kwargs)
    return model

def unireplknet_base(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(128, 256, 512, 1024), **kwargs)
    return model

def unireplknet_large(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(192, 384, 768, 1536), **kwargs)
    return model

def unireplknet_huge(**kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(352, 704, 1408, 2816), **kwargs)
    return model
