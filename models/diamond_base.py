import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN, CoordAttBlock


def pooled_mean_std_all_channels(x, k, stride=None, eps=1e-6):
    # x: [B, C, H, W]
    stride = k if stride is None else stride

    # E[X] and E[X^2] over spatial window, then over channels
    ex  = F.avg_pool2d(x, k, stride).mean(dim=1, keepdim=True)          # [B,1,H',W']
    ex2 = F.avg_pool2d(x.square(), k, stride).mean(dim=1, keepdim=True) # [B,1,H',W']

    var = (ex2 - ex.square()).clamp_min(0.0)
    std = torch.sqrt(var + eps)

    # std: [B, 1, H//k, W//k]
    return std


class BlockStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockStem, self).__init__()

        self.activation = nn.ReLU6(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeeze = CoordAttBlock(self.out_channels, self.out_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = nn.BatchNorm2d(out_channels),

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=self.out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=1)
        self.norm3 = nn.BatchNorm2d(out_channels),


    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.squeeze(x)
        x = self.activation(x)

        return x


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.ca = CoordAttBlock(dim, dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.ca(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = identity + self.drop_path(x)
        return x


class DiamondNet(nn.Module):
    def __init__(self, *,
        in_chans=3,
        num_classes=1,
        depths=None,
        dims=None,
        drop_path_rate=None,
        head_init_scale=1.0,
    ):
        super(DiamondNet, self).__init__()

        
        self.depths = [2, 2, 2, 2] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.drop_path_rate = drop_path_rate if drop_path_rate is not None else 0.0
        self.head_init_scale = head_init_scale if head_init_scale is not None else 0

        self.pool_x2 = layers.AveragePooling2D(pool_size=2, strides=2, padding="valid")
        self.pool_x4 = layers.AveragePooling2D(pool_size=4, strides=4, padding="valid")
        self.pool_x8 = layers.AveragePooling2D(pool_size=8, strides=8, padding="valid")

        self.stem = nn.Sequential(
            BlockStem(self.in_chans, self.dims[0]),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Model output
        self.head = nn.Sequential(
            CNNBlock(self.dims[-1], self.dims[-1]),
            nn.Conv2d(self.dims[-1], self.num_classes, 1),
        )

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        raw_x_256 = x
        raw_x_128 = self.pool_x2(raw_x_256)
        raw_x_64  = self.pool_x4(raw_x_256)
        raw_x_32  = self.pool_x8(raw_x_256)

        std_x_128 = pooled_mean_std_all_channels(raw_x_256, k=2)
        std_x_64  = pooled_mean_std_all_channels(raw_x_256, k=4)
        std_x_32  = pooled_mean_std_all_channels(raw_x_256, k=8)

        x_256 = x
        x_128 = f.concat([raw_x_128, std_x_128], dim=1)
        x_64  = f.concat([raw_x_64, std_x_64], dim=1)
        x_32  = f.concat([raw_x_32, std_x_32], dim=1)

        # From 32x32 -> 256x256

        x = self.stem(x_32)


        x = self.head(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 32
    HEIGHT = 256
    WIDTH = 256

    model = DiamondNet(
        input_dim=CHANNELS,
        output_dim=1,
        input_size=BATCH_SIZE,
        depths = [3, 3, 3, 3],
        dims = [32, 48, 64, 80],
        # dims=[96, 192, 384, 768],
    )

    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
