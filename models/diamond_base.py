import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_, DropPath
try:
    from .utils import LayerNorm, GRN, CoordAttBlock
except ImportError:
    from utils import LayerNorm, GRN, CoordAttBlock


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
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=self.out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=1)
        self.norm3 = nn.BatchNorm2d(out_channels)


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


class LightBlock(nn.Module):
    """Lightweight block for high-resolution stages. No 4x MLP expansion."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.pwconv(self.norm(self.dwconv(x))))


class DiamondNet(nn.Module):
    def __init__(self, *,
        in_chans=3,
        num_classes=1,
        depths=None,
        dims=None,
        drop_path_rate=0.0,
        head_init_scale=1.0,
    ):
        super(DiamondNet, self).__init__()

        self.depths = [2, 2, 6, 2] if depths is None else depths
        self.dims = [64, 128, 256, 512] if dims is None else dims
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_path_rate = drop_path_rate
        self.head_init_scale = head_init_scale

        self.dp_rates = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.stem_narrow = nn.Sequential(
            BlockStem(self.in_chans + 1, self.dims[-1]),
            LayerNorm(self.dims[-1], eps=1e-6, data_format="channels_first")
        )
        self.stem_wide = nn.Sequential(
            nn.Conv2d(self.in_chans, self.dims[0], 1),
            nn.BatchNorm2d(self.dims[0]),
        )

        # Upsample layers: bilinear 2x upsample is done inline in forward,
        # then features are concatenated with raw inputs and projected here
        # i=0: dims[1]+in_chans -> dims[0], i=1: dims[2]+(in_chans+1) -> dims[1], i=2: dims[3]+(in_chans+1) -> dims[2]
        raw_channels = [self.in_chans, self.in_chans + 1, self.in_chans + 1]
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            in_ch = self.dims[i+1] + raw_channels[i]
            upsample_layer = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.Conv2d(in_ch, self.dims[i], 1),
            )
            self.upsample_layers.append(upsample_layer)

        # Downsample layers: change channels and halve spatial resolution
        # i=0: dims[0]->dims[1], i=1: dims[1]->dims[2], i=2: dims[2]->dims[3]
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.BatchNorm2d(self.dims[i]),
                    nn.Conv2d(self.dims[i], self.dims[i+1], 2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        dp_rates = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.stages_up = nn.ModuleList()
        cur = 0
        for i in range(4):
            block_cls = LightBlock if i == 0 else Block
            stage = nn.Sequential(
                *[block_cls(dim=self.dims[i], drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages_up.append(stage)
            cur += self.depths[i]

        self.stages_down = nn.ModuleList()
        cur = 0
        for i in range(4):
            block_cls = LightBlock if i == 0 else Block
            stage = nn.Sequential(
                *[block_cls(dim=self.dims[i], drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages_down.append(stage)
            cur += self.depths[i]

        # Model output
        self.norm = nn.LayerNorm(self.dims[-1])
        self.head = nn.Linear(self.dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    @staticmethod
    def _preprocess_scale(x, k):
        """Compute pooled raw + std features at a given spatial scale."""
        raw = F.avg_pool2d(x, k, k)
        std = pooled_mean_std_all_channels(x, k=k)
        return torch.cat([raw, std], dim=1)

    def _run_stage(self, stage, x):
        if self.training:
            return checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward_features(self, x):
        x_256 = x

        # x_32 needed first for stem_narrow — compute on main thread
        x_32 = self._preprocess_scale(x, 8)

        # Fork independent work to overlap with stem_narrow + stages_up[3]
        fut_64  = torch.jit.fork(self._preprocess_scale, x, 4)
        fut_128 = torch.jit.fork(self._preprocess_scale, x, 2)
        fut_wide = torch.jit.fork(self.stem_wide, x_256)

        # ---- Up path: 32x32 -> 256x256 ----
        x = self.stem_narrow(x_32)

        x = self.stages_up[3](x)
        skip_32 = x
        x_64 = torch.jit.wait(fut_64)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upsample_layers[2](torch.cat([x, x_64], dim=1))

        x = self.stages_up[2](x)
        skip_64 = x
        x_128 = torch.jit.wait(fut_128)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upsample_layers[1](torch.cat([x, x_128], dim=1))

        x = self._run_stage(self.stages_up[1], x)
        skip_128 = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upsample_layers[0](torch.cat([x, x_256], dim=1))
        x = self._run_stage(self.stages_up[0], x)

        # ---- Down path: 256x256 -> 32x32 ----
        stem_wide_out = torch.jit.wait(fut_wide)
        x = x + stem_wide_out

        x = self._run_stage(self.stages_down[0], x)
        x = self.downsample_layers[0](x)

        x = x + skip_128
        x = self._run_stage(self.stages_down[1], x)
        x = self.downsample_layers[1](x)

        x = x + skip_64
        x = self.stages_down[2](x)
        x = self.downsample_layers[2](x)

        x = x + skip_32
        x = self.stages_down[3](x)

        return self.norm(x.mean([-2, -1]))


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 3
    HEIGHT = 256
    WIDTH = 256

    model = DiamondNet(
        in_chans=CHANNELS,
        num_classes=1,
        depths = [2, 2, 3, 2],
        dims = [32, 32, 32, 32],
    )

    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
