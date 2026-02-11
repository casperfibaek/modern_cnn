# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a PeLK (Parameter-efficient Large Kernel) implementation
# based on ConvNeXt V2, featuring peripheral large-kernel depthwise convolution
# for larger receptive field with parameter sharing, inspired by:
# "PeLK: Parameter-efficient Large Kernel ConvNets with Peripheral Convolution" (arXiv:2403.07589)
#
# The architecture uses only peripheral large-kernel convolutions without inception-style components.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import _trunc_normal_, LayerNorm, GRNwithNHWC


# ----------------------------
# Token mixer: Peripheral Large Kernel
# ----------------------------

def _make_peripheral_coords(radius: int, dense_inner: int = 2) -> List[int]:
    """
    Coordinate set for peripheral sharing along one axis.
    Keeps dense coords near centre, then powers-of-two outwards, and always includes radius.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    coords = list(range(0, min(radius, dense_inner) + 1))
    v = 1
    while v < radius:
        v *= 2
        if v > dense_inner and v < radius:
            coords.append(v)
    if radius not in coords:
        coords.append(radius)
    coords = sorted(set(coords))

    return coords


def _quantise_to_coords(a: int, coords: Sequence[int]) -> int:
    """
    Map absolute offset 'a' to the largest coordinate <= a (piecewise-constant outward quantisation).
    """
    # coords is sorted ascending, starts at 0.
    # For speed, linear scan is fine since coords is ~log(radius).
    q = 0
    for c in coords:
        if c <= a:
            q = c
        else:
            break
    return q


class PeripheralDWConv2d(nn.Module):
    """
    Peripheral large-kernel depthwise convolution with parameter sharing.

    We parameterise a compressed kernel defined on a sparse coordinate set (per axis),
    then expand to a dense K x K kernel via coordinate quantisation.

    This follows the *spirit* of PeLK's peripheral convolution (arXiv:2403.07589),
    giving large receptive field with significantly fewer parameters than a dense K x K DW conv.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 51,
        dense_inner: int = 2,
        use_kpe: bool = True,
        shared_kpe: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        radius = kernel_size // 2
        coords = _make_peripheral_coords(radius=radius, dense_inner=dense_inner)
        # Full signed coordinate list: [-..., -2, -1, 0, 1, 2, ...]
        signed = [-c for c in reversed(coords[1:])] + list(coords)
        self._signed_coords = signed
        self.P = len(signed)

        coord_to_idx = {c: i for i, c in enumerate(signed)}
        # Build index map from dense offsets to compressed grid indices.
        idx = []
        for i in range(kernel_size):
            oi = i - radius
            qi = _quantise_to_coords(abs(oi), coords)
            mi = -qi if oi < 0 else qi
            ii = coord_to_idx[mi]
            for j in range(kernel_size):
                oj = j - radius
                qj = _quantise_to_coords(abs(oj), coords)
                mj = -qj if oj < 0 else qj
                jj = coord_to_idx[mj]
                idx.append(ii * self.P + jj)
        self.register_buffer("idx_map", torch.tensor(idx, dtype=torch.long), persistent=False)

        # Compressed weights: (C, P*P) for depthwise.
        self.weight_comp = nn.Parameter(torch.zeros(dim, self.P * self.P))
        _trunc_normal_(self.weight_comp, std=0.02)

        if use_kpe:
            if shared_kpe is not None:
                self.kpe = shared_kpe
            else:
                self.kpe = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
                _trunc_normal_(self.kpe, std=0.02)
        else:
            self.kpe = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand compressed weights to dense weights for conv2d.
        # weight_flat: (C, K*K)
        idx = self.idx_map.unsqueeze(0).expand(self.dim, -1)
        weight_flat = self.weight_comp.gather(1, idx)
        weight = weight_flat.view(self.dim, 1, self.kernel_size, self.kernel_size)
        if self.kpe is not None:
            weight = weight + self.kpe
        return F.conv2d(x, weight, bias=None, stride=1, padding=self.padding, groups=self.dim)


def build_token_mixer(
    dim: int,
    cfg: Optional[Dict[str, Any]],
    shared_kpe: Optional[nn.Parameter] = None,
) -> nn.Module:
    """
    Factory for token mixers.
    cfg:
      None or {"type": "dwconv"} -> standard 7x7 depthwise conv
      {"type": "pelk_dw", ...}  -> PeripheralDWConv2d
    """
    if cfg is None or cfg.get("type", "dwconv") == "dwconv":
        k = int(cfg.get("kernel_size", 7)) if cfg else 7
        return nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)

    t = cfg["type"].lower()
    if t in ("pelk_dw", "peripheral_dw", "peripheral"):
        return PeripheralDWConv2d(
            dim=dim,
            kernel_size=int(cfg.get("kernel_size", 51)),
            dense_inner=int(cfg.get("dense_inner", 2)),
            use_kpe=bool(cfg.get("use_kpe", True)),
            shared_kpe=shared_kpe if bool(cfg.get("share_kpe", True)) else None,
        )

    raise ValueError(f"Unknown token mixer type: {cfg['type']}")


# ----------------------------
# PeLK Block and Model
# ----------------------------

class Block(nn.Module):
    """
    PeLK block (ConvNeXt V2 style with peripheral large-kernel token mixer).
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        token_mixer_cfg: Optional[Dict[str, Any]] = None,
        shared_kpe: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.dwconv = build_token_mixer(dim, token_mixer_cfg, shared_kpe=shared_kpe)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRNwithNHWC(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)                          # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)                   # (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)                   # (N, C, H, W)
        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """
    Stochastic depth per sample, as in timm.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class PeLK(nn.Module):
    """
    PeLK: Parameter-efficient Large Kernel ConvNet
    ConvNeXt V2 backbone with peripheral large-kernel token mixers.

    token_mixer_cfgs: a list of 4 dicts, one per stage, each dict passed to build_token_mixer().
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        token_mixer_cfgs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ):
        super().__init__()

        if token_mixer_cfgs is None:
            token_mixer_cfgs = [None, None, None, None]
        if len(token_mixer_cfgs) != 4:
            raise ValueError("token_mixer_cfgs must have length 4 (one per stage).")

        self.num_classes = num_classes
        self.depths = list(depths)
        self.dims = list(dims)

        # Stem and downsampling layers.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            down = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        # Stages.
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.stages = nn.ModuleList()
        cur = 0

        # Optional per-stage shared KPE parameters (only used if cfg says share_kpe=True).
        shared_kpes: List[Optional[nn.Parameter]] = [None, None, None, None]
        for si, cfg in enumerate(token_mixer_cfgs):
            if cfg is not None and str(cfg.get("type", "")).lower() in ("pelk_dw", "peripheral_dw", "peripheral"):
                if bool(cfg.get("use_kpe", True)) and bool(cfg.get("share_kpe", True)):
                    k = int(cfg.get("kernel_size", 51))
                    shared_kpe = nn.Parameter(torch.zeros(1, 1, k, k))
                    _trunc_normal_(shared_kpe, std=0.02)
                    shared_kpes[si] = shared_kpe

        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        token_mixer_cfg=token_mixer_cfgs[i],
                        shared_kpe=shared_kpes[i],
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        _trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # Global average pooling
        x = x.mean(dim=(-2, -1))  # (N, C)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ----------------------------
# Model builders
# ----------------------------

def _make_model(
    depths: Sequence[int],
    dims: Sequence[int],
    **kwargs: Any,
) -> PeLK:
    return PeLK(depths=depths, dims=dims, **kwargs)


def pelk_atto(**kwargs: Any) -> PeLK:
    return _make_model(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), **kwargs)


def pelk_femto(**kwargs: Any) -> PeLK:
    return _make_model(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), **kwargs)


def pelk_pico(**kwargs: Any) -> PeLK:
    return _make_model(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), **kwargs)


def pelk_nano(**kwargs: Any) -> PeLK:
    return _make_model(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), **kwargs)


def pelk_tiny(**kwargs: Any) -> PeLK:
    return _make_model(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)


def pelk_base(**kwargs: Any) -> PeLK:
    return _make_model(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), **kwargs)


def pelk_large(**kwargs: Any) -> PeLK:
    return _make_model(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536), **kwargs)


def pelk_huge(**kwargs: Any) -> PeLK:
    return _make_model(depths=(3, 3, 27, 3), dims=(352, 704, 1408, 2816), **kwargs)


# ----------------------------
# PeLK builders with large kernels
# ----------------------------

def _pelk_cfg(
    pelk_k0: int = 7,
    pelk_k1: int = 7,
    pelk_k2: int = 31,
    pelk_k3: int = 51,
    pelk_dense_inner: int = 2,
) -> List[Dict[str, Any]]:
    """
    Default PeLK configuration with peripheral large-kernel convolutions.
    """
    return [
        {"type": "dwconv", "kernel_size": pelk_k0} if pelk_k0 <= 7 else {"type": "pelk_dw", "kernel_size": pelk_k0, "dense_inner": pelk_dense_inner, "use_kpe": True, "share_kpe": True},
        {"type": "dwconv", "kernel_size": pelk_k1} if pelk_k1 <= 7 else {"type": "pelk_dw", "kernel_size": pelk_k1, "dense_inner": pelk_dense_inner, "use_kpe": True, "share_kpe": True},
        {"type": "pelk_dw", "kernel_size": pelk_k2, "dense_inner": pelk_dense_inner, "use_kpe": True, "share_kpe": True},
        {"type": "pelk_dw", "kernel_size": pelk_k3, "dense_inner": pelk_dense_inner, "use_kpe": True, "share_kpe": True},
    ]


def pelk_tiny_lk(
    pelk_k2: int = 31,
    pelk_k3: int = 51,
    **kwargs: Any,
) -> PeLK:
    return pelk_tiny(token_mixer_cfgs=_pelk_cfg(pelk_k2=pelk_k2, pelk_k3=pelk_k3), **kwargs)


def pelk_pico_lk(
    pelk_k2: int = 31,
    pelk_k3: int = 51,
    **kwargs: Any,
) -> PeLK:
    return pelk_pico(token_mixer_cfgs=_pelk_cfg(pelk_k2=pelk_k2, pelk_k3=pelk_k3), **kwargs)


def pelk_base_lk(
    pelk_k2: int = 31,
    pelk_k3: int = 51,
    **kwargs: Any,
) -> PeLK:
    return pelk_base(token_mixer_cfgs=_pelk_cfg(pelk_k2=pelk_k2, pelk_k3=pelk_k3), **kwargs)
