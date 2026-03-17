"""
Hybrid CoAtMini: CNN stages + Swin-like window attention for DR classification.

Architecture:
- Conv stem
- Stage 1-2: CNN (depthwise conv blocks)
- Stage 3-4: Window attention (Swin-like)
- Classification head

~15-25M parameters, supports 224 and 384 input.
Random initialization only (no pretrained).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (timm-style)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor = random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (timm-style)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseConvBlock(nn.Module):
    """Depthwise separable conv block (MBConv-like without expand)."""
    def __init__(self, dim: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel, stride=stride, padding=kernel // 2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn(self.dw(x)))
        return self.act(self.bn2(self.pw(x))) + x


class WindowAttention(nn.Module):
    """Multi-head self-attention within non-overlapping windows (Swin-like)."""
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """Swin-like block: window attention + FFN."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _pad_to_multiple(x: torch.Tensor, window_size: int):
    """Pad H,W to be divisible by window_size."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, H, W
    x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
    return x, H + pad_h, W + pad_w


def window_partition(x: torch.Tensor, window_size: int):
    """Split into non-overlapping windows."""
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, f"H={H} W={W} must divide window_size={window_size}"
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Merge windows back."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class HybridCoAtMini(nn.Module):
    """
    Hybrid CNN + Swin-like window attention for DR classification.

    Args:
        in_chans: Input channels (3 for RGB)
        num_classes: 5 for DR severity
        embed_dims: [64, 128, 256, 512] for stages
        depths: [2, 2, 2, 2] blocks per stage
        window_size: 7 for attention
        num_heads: 8
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 5,
        embed_dims: tuple = (64, 128, 256, 512),
        depths: tuple = (2, 2, 2, 2),
        window_size: int = 7,
        num_heads: int = 8,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.window_size = window_size

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_chans, embed_dims[0], kernel=3, stride=2),
            ConvBNReLU(embed_dims[0], embed_dims[0], kernel=3, stride=2),
        )

        # Stage 1-2: CNN
        self.stage1 = self._make_cnn_stage(embed_dims[0], embed_dims[1], depths[0], stride=2)
        self.stage2 = self._make_cnn_stage(embed_dims[1], embed_dims[2], depths[1], stride=2)

        # Stage 3-4: Window attention
        total_transformer_blocks = int(depths[2] + depths[3])
        if total_transformer_blocks > 0 and drop_path_rate > 0.0:
            dpr = torch.linspace(0.0, float(drop_path_rate), total_transformer_blocks).tolist()
        else:
            dpr = [0.0] * max(total_transformer_blocks, 0)

        self.stage3 = self._make_swin_stage(embed_dims[2], depths[2], num_heads, dpr[: depths[2]])
        self.proj34 = nn.Linear(embed_dims[2], embed_dims[3]) if embed_dims[2] != embed_dims[3] else nn.Identity()
        self.stage4 = self._make_swin_stage(embed_dims[3], depths[3], num_heads, dpr[depths[2] :])

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.drop = nn.Dropout(drop_rate)

        self._init_weights()

    def _make_cnn_stage(self, in_ch: int, out_ch: int, depth: int, stride: int):
        layers = []
        if stride > 1:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
        for _ in range(depth - 1):
            layers.append(DepthwiseConvBlock(out_ch))
        return nn.Sequential(*layers)

    def _make_swin_stage(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        drop_path_probs: Optional[list] = None,
    ):
        if drop_path_probs is None:
            drop_path_probs = [0.0] * depth
        assert len(drop_path_probs) == depth, "drop_path_probs must match depth"
        blocks = nn.ModuleList(
            [SwinBlock(dim, num_heads, self.window_size, drop_path_prob=drop_path_probs[i]) for i in range(depth)]
        )
        return blocks

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # B, 64, H/4, W/4
        x = self.stage1(x)  # B, 128, H/8, W/8
        x = self.stage2(x)  # B, 256, H/16, W/16

        # Conv to transformer: reshape to sequence
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x, H, W = _pad_to_multiple(x, self.window_size)
        H2, W2 = H // 2, W // 2
        x = F.avg_pool2d(x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)  # B, H/2, W/2, C
        x, H, W = _pad_to_multiple(x, self.window_size)

        # Stage 3: window attention (H x W, e.g. 14x14)
        for blk in self.stage3:
            x_win = window_partition(x, self.window_size)
            x_win = blk(x_win)
            x = window_reverse(x_win, self.window_size, H, W)

        # Project and downsample for stage 4
        x = self.proj34(x)
        x = F.avg_pool2d(x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        x, H, W = _pad_to_multiple(x, self.window_size)

        # Stage 4: window attention (e.g. 7x7, window 7 -> 1 window)
        for blk in self.stage4:
            x_win = window_partition(x, self.window_size)
            x_win = blk(x_win)
            x = window_reverse(x_win, self.window_size, H, W)

        x = self.norm(x)
        return x.mean(dim=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.drop(x)
        return self.head(x)


def build_hybrid_coatmini(
    num_classes: int = 5,
    drop_rate: float = 0.0,
    **kwargs
) -> HybridCoAtMini:
    """Build hybrid_coatmini model (random init, no pretrained).

    Config: embed_dims=(64,128,256,512), depths=(2,2,2,2) → ~8.5M params.
    """
    return HybridCoAtMini(
        num_classes=num_classes,
        embed_dims=(64, 128, 256, 512),
        depths=(2, 2, 2, 2),
        window_size=7,
        num_heads=8,
        drop_rate=drop_rate,
        **kwargs
    )


def build_hybrid_coat_small(
    num_classes: int = 5,
    drop_rate: float = 0.0,
    **kwargs
) -> HybridCoAtMini:
    """Build hybrid_coat_small model (random init, no pretrained).

    A moderately wider variant of hybrid_coatmini obtained by scaling
    embed_dims by 1.25x: (80,160,320,640). Everything else is identical
    (depths, window_size, num_heads), keeping the comparison clean.

    Config: embed_dims=(80,160,320,640), depths=(2,2,2,2) → ~13.3M params.
    """
    return HybridCoAtMini(
        num_classes=num_classes,
        embed_dims=(80, 160, 320, 640),
        depths=(2, 2, 2, 2),
        window_size=7,
        num_heads=8,
        drop_rate=drop_rate,
        **kwargs
    )


def build_hybrid_coat_deep(
    num_classes: int = 5,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    **kwargs
) -> HybridCoAtMini:
    """Build hybrid_coat_deep model (random init, no pretrained).

    Same width as hybrid_coatmini, but deeper: depths=(3,3,6,3) with stochastic depth
    applied to transformer stages only.

    Config: embed_dims=(64,128,256,512), depths=(3,3,6,3) → ~12–14M params (approx).
    """
    return HybridCoAtMini(
        num_classes=num_classes,
        embed_dims=(64, 128, 256, 512),
        depths=(3, 3, 6, 3),
        window_size=7,
        num_heads=8,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
