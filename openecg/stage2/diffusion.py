"""Minimal 1D conditional UNet for DDPM training on ECG signals.

Standard ε-prediction architecture:
  - Sinusoidal time embedding → MLP
  - 4-level UNet with skip connections (channels 64/128/256/256)
  - ResBlock = GroupNorm + SiLU + Conv1d (×2) + time-emb gate
  - Self-attention at the deepest level
  - Strided conv for downsample, transpose conv for upsample

Input/Output: ``x`` shape ``[B, 1, T]``, ``t`` shape ``[B]`` (int).
Output is the predicted noise ε, same shape as ``x``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for an integer timestep.

    Returns shape ``[B, dim]``. Matches the standard transformer pos-enc
    formula used in DDPM (https://arxiv.org/abs/2006.11239).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_emb_dim: int,
                 groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, c_out)
        self.skip = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out
                      else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Channel-axis multi-head self-attention applied along the time
    dim. Used at the deepest UNet level where T is small (~156 frames)
    so quadratic attention cost is manageable."""

    def __init__(self, channels: int, n_heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.n_heads = n_heads
        assert channels % n_heads == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3C, T]
        q, k, v = qkv.chunk(3, dim=1)
        # → [B, n_heads, dim_head, T]
        q = q.view(B, self.n_heads, C // self.n_heads, T)
        k = k.view(B, self.n_heads, C // self.n_heads, T)
        v = v.view(B, self.n_heads, C // self.n_heads, T)
        # Promote attention to fp32 for numerical stability under bf16
        # autocast — softmax over T=156 frames can blow up otherwise.
        with torch.cuda.amp.autocast(enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            attn = torch.einsum("bhct,bhcs->bhts", qf, kf) / math.sqrt(C // self.n_heads)
            attn = attn.softmax(dim=-1)
            out = torch.einsum("bhts,bhcs->bhct", attn, vf)
        out = out.to(x.dtype).reshape(B, C, T)
        return x + self.proj(out)


class UNet1DDDPM(nn.Module):
    """4-level 1D UNet for DDPM noise prediction on ECG signals.

    Default config targets ``T=2500`` (10s @ 250 Hz) signals and is
    ~12 M params with channels (64, 128, 256, 256).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Tuple[int, ...] = (64, 128, 256, 256),
        t_emb_dim: int = 256,
        groups: int = 8,
        num_classes: int = 0,
    ):
        """``num_classes=0`` (default) keeps the model unconditional —
        compatible with the existing Lydus-only training script.
        ``num_classes>0`` enables class-conditional generation: a learned
        ``nn.Embedding`` of size ``(num_classes + 1, t_emb_dim)`` is
        added to the time embedding. The +1 slot is the "null class"
        used at inference for classifier-free guidance and during training
        for the CFG dropout regime (caller passes class_id=num_classes
        with some probability to teach the model the unconditional path).
        """
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.num_classes = int(num_classes)
        # Time embedding MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )
        if self.num_classes > 0:
            self.class_emb = nn.Embedding(self.num_classes + 1, t_emb_dim)
            # Init small so class signal grows during training instead of
            # overwhelming the (already trained) time signal.
            nn.init.normal_(self.class_emb.weight, std=0.02)
        else:
            self.class_emb = None
        # Input stem
        self.stem = nn.Conv1d(in_channels, channels[0], 7, padding=3)

        # Encoder
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_pool = nn.ModuleList()
        prev_c = channels[0]
        for i, c in enumerate(channels):
            self.down_res.append(ResBlock1D(prev_c, c, t_emb_dim, groups))
            self.down_res.append(ResBlock1D(c, c, t_emb_dim, groups))
            # Attention only at the deepest level
            self.down_attn.append(
                SelfAttention1D(c, groups=groups) if i == len(channels) - 1
                else nn.Identity()
            )
            # Downsample (stride 2 conv) except for the last level
            if i < len(channels) - 1:
                self.down_pool.append(nn.Conv1d(c, c, 4, stride=2, padding=1))
            else:
                self.down_pool.append(nn.Identity())
            prev_c = c

        # Bottleneck
        self.mid_res1 = ResBlock1D(prev_c, prev_c, t_emb_dim, groups)
        self.mid_attn = SelfAttention1D(prev_c, groups=groups)
        self.mid_res2 = ResBlock1D(prev_c, prev_c, t_emb_dim, groups)

        # Decoder
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.up_pool = nn.ModuleList()
        rev_channels = list(reversed(channels))
        prev_c = rev_channels[0]
        for i, c in enumerate(rev_channels):
            # Concat with skip → 2c
            self.up_res.append(ResBlock1D(prev_c + c, c, t_emb_dim, groups))
            self.up_res.append(ResBlock1D(c, c, t_emb_dim, groups))
            self.up_attn.append(
                SelfAttention1D(c, groups=groups) if i == 0
                else nn.Identity()
            )
            if i < len(rev_channels) - 1:
                # Upsample (conv-T stride 2)
                self.up_pool.append(nn.ConvTranspose1d(c, rev_channels[i + 1],
                                                       4, stride=2, padding=1))
                prev_c = rev_channels[i + 1]
            else:
                self.up_pool.append(nn.Identity())

        # Output head
        self.out_norm = nn.GroupNorm(groups, channels[0])
        self.out_conv = nn.Conv1d(channels[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                  class_id: torch.Tensor | None = None) -> torch.Tensor:
        # Pad time dim to multiple of 8 (3 stride-2 downsamples → factor 8).
        # Allows arbitrary input length T, including 2500 (10 s @ 250 Hz).
        T_in = x.shape[-1]
        T_pad = ((T_in + 7) // 8) * 8
        if T_pad > T_in:
            x = F.pad(x, (0, T_pad - T_in))

        t_emb = _timestep_embedding(t, self.t_emb_dim)
        t_emb = self.t_mlp(t_emb)

        if self.class_emb is not None:
            if class_id is None:
                # No class given → use the null-class slot (unconditional).
                class_id = torch.full(
                    (x.shape[0],), self.num_classes,
                    dtype=torch.long, device=x.device,
                )
            t_emb = t_emb + self.class_emb(class_id.long())

        h = self.stem(x)
        skips = []
        # Encoder
        for i in range(len(self.down_pool)):
            h = self.down_res[i * 2](h, t_emb)
            h = self.down_res[i * 2 + 1](h, t_emb)
            h = self.down_attn[i](h)
            skips.append(h)
            h = self.down_pool[i](h)

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Decoder
        for i in range(len(self.up_pool)):
            skip = skips[-(i + 1)]
            # Match time dim — conv stem + 4 downsamples may leave a 1-sample
            # mismatch on odd-length inputs. Crop/pad skip to match.
            if h.shape[-1] != skip.shape[-1]:
                if h.shape[-1] < skip.shape[-1]:
                    skip = skip[..., : h.shape[-1]]
                else:
                    h = h[..., : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = self.up_res[i * 2](h, t_emb)
            h = self.up_res[i * 2 + 1](h, t_emb)
            h = self.up_attn[i](h)
            h = self.up_pool[i](h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        if h.shape[-1] != T_in:
            h = h[..., :T_in]
        return h
