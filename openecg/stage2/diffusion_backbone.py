"""Use the UNet1DDDPM encoder (Lydus-pretrained) as an ECG foundation
backbone for segmentation. Drops the DDPM decoder; replaces with our
v45g-style segmentation head (cls 4-class + reg 6-d + dual aux at
mid_split).

Two FT modes:
  * ``freeze_encoder=True``: only the segmentation head trains; the
    encoder produces fixed features. Cheap, tests pure feature quality.
  * ``freeze_encoder=False`` (default): full FT with smaller lr on the
    encoder than on the head.

Input: [B, 2, 2500] (sig + qrs_box) — same as v45g. The qrs_box channel
is mean-pooled per patch and concatenated as a prior into the upper
stack (kept from v45g for parity). The DDPM was trained on 1-ch sig
only, so the qrs_box conditioning is added INSIDE the segmentation
head, not the encoder itself — preserves encoder-FT semantics.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from openecg.stage2.diffusion import (
    UNet1DDDPM, ResBlock1D, SelfAttention1D, _timestep_embedding,
)


class DiffusionEncoder(nn.Module):
    """Wraps a trained UNet1DDDPM and exposes its encoder (stem + down
    ResBlocks + bottleneck) as a feature extractor. Time embedding is
    passed a constant ``t=0`` since segmentation isn't a denoising task.

    Output: [B, channels[-1], T/8] where ``T`` is the (padded) input
    length. For T=2504 → output T'=313 frames at channel 256.
    """

    def __init__(self, ddpm: UNet1DDDPM):
        super().__init__()
        self.ddpm = ddpm
        self.channels = tuple(int(c.shape[0])
                              for c in [self.ddpm.stem.weight,
                                         *(b.conv1.weight
                                           for b in self.ddpm.down_res[::2])])

    @property
    def out_channels(self) -> int:
        return self.ddpm.down_res[-1].conv2.out_channels

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """sig: [B, 1, T] → features [B, C, T/8]. No time conditioning
        (passes t=0 vector). class_id ignored if the DDPM was conditional."""
        T_in = sig.shape[-1]
        T_pad = ((T_in + 7) // 8) * 8
        if T_pad > T_in:
            sig = F.pad(sig, (0, T_pad - T_in))

        # Build a zero time embedding so the DDPM's time-gated ResBlocks
        # operate in their "no-noise" regime.
        B = sig.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=sig.device)
        t_emb = _timestep_embedding(t, self.ddpm.t_emb_dim)
        t_emb = self.ddpm.t_mlp(t_emb)
        if self.ddpm.class_emb is not None:
            # Use the null-class slot — segmentation should not depend on
            # class labels for input-conditioned features.
            null = torch.full((B,), self.ddpm.num_classes,
                              dtype=torch.long, device=sig.device)
            t_emb = t_emb + self.ddpm.class_emb(null)

        h = self.ddpm.stem(sig)
        for i in range(len(self.ddpm.down_pool)):
            h = self.ddpm.down_res[i * 2](h, t_emb)
            h = self.ddpm.down_res[i * 2 + 1](h, t_emb)
            h = self.ddpm.down_attn[i](h)
            h = self.ddpm.down_pool[i](h)
        h = self.ddpm.mid_res1(h, t_emb)
        h = self.ddpm.mid_attn(h)
        h = self.ddpm.mid_res2(h, t_emb)
        return h


class FrameClassifierDiffusionFM(nn.Module):
    """ECG foundation-model backbone (DDPM encoder) + v45g-style head.

    Pipeline:
      1. sig channel → DiffusionEncoder → [B, 256, 313]
      2. Upsample 313 → 500 frames (interp) — matches the openecg
         500-frame label resolution
      3. qrs_box channel → max-pool per 5-sample patch → [B, 500]
      4. concat (encoder_feat, qrs_box) → projection → upper transformer
         (2 layers) + cls/reg/aux heads
    """

    aux_target = "dual_binary"

    def __init__(
        self,
        ddpm_ckpt_state: dict,
        ddpm_kwargs: Optional[dict] = None,
        d_model_head: int = 128,
        n_upper_layers: int = 2,
        n_classes: int = 4,
        n_reg: int = 6,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        ddpm_kwargs = ddpm_kwargs or {}
        ddpm = UNet1DDDPM(**ddpm_kwargs)
        ddpm.load_state_dict(ddpm_ckpt_state)
        if freeze_encoder:
            for p in ddpm.parameters():
                p.requires_grad = False
        self.encoder = DiffusionEncoder(ddpm)

        enc_c = self.encoder.out_channels  # 256

        # Project encoder feat (256) → d_model_head
        self.feat_proj = nn.Linear(enc_c, d_model_head)

        # qrs_box prior — concat per patch (500 frames after upsample)
        # We concatenate a 1-dim qrs box per frame to the projected feat:
        # → d_model_head + 1, then linear back to d_model_head.
        self.qrs_inject = nn.Linear(d_model_head + 1, d_model_head)

        # Dual aux heads at the boundary between encoder feat and upper
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_head, nhead=4, dim_feedforward=4 * d_model_head,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.upper = nn.TransformerEncoder(encoder_layer, num_layers=n_upper_layers)

        self.aux_qrs_head = nn.Linear(d_model_head, 2)
        self.aux_p_head = nn.Linear(d_model_head, 2)
        # priors inject — qrs_box(1) + aux_qrs softmax(2) + aux_p softmax(2) = 5
        self.priors_proj = nn.Linear(d_model_head + 5, d_model_head)

        self.cls_head = nn.Linear(d_model_head, n_classes)
        self.reg_head = nn.Linear(d_model_head, n_reg)

        # ``aux_target`` must be set on the model so train.py routes the
        # dual-binary loss correctly.
        self.model_config = {
            "arch": "diffusion_fm_head",
            "d_model": d_model_head,
            "n_upper": n_upper_layers,
            "n_classes": n_classes,
            "n_reg": n_reg,
            "freeze_encoder": freeze_encoder,
        }
        # patch_size attribute expected by some downstream eval helpers
        self.patch_size = 5

    def forward(self, x: torch.Tensor, lead_id: torch.Tensor):
        # x: [B, 2, T]; lead_id unused (encoder is lead-agnostic).
        sig = x[:, 0:1]
        qrs_pp = x[:, 1]
        B, T = sig.shape[0], sig.shape[-1]
        feat = self.encoder(sig)              # [B, C, T/8] e.g. [B,256,313]
        # Upsample feat to 500 frames (target patch resolution: T=2500 → 500)
        feat = F.interpolate(feat, size=500, mode="linear", align_corners=False)
        feat = feat.transpose(1, 2)           # [B, 500, 256]
        h = self.feat_proj(feat)              # [B, 500, d_head]

        # qrs_box: 2500 → 500 via per-patch max
        qrs_per_patch = qrs_pp.view(B, 500, 5).amax(dim=2).unsqueeze(-1)
        h_lower = self.qrs_inject(torch.cat([h, qrs_per_patch], dim=-1))

        aux_qrs_logits = self.aux_qrs_head(h_lower)
        aux_p_logits = self.aux_p_head(h_lower)
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs = torch.softmax(aux_p_logits, dim=-1)

        h_upper_in = self.priors_proj(
            torch.cat([h_lower, qrs_per_patch, aux_qrs_probs, aux_p_probs],
                      dim=-1)
        )
        h_upper = self.upper(h_upper_in)
        cls_logits = self.cls_head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_qrs_logits, aux_p_logits


__all__ = ["DiffusionEncoder", "FrameClassifierDiffusionFM"]
