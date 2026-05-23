"""ECG-FM backbone + v16 head: a hybrid that puts the v16/v15 architectural
priors (boundary regression head, QRS-binary aux head with concat-to-upper)
on top of the wav2vec 2.0 ECG-FM features.

Rationale (from the postmortem comparison, 2026-05-07):
    v12_ecgfm_ft (90 M SSL backbone, no priors) ≈ v12_reg
    v16_qrs_binary (1 M in-house + boundary reg + qrs_binary aux + paced synth)
        beats ECG-FM-FT by 0.014 / 0.013 / 0.045 (LUDB / ISP / QTDB).

The bet: ECG-FM gives stronger frame-level features (val score 0.887 vs
0.885 for v16) but lacks task-specific machinery for ms-precise boundary
localization. Bolting v16's head onto ECG-FM features should keep the
SSL representation gain *and* recover the boundary-regression / paced-aware
gains. Architecture:

    sig [B, 2500] @ 250 Hz  --(ECGFMAdapter, frozen for LP)-->  h_lower [B, 500, 768]
    aux_logits = aux_head(h_lower)                              [B, 500, 2]
    aux_probs  = softmax(aux_logits)
    h_upper_in = aux_to_upper_proj(cat(h_lower, aux_probs))     [B, 500, 768]
    h_upper    = upper_transformer(h_upper_in)                  [B, 500, 768]
    cls_logits = cls_head(h_upper)                              [B, 500, 4]
    reg_off    = reg_head(h_upper)                              [B, 500, 6]
    return (cls_logits, reg_off, aux_logits)

The aux head supervises h_lower (the ECG-FM output, before the upper
transformer sees it), keeping the v16 inductive bias of "commit to QRS
first, then locate P/T relative to it" — except now the lower stack is
the SSL-pretrained ECG-FM, not a randomly initialised conv+ViT.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from openecg.stage2.ssl.ecgfm import ECGFMAdapter, ECGFM_DEFAULT_CKPT


class ECGFMV16Head(nn.Module):
    """Hybrid model: ECG-FM (768-d) + v16-style upper head."""

    AUX_TARGETS = ("qrs_binary",)

    def __init__(self, weights_path: str | Path | None = None,
                 d_model: int = 768,
                 upper_layers: int = 4,
                 upper_heads: int = 8,
                 upper_ff: int = 3072,
                 dropout: float = 0.1,
                 n_classes: int = 4,
                 n_reg: int = 6,
                 aux_target: str = "qrs_binary"):
        super().__init__()
        if aux_target not in self.AUX_TARGETS:
            raise ValueError(
                f"aux_target must be in {self.AUX_TARGETS}, got {aux_target!r}"
            )
        self.aux_target = aux_target
        self.aux_n_classes = 2
        self.n_classes = n_classes
        self.n_reg = n_reg

        self.backbone = ECGFMAdapter(weights_path=weights_path)
        if self.backbone.hidden_dim != d_model:
            raise ValueError(
                f"d_model={d_model} must match ECGFMAdapter.hidden_dim="
                f"{self.backbone.hidden_dim}"
            )

        self.aux_head = nn.Linear(d_model, self.aux_n_classes)
        self.aux_to_upper_proj = nn.Linear(d_model + self.aux_n_classes, d_model)

        upper_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=upper_heads, dim_feedforward=upper_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.upper_transformer = nn.TransformerEncoder(upper_layer,
                                                        num_layers=upper_layers)

        self.head = nn.Linear(d_model, n_classes)
        self.reg_head = nn.Linear(d_model, n_reg)

        self.model_config = {
            "arch": "ecgfm_v16head",
            "d_model": d_model,
            "n_classes": n_classes,
            "n_reg": n_reg,
            "aux_target": aux_target,
            "aux_n_classes": self.aux_n_classes,
            "upper_layers": upper_layers,
            "upper_heads": upper_heads,
            "upper_ff": upper_ff,
        }

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor):
        h_lower = self.backbone(sig, lead_id)
        aux_logits = self.aux_head(h_lower)
        aux_probs = torch.softmax(aux_logits, dim=-1)
        h_upper_in = self.aux_to_upper_proj(torch.cat([h_lower, aux_probs], dim=-1))
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_logits
