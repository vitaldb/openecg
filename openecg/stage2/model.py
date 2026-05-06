"""Stage 2 FrameClassifier: Conv + Transformer + Linear -> per-frame 4-class logits."""

import math

import torch
from torch import nn


class FrameClassifier(nn.Module):
    """Input: signal [B, 2500] @ 250Hz, lead_id [B] in {0..11}.
    Output: logits [B, 500, 4] (per-frame supercategory).
    """

    def __init__(
        self,
        n_leads=12,
        d_model=64,
        n_heads=4,
        n_layers=4,
        ff=256,
        n_classes=4,
        dropout=0.1,
        use_lead_emb=True,
    ):
        super().__init__()
        self.model_config = {
            "n_leads": n_leads,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "ff": ff,
            "n_classes": n_classes,
            "dropout": dropout,
            "use_lead_emb": use_lead_emb,
        }
        self.use_lead_emb = use_lead_emb
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=5, padding=7)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2)
        if use_lead_emb:
            self.lead_emb = nn.Embedding(n_leads, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, lead_id):
        h = torch.nn.functional.gelu(self.conv1(x.unsqueeze(1)))
        h = torch.nn.functional.gelu(self.conv2(h))
        h = h.transpose(1, 2)
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)


class FrameClassifierViT(nn.Module):
    """ViT-style: non-overlapping patch + Linear projection + positional encoding
    + Transformer + Linear head.

    Input:  signal [B, 2500] @ 250Hz, lead_id [B] in {0..11}.
    Output: logits [B, n_patches, n_classes] (per-frame supercategory).

    Options:
      pos_type: 'sinusoidal' (fixed), 'learnable' (nn.Embedding), or 'none'
      use_lead_emb: add per-lead embedding broadcast across all patches
      conv_stem: pre-patch Conv1d block to extract local features before
                 the linear patch embedding. Conv stem produces a richer
                 input than raw signal samples for the linear projection.
                 conv_stem=True applies: Conv1d(1->16, k=7, p=3) + GELU
                 + Conv1d(16->32, k=5, p=2) + GELU, length-preserving.
    """

    def __init__(
        self,
        patch_size=5,
        n_leads=12,
        d_model=64,
        n_heads=4,
        n_layers=4,
        ff=256,
        n_classes=4,
        dropout=0.1,
        use_lead_emb=True,
        pos_type="sinusoidal",
        conv_stem=False,
        max_seq_len=512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_lead_emb = use_lead_emb
        self.pos_type = pos_type
        self.conv_stem = conv_stem
        self.model_config = {
            "patch_size": patch_size,
            "n_leads": n_leads,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "ff": ff,
            "n_classes": n_classes,
            "dropout": dropout,
            "use_lead_emb": use_lead_emb,
            "pos_type": pos_type,
            "conv_stem": conv_stem,
            "arch": "vit",
        }
        if conv_stem:
            self.stem_conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
            self.stem_conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            patch_in = 32 * patch_size
        else:
            patch_in = patch_size
        self.patch_embed = nn.Linear(patch_in, d_model)
        if use_lead_emb:
            self.lead_emb = nn.Embedding(n_leads, d_model)

        if pos_type == "sinusoidal":
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                                  * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_enc", pe.unsqueeze(0))
        elif pos_type == "learnable":
            self.pos_enc = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.pos_enc, std=0.02)
        elif pos_type == "none":
            self.pos_enc = None
        else:
            raise ValueError(f"unknown pos_type: {pos_type}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, lead_id):
        B, N = x.shape  # N=2500
        assert N % self.patch_size == 0, f"signal length {N} not divisible by patch {self.patch_size}"
        n_patches = N // self.patch_size  # 500
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(x.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))  # [B, 32, N]
            # Reshape into patches: [B, n_patches, 32 * patch_size]
            h = h.transpose(1, 2)  # [B, N, 32]
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = x.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)  # [B, n_patches, d_model]
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)


class FrameClassifierViTReg(FrameClassifierViT):
    """ViT backbone with parallel classification + boundary-regression heads.

    Forward returns (cls_logits[B, N_patches, n_classes],
                     reg_offsets[B, N_patches, n_reg]).
    n_reg defaults to 6: signed sample-offset to nearest GT boundary of each of
    {p_on, p_off, qrs_on, qrs_off, t_on, t_off}.
    """

    def __init__(self, n_reg=6, **kwargs):
        super().__init__(**kwargs)
        self.n_reg = int(n_reg)
        self.reg_head = nn.Linear(self.head.in_features, self.n_reg)
        self.model_config = dict(self.model_config)
        self.model_config["n_reg"] = self.n_reg
        self.model_config["arch"] = "vit_reg"

    def forward(self, x, lead_id):
        B, N = x.shape
        n_patches = N // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(x.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = x.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        cls_logits = self.head(h)
        reg_offsets = self.reg_head(h)
        return cls_logits, reg_offsets


class FrameClassifierViTRegAux(FrameClassifierViTReg):
    """v13 Phase 1: ViT backbone with an auxiliary 4-class head tapped at
    an intermediate transformer layer.

    Splits the transformer into a lower stack (default 4 layers) and an upper
    stack (remaining layers). Adds an aux classification head that supervises
    the lower stack directly, encouraging the early layers to learn QRS-aware
    features explicitly — the clinical workflow ("first identify QRS, then
    locate P/T relative to it") expressed as an inductive bias.

    The aux logits are NOT concatenated into the upper stack's input — that
    is Phase 2. This is the lightest-touch variant.

    Forward returns
        cls_logits  [B, n_patches, n_classes]   (final 4-class output)
        reg_offsets [B, n_patches, n_reg]       (boundary regression)
        aux_logits  [B, n_patches, n_classes]   (intermediate 4-class output)
    """

    def __init__(self, aux_layer_split: int = 4, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_lower = int(aux_layer_split)
        n_upper = n_total - n_lower
        if not (0 < n_lower < n_total):
            raise ValueError(
                f"aux_layer_split={aux_layer_split} must be in (0, {n_total})"
            )

        def _make_stack(n: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=n)

        self.lower_transformer = _make_stack(n_lower)
        self.upper_transformer = _make_stack(n_upper)
        del self.transformer
        self.aux_head = nn.Linear(d_model, self.model_config["n_classes"])
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_aux"
        self.model_config["aux_layer_split"] = n_lower

    def forward(self, x, lead_id):
        B, N = x.shape
        n_patches = N // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(x.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = x.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h_lower = self.lower_transformer(h)
        aux_logits = self.aux_head(h_lower)
        h_upper = self.upper_transformer(h_lower)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_logits
