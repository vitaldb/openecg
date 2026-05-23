"""ECG-FM adapter - minimal PyTorch port of the wav2vec 2.0 backbone used by
wanglab/ecg-fm (mimic_iv_ecg_physionet_pretrained.pt) so we can load the
released weights without depending on fairseq_signals.

Architecture is reverse-engineered from the released state_dict:
    feature_extractor: 4x [Conv1d(stride=2, kernel=2, ch=256)]
                       (block 0 only: GroupNorm(256) + GELU after conv;
                        blocks 1..3: GELU only)
    layer_norm:        LayerNorm(256) at the end of the conv stack
    post_extract_proj: Linear(256 -> 768)
    conv_pos:          weight-normed Conv1d(768, 768, k=128, groups=16) + GELU
    encoder:           12 transformer blocks, post-LN style
                       embed_dim=768, heads=16, ffn=3072
    encoder.layer_norm: final LayerNorm(768)

Total temporal downsample 2^4 = 16x at 500 Hz -> 31.25 Hz token rate
(32 ms per token), comfortably below the QRS 40 ms tolerance.

Pipeline:
    sig [B, 2500] @ 250 Hz, single lead
      -> resample to 500 Hz length 5000 (matches paper's 10 s @ 500 Hz)
      -> replicate the lead across 12 channels (matches in_d=12)
      -> per-signal mean-subtract (paper used z-score; mean-only here keeps
         the conv stem's pretrained scale, similar to wav2vec2 audio adapters)
      -> wav2vec 2.0 forward -> [B, T, 768] with T ~ 312
      -> linear interp to [B, 500, 768]

Design notes:
    * The pretraining was 12-lead with random_leads_masking; replicating one
      lead 12 times is a reasonable single-lead inference strategy that keeps
      the conv input shape consistent with what the kernels saw.
    * We skip the quantizer / project_q / final_proj heads (they are
      pretraining-only auxiliary targets and never used downstream).
    * No masking at inference time (mask_prob=0 effectively).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
ECGFM_DEFAULT_CKPT = REPO_ROOT / "data" / "checkpoints" / "external" / "ecgfm_pretrained.pt"


def _make_conv_block(in_ch: int, out_ch: int, kernel: int, stride: int,
                     use_group_norm: bool) -> nn.Sequential:
    """Block layout matches fairseq wav2vec2 ConvFeatureExtractionModel:
        index 0: Conv1d (no bias)
        index 1: Dropout(0.0) (no params; placeholder)
        index 2: GroupNorm (only when use_group_norm)
        last:    GELU
    """
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, bias=False)
    if use_group_norm:
        return nn.Sequential(
            conv,
            nn.Dropout(0.0),
            nn.GroupNorm(num_groups=out_ch, num_channels=out_ch, affine=True),
            nn.GELU(),
        )
    return nn.Sequential(conv, nn.Dropout(0.0), nn.GELU())


class _ConvFeatureExtractor(nn.Module):
    """4-block conv stem matching wav2vec 2.0 'group_norm' mode (norm only on
    the first block). Stride 2 / kernel 2 each."""

    def __init__(self, in_d: int = 12, hidden: int = 256, n_blocks: int = 4):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(_make_conv_block(
                in_ch=in_d if i == 0 else hidden,
                out_ch=hidden,
                kernel=2, stride=2,
                use_group_norm=(i == 0),
            ))
        self.conv_layers = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_layers:
            x = block(x)
        return x


class _ConvPositionalEmbedding(nn.Module):
    """Group-conv positional encoding from wav2vec 2.0.

    Layout matches fairseq's SamePad convention:
        pos_conv = Sequential(weight_normed Conv1d, SamePad(drop last if even),
                              GELU)
    """

    def __init__(self, dim: int = 768, kernel: int = 128, groups: int = 16):
        super().__init__()
        conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2,
                         groups=groups)
        conv = nn.utils.weight_norm(conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(
            conv,
            _RemoveLastTokenIfEven(kernel),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        return x + x_conv


class _RemoveLastTokenIfEven(nn.Module):
    """When kernel is even, Conv1d with padding kernel//2 emits one extra
    token vs the input length. Drop the last frame to keep T unchanged."""

    def __init__(self, kernel: int):
        super().__init__()
        self.drop = (kernel % 2 == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop:
            return x[..., :-1]
        return x


class _TransformerBlock(nn.Module):
    """Pre-LN transformer block matching fairseq wav2vec 2.0 with
    layer_norm_first=True (the variant ECG-FM was trained with — confirmed
    empirically: post-LN produces rank-collapsed outputs with these weights,
    pre-LN does not, and the presence of a separate `encoder.layer_norm` at
    the end of the stack is itself a pre-LN signature).
    """

    def __init__(self, dim: int = 768, heads: int = 16, ffn: int = 3072,
                 dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                 batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n = self.self_attn_layer_norm(x)
        x_attn, _ = self.self_attn(x_n, x_n, x_n, need_weights=False)
        x = residual + self.dropout(x_attn)
        residual = x
        x_n = self.final_layer_norm(x)
        x_ff = self.fc2(self.dropout(self.act(self.fc1(x_n))))
        x = residual + self.dropout(x_ff)
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 16, ffn: int = 3072,
                 n_layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerBlock(dim, heads, ffn) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x


class _Wav2Vec2ECG(nn.Module):
    def __init__(self, in_d: int = 12, conv_dim: int = 256,
                 model_dim: int = 768, heads: int = 16, ffn: int = 3072,
                 n_layers: int = 12):
        super().__init__()
        self.feature_extractor = _ConvFeatureExtractor(in_d=in_d, hidden=conv_dim)
        self.layer_norm = nn.LayerNorm(conv_dim)
        self.post_extract_proj = nn.Linear(conv_dim, model_dim)
        self.conv_pos = _ConvPositionalEmbedding(dim=model_dim, kernel=128, groups=16)
        self.encoder = _TransformerEncoder(model_dim, heads, ffn, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        feats = feats.transpose(1, 2)
        feats = self.layer_norm(feats)
        feats = self.post_extract_proj(feats)
        feats = self.conv_pos(feats)
        return self.encoder(feats)


def _remap_state_dict(sd: dict) -> dict:
    """Map wav2vec2 fairseq-style keys to our module hierarchy.

    Most names already match; this hook lets us drop / rename only the
    pretraining-specific keys (quantizer / project_q / final_proj / mask_emb)
    and any keys that do not exist on _Wav2Vec2ECG.
    """
    drop_prefixes = ("quantizer.", "project_q.", "final_proj.")
    drop_exact = {"mask_emb"}
    out = {}
    for k, v in sd.items():
        if k in drop_exact:
            continue
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        out[k] = v
    return out


class ECGFMAdapter(nn.Module):
    """Loads wanglab/ecg-fm pretrained weights and exposes [B, 500, 768]
    per-frame features for the Stage 2 head.
    """

    HIDDEN_DIM = 768

    def __init__(self, weights_path: str | None = None,
                 device: str = "cpu",
                 native_fs: int = 250,
                 target_fs: int = 500,
                 target_samples: int = 5000):
        super().__init__()
        self.backbone = _Wav2Vec2ECG()
        if weights_path is None:
            weights_path = ECGFM_DEFAULT_CKPT
        weights_path = Path(weights_path)
        if weights_path.exists():
            blob = torch.load(weights_path, map_location="cpu", weights_only=False)
            sd = blob["model"] if "model" in blob else blob.get("state_dict", blob)
            sd = _remap_state_dict(sd)
            self._fuse_qkv_into_in_proj(sd)
            sd = self._strip_qkv_originals(sd)
            log = self.backbone.load_state_dict(sd, strict=False)
            critical_missing = [k for k in log.missing_keys if k]
            if critical_missing:
                raise RuntimeError(
                    f"Missing ECG-FM keys: {critical_missing[:8]}")
            if log.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected ECG-FM keys: {log.unexpected_keys[:8]}")
        self.hidden_dim = self.HIDDEN_DIM
        self.native_fs = int(native_fs)
        self.target_fs = int(target_fs)
        self.target_samples = int(target_samples)

    def _fuse_qkv_into_in_proj(self, sd: dict) -> None:
        """nn.MultiheadAttention uses fused in_proj_{weight,bias}; the
        checkpoint has separate q_proj / k_proj / v_proj. Stack into the
        fused tensors and add them to sd directly."""
        n_layers = len(self.backbone.encoder.layers)
        for i in range(n_layers):
            qw = sd[f"encoder.layers.{i}.self_attn.q_proj.weight"]
            kw = sd[f"encoder.layers.{i}.self_attn.k_proj.weight"]
            vw = sd[f"encoder.layers.{i}.self_attn.v_proj.weight"]
            qb = sd[f"encoder.layers.{i}.self_attn.q_proj.bias"]
            kb = sd[f"encoder.layers.{i}.self_attn.k_proj.bias"]
            vb = sd[f"encoder.layers.{i}.self_attn.v_proj.bias"]
            sd[f"encoder.layers.{i}.self_attn.in_proj_weight"] = torch.cat([qw, kw, vw], dim=0)
            sd[f"encoder.layers.{i}.self_attn.in_proj_bias"] = torch.cat([qb, kb, vb], dim=0)

    def _strip_qkv_originals(self, sd: dict) -> dict:
        """Remove the per-projection q/k/v keys after fusing them."""
        return {k: v for k, v in sd.items()
                if not (".self_attn.q_proj." in k
                        or ".self_attn.k_proj." in k
                        or ".self_attn.v_proj." in k)}

    def _resample(self, sig: torch.Tensor) -> torch.Tensor:
        x = sig.unsqueeze(1)
        x = F.interpolate(x, size=self.target_samples, mode="linear",
                          align_corners=False)
        return x.squeeze(1)

    def _zscore(self, sig: torch.Tensor) -> torch.Tensor:
        mean = sig.mean(dim=-1, keepdim=True)
        std = sig.std(dim=-1, keepdim=True) + 1e-8
        return (sig - mean) / std

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor) -> torch.Tensor:
        del lead_id
        if sig.dim() != 2:
            raise ValueError(f"Expected sig [B, N], got {tuple(sig.shape)}")
        x = self._resample(sig)
        x = self._zscore(x)
        x = x.unsqueeze(1).expand(-1, 12, -1).contiguous()
        feats = self.backbone(x)
        feats = feats.transpose(1, 2)
        feats = F.interpolate(feats, size=500, mode="linear", align_corners=False)
        return feats.transpose(1, 2).contiguous()
