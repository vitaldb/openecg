"""Variants of v40c (FrameClassifierViTRegHybridAuxQRSPace) for arch ablation:

  * :class:`FrameClassifierMambaUpper` — replaces the transformer upper
    stack with Mamba2 selective-scan blocks. Linear-time sequence model
    with locality bias. Tests "is state-space modeling a better fit for
    ECG segmentation than self-attention?"
  * :class:`FrameClassifierSparseAttnUpper` — keeps the transformer
    layout but constrains self-attention to a local sliding window
    (default 128 patches ≈ 2.5 s @ 50 Hz frame rate). Tests "is local
    attention sufficient for ECG segmentation?"

Both inherit from FrameClassifierViTRegHybridAuxQRSPace, replace the
`upper_transformer` module, and override forward to apply the new upper
stack while preserving the lower CNN + aux + rule-prior + concat path.
"""
from __future__ import annotations

import torch
from torch import nn

from openecg.stage2.model import FrameClassifierViTRegHybridAuxQRSPace


class _LocalWindowEncoder(nn.Module):
    """TransformerEncoder that enforces a sliding-window attention mask.

    Each patch attends only to ±window_size//2 neighbours. Implemented by
    building a causal-ish band mask once and reusing it on every forward.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer,
                  num_layers: int, window_size: int):
        super().__init__()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.window_size = int(window_size)
        self._cached_mask: torch.Tensor | None = None
        self._cached_n: int = -1

    def _make_mask(self, n: int, device, dtype):
        if self._cached_mask is not None and self._cached_n == n:
            return self._cached_mask.to(device=device, dtype=dtype)
        idx = torch.arange(n)
        diff = (idx[:, None] - idx[None, :]).abs()
        # True where attention is BLOCKED (outside window).
        mask = diff > (self.window_size // 2)
        # Convert to additive float mask: -inf where blocked, 0 where allowed.
        attn_mask = torch.zeros(n, n, dtype=dtype)
        attn_mask.masked_fill_(mask, float("-inf"))
        self._cached_mask = attn_mask
        self._cached_n = n
        return attn_mask.to(device=device)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n = src.shape[1]
        mask = self._make_mask(n, src.device, src.dtype)
        return self.encoder(src, mask=mask)


class FrameClassifierSparseAttnUpper(FrameClassifierViTRegHybridAuxQRSPace):
    """v40c with sliding-window self-attention in the upper stack.

    Window size defaults to 128 patches (2.56 s @ 50 fps frame rate, or
    roughly 2-3 cardiac cycles at HR 60-100). Lower CNN stack and
    aux/rule-prior inject path are unchanged from v40c.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7,
                  attn_window: int = 128, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_upper = n_total - int(mid_split)

        upper_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.upper_transformer = _LocalWindowEncoder(
            upper_layer, num_layers=n_upper, window_size=attn_window,
        )
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_hybrid_aux_qrs_pace_sparseattn"
        self.model_config["attn_window"] = int(attn_window)


class FrameClassifierMambaUpper(FrameClassifierViTRegHybridAuxQRSPace):
    """v40c with Mamba2 upper stack (linear-time selective scan).

    Requires `mamba_ssm` package. Mamba2 block: d_model + d_state=16 +
    d_conv=4 (paper defaults). Inputs/outputs shape preserved as the
    transformer it replaces.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7,
                  d_state: int = 16, d_conv: int = 4, expand: int = 2,
                  **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        # Use Mamba (v1) instead of Mamba2: v2 has stricter alignment
        # constraints (channel-last stride must be multiple of 8) that
        # break with d_model=128. v1 is simpler and works at this scale.
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except Exception as exc:
            raise RuntimeError(
                f"mamba_ssm not installed or kernel build failed: {exc}. "
                f"Install with: pip install mamba-ssm causal-conv1d"
            ) from exc

        d_model = self.model_config["d_model"]
        n_total = self.model_config["n_layers"]
        n_upper = n_total - int(mid_split)

        # Stack of Mamba (v1) blocks with residual + LN — drop-in for transformer.
        blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand)
            for _ in range(n_upper)
        ])
        norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_upper)])
        self.upper_transformer = _MambaStack(blocks, norms)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_hybrid_aux_qrs_pace_mamba"
        self.model_config["mamba_version"] = "v1"
        self.model_config["mamba_d_state"] = int(d_state)
        self.model_config["mamba_d_conv"] = int(d_conv)
        self.model_config["mamba_expand"] = int(expand)


class _MambaStack(nn.Module):
    """Pre-norm residual stack of Mamba2 blocks: x -> x + block(LN(x))."""

    def __init__(self, blocks: nn.ModuleList, norms: nn.ModuleList):
        super().__init__()
        self.blocks = blocks
        self.norms = norms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))
        return x


class FrameClassifierMambaDualAux(FrameClassifierMambaUpper):
    """Mamba upper + DUAL aux heads (QRS-binary AND P-binary).

    Both aux supervisions feed back into the upper stack via concat at mid_split,
    so the upper Mamba sees explicit QRS-presence AND P-presence priors per
    frame. Tests whether explicit P-supervision helps when added on top of
    the QRS-aware backbone (single-aux ablations showed P-only replacement
    of QRS aux did not help).

    Forward returns (cls, reg, aux_qrs, aux_p) — a 4-tuple. Use
    `fit_reg_dual_aux` trainer.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7,
                  d_state: int = 16, d_conv: int = 4, expand: int = 2,
                  **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel,
                          d_state=d_state, d_conv=d_conv, expand=expand,
                          **kwargs)
        d_model = self.model_config["d_model"]
        # Existing self.aux_head is QRS-binary (from parent). Add P-binary head.
        self.aux_head_p = nn.Linear(d_model, 2)
        # Mid concat now: d + qrs_pp(1) + pace_pp(1) + aux_qrs_softmax(2)
        # + aux_p_softmax(2) = d + 6. Override the parent's d+4 projection.
        self.priors_inject_proj = nn.Linear(d_model + 6, d_model)
        self.aux_target = "dual_binary"
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_hybrid_aux_qrs_pace_mamba_dual"
        self.model_config["aux_target"] = "dual_binary"

    def forward(self, x, lead_id):
        sig, qrs_pp, pace_pp = self._split_signal_qrs_pace(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        # Two aux heads, two softmaxes.
        aux_qrs_logits = self.aux_head(h_lower)               # [B, P, 2]
        aux_p_logits   = self.aux_head_p(h_lower)             # [B, P, 2]
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs   = torch.softmax(aux_p_logits,   dim=-1)

        qrs_per_patch  = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        pace_per_patch = pace_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = torch.stack([qrs_per_patch, pace_per_patch], dim=-1)  # [B, P, 2]

        h_upper_in = self.priors_inject_proj(
            torch.cat([h_lower, rules, aux_qrs_probs, aux_p_probs], dim=-1),
        )
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_qrs_logits, aux_p_logits


class FrameClassifierTransformerDualAux2Ch(FrameClassifierViTRegHybridAuxQRSPace):
    """v47 — pure-PyTorch transformer dual-aux on 2-channel input (sig, qrs_box).

    No mamba_ssm dependency. Same hybrid CNN-lower + Transformer-upper
    backbone as v40c, plus the dual aux heads (QRS-binary + P-binary) and
    a 2-channel input (sig + qrs_box) — the qrs_box channel replaces the
    1-pt qrs + pace + wide trio from earlier variants with a single box
    channel whose width naturally encodes wide-vs-narrow QRS.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        d_model = self.model_config["d_model"]
        # Add P-binary aux head on top of the inherited QRS-binary aux head.
        self.aux_head_p = nn.Linear(d_model, 2)
        # Concat width: d + qrs(1) + aux_qrs_softmax(2) + aux_p_softmax(2) = d + 5
        self.priors_inject_proj = nn.Linear(d_model + 5, d_model)
        self.aux_target = "dual_binary"
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_transformer_dual_aux_2ch"
        self.model_config["aux_target"] = "dual_binary"
        self.model_config["n_input_channels"] = 2

    # v54d+: int16 rank input support. When the dataset yields int16 (e.g.
    # _RankNormalizedDataset(output_int=True)), ch0 contains rank linearly
    # quantized to [-32767, +32767] and ch1 contains qrs_box 0/1 (int16).
    # Cast back at the forward boundary so the conv/transformer stack stays
    # float — memory savings come from halved input-batch bytes through
    # DataLoader collation, not from any int compute.
    _RANK_INT_SCALE = 32767.0

    @staticmethod
    def _split_signal_qrs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(
                f"2ch transformer dual-aux expects [B, 2, T], got {tuple(x.shape)}"
            )
        sig, qrs_pp = x[:, 0], x[:, 1]
        if not sig.is_floating_point():
            sig = sig.float() / FrameClassifierTransformerDualAux2Ch._RANK_INT_SCALE
            qrs_pp = qrs_pp.float()
        return sig, qrs_pp

    def forward(self, x, lead_id):
        sig, qrs_pp = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        aux_qrs_logits = self.aux_head(h_lower)
        aux_p_logits   = self.aux_head_p(h_lower)
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs   = torch.softmax(aux_p_logits,   dim=-1)

        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = qrs_per_patch.unsqueeze(-1)  # [B, P, 1]

        h_upper_in = self.priors_inject_proj(
            torch.cat([h_lower, rules, aux_qrs_probs, aux_p_probs], dim=-1),
        )
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_qrs_logits, aux_p_logits


class FrameClassifierTransformerDualAuxNoReg2Ch(FrameClassifierTransformerDualAux2Ch):
    """v50d — same backbone as v47 (transformer hybrid dual aux 2-ch) but
    the `reg_head` is removed. Forward returns ``(cls, None, aux_qrs, aux_p)``
    (None in the reg slot so the 4-tuple shape is preserved and existing
    `train_one_epoch_reg_aux` / `score_all_1ch` paths can branch on it).

    Boundary precision is reduced to the frame grid (20 ms = 5 samples @
    250 Hz). With Martinez tolerance ±40-100 ms this is still well within
    the F1 tolerance band, but sub-frame regression accuracy (typically
    +0.03 mean F1, [[v12-postmortem]]) is lost.

    The class exists primarily as a v45m1 ablation control to measure
    reg_head's contribution. Not intended as a deploy default.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        # Drop the reg head from the inherited stack.
        del self.reg_head
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_dual_aux_2ch_noreg"
        self.model_config["use_reg"] = False
        self.model_config["n_reg"] = 0

    def forward(self, x, lead_id):
        sig, qrs_pp = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        aux_qrs_logits = self.aux_head(h_lower)
        aux_p_logits   = self.aux_head_p(h_lower)
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs   = torch.softmax(aux_p_logits,   dim=-1)

        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = qrs_per_patch.unsqueeze(-1)  # [B, P, 1]

        h_upper_in = self.priors_inject_proj(
            torch.cat([h_lower, rules, aux_qrs_probs, aux_p_probs], dim=-1),
        )
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        # No reg head — keep 4-tuple shape with None in reg slot so callers
        # that expect (cls, reg, aux_qrs, aux_p) can branch on `reg is None`.
        return cls_logits, None, aux_qrs_logits, aux_p_logits


class FrameClassifierTransformerNoAux2Ch(FrameClassifierTransformerDualAux2Ch):
    """v52 — same backbone as v47 (CNN-lower + Transformer-upper, 2-ch input
    sig+qrs_box) but the **mid-stack aux heads + priors_inject_proj** are
    removed. Forward returns ``(cls, reg, None, None)`` (Nones in aux slots
    so the 4-tuple shape is preserved for `train_one_epoch_reg_aux`).

    The qrs_box channel is still consumed (preserves the 2-ch input contract
    with `compose_sig_qrs_box_2ch`) but discarded in forward — there is no
    longer a mid-concat injection point. The model is therefore: patch_embed
    → pos + lead_emb → CNN-lower (mid_split layers) → upper Transformer
    (n_layers - mid_split layers) → cls_head + reg_head. No aux supervision.

    Purpose: ablation of the dual aux head (`aux_qrs_head` + `aux_p_head`)
    contribution. Compare to FrameClassifierTransformerDualAux2Ch at the
    same depth (L12/L8) to measure the +alpha_aux × (CE_qrs + CE_p) effect.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        # Drop the dual aux heads + the mid-concat projection from the
        # inherited dual-aux 2-ch stack.
        del self.aux_head
        del self.aux_head_p
        del self.priors_inject_proj
        self.aux_target = "none"
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_noaux_2ch"
        self.model_config["aux_target"] = "none"
        self.model_config["use_aux"] = False

    def forward(self, x, lead_id):
        sig, _qrs_pp_unused = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        # CNN lower (residual + LayerNorm) — same as parent.
        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        # No aux heads, no mid concat — upper Transformer takes h_lower directly.
        h_upper = self.upper_transformer(h_lower)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        # 4-tuple shape kept; aux slots None to mark "no aux supervision".
        return cls_logits, reg_offsets, None, None


class FrameClassifierTransformerNoAux1Ch(FrameClassifierTransformerNoAux2Ch):
    """v55a — true 1-channel variant of v54i (NoAux2Ch). Input is the raw
    signal only; the qrs_box channel from the 2-ch contract is removed
    entirely.

    Weights and forward computation are **bit-identical** to
    :class:`FrameClassifierTransformerNoAux2Ch` because the parent's
    forward already discards ``x[:, 1]``. The only change is the input
    shape contract (and the metadata):

      * ``FrameClassifierTransformerNoAux2Ch`` expects ``(B, 2, T)``;
        forward then does ``sig = x[:, 0]`` and ignores ``x[:, 1]``.
      * This subclass accepts ``(B, T)`` directly (or ``(B, 1, T)``);
        ``_split_signal_qrs`` returns ``(sig, zeros_like(sig))`` so the
        inherited forward can run unchanged.

    Practical effects:
      * One fewer channel-worth of bytes through the dataloader.
      * Cleaner deploy ONNX signature (single input, no dummy zeros).
      * v54i checkpoint weights load directly via
        :func:`copy_v54i_to_v55a` — no fine-tuning needed.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_noaux_1ch"
        self.model_config["n_input_channels"] = 1

    @staticmethod
    def _split_signal_qrs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Accepted layouts:
        #   * ``(B, T)``        — preferred 1-channel deploy input.
        #   * ``(B, 1, T)``     — data pipeline that emits singleton chan dim.
        #   * ``(B, 2, T)``     — backward-compat with the v54i 2-ch eval
        #     pipeline (compose_sig_qrs_box_2ch). Channel 1 is discarded.
        if x.dim() == 3:
            if x.shape[1] == 1:
                sig = x[:, 0]
            elif x.shape[1] == 2:
                # 2-ch input from compose_sig_qrs_box_2ch: drop ch1.
                sig = x[:, 0]
            else:
                raise ValueError(
                    f"1ch expects (B, 1, T) or (B, 2, T) when 3-d, "
                    f"got {tuple(x.shape)}")
        elif x.dim() == 2:
            sig = x
        else:
            raise ValueError(f"1ch expects 2-d or 3-d input, got {tuple(x.shape)}")
        qrs_pp = torch.zeros_like(sig)
        return sig, qrs_pp


class FrameClassifierTransformerLayered1Ch(FrameClassifierTransformerNoAux1Ch):
    """v57 — single-pass multi-head delineator emitting all three layers
    of the :mod:`openecg.layered` codec from one forward.

    Heads (all per-patch over ``h_upper`` — output shape ``(B, n_patches,
    n_classes)`` at the 50 Hz frame grid):

      * ``head``        -> N_FRAME (5)  other/P/QRS/T/paced_QRS    [Layer 0]
      * ``reg_head``    -> 6           boundary on/off offsets     [reg]
      * ``beat_head``   -> N_BEAT (6)  none/sinus/vpc/paced/fusion/unknown
                                                                   [Layer 1]
      * ``rhythm_head`` -> N_RHYTHM (6) sinus/avb/paced/afib/bbb/ventricular
                                                                   [Layer 2]

    Forward returns ``(cls, reg, beat, rhythm)`` — slots 2/3 are repurposed
    relative to the inherited ``(cls, reg, None, None)`` tuple from
    :class:`FrameClassifierTransformerNoAux1Ch`. The tuple *shape* is kept
    so legacy ``train_one_epoch_reg`` paths still consume slot 0/1
    unchanged; new training code should pull beat/rhythm from slots 2/3.

    Init-from-v56d: ``head``, ``reg_head``, backbone weights load via
    ``load_state_dict(..., strict=False)``. ``beat_head`` and
    ``rhythm_head`` are zero-initialised so the untrained model's argmax
    output is ``BEAT_NONE`` / ``RHYTHM_SINUS`` everywhere — encode()
    callers see "no anomaly" defaults until the new heads are trained.

    Rhythm head is per-patch (not pooled), so sub-window rhythm
    transitions (e.g. NSR → VT episode mid-window) are representable
    end-to-end. The window-constant fallback in
    :mod:`openecg.layered` is preserved for legacy single-label
    classifiers.
    """

    def __init__(self, *,
                 beat_n_classes: int = 6,
                 rhythm_n_classes: int = 6,
                 mid_split: int = 4,
                 lower_kernel: int = 7,
                 **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        d_model = self.head.in_features
        self.beat_head   = nn.Linear(d_model, beat_n_classes)
        self.rhythm_head = nn.Linear(d_model, rhythm_n_classes)
        # Zero-init the new heads so untrained checkpoints (e.g. when
        # loaded from v56d via strict=False) emit a safe default
        # (argmax = 0 = BEAT_NONE / RHYTHM_SINUS) instead of garbage.
        nn.init.zeros_(self.beat_head.weight)
        nn.init.zeros_(self.beat_head.bias)
        nn.init.zeros_(self.rhythm_head.weight)
        nn.init.zeros_(self.rhythm_head.bias)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_layered_1ch"
        self.model_config["beat_n_classes"] = int(beat_n_classes)
        self.model_config["rhythm_n_classes"] = int(rhythm_n_classes)

    def forward(self, x, lead_id):
        sig, _qrs_pp_unused = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        h_upper = self.upper_transformer(h_lower)
        cls_logits    = self.head(h_upper)
        reg_offsets   = self.reg_head(h_upper)
        beat_logits   = self.beat_head(h_upper)
        rhythm_logits = self.rhythm_head(h_upper)
        return cls_logits, reg_offsets, beat_logits, rhythm_logits


class FrameClassifierTransformerSampleRes1Ch(FrameClassifierTransformerNoAux1Ch):
    """v57a — frame head emits per-sample logits at the input sample
    rate (250 Hz, 4 ms granularity) instead of per-patch (50 Hz, 20 ms).
    ``reg_head`` is dropped: sample-resolution argmax already gives
    4-ms boundary precision, so sub-frame regression is redundant.

    Purpose: test option A from the layered-codec design discussion —
    does a per-sample softmax match or beat v56c's per-patch + reg-head
    on the Martínez boundary F1 / median-timing-error benchmarks?

    Forward returns ``(cls_logits, None, None, None)`` where
    ``cls_logits`` has shape ``(B, T, n_classes)`` with T equal to the
    full input sample count.  Slot 1 is None (no reg); slots 2/3 are
    None (no beat/rhythm heads in this variant; the full multi-head
    v57 is :class:`FrameClassifierTransformerLayered1Ch`).

    **Bit-equivalent v56c init.** The two-conv refine block is
    zero-initialised on its output side, so at load time the refine
    pathway contributes zero and ``cls_logits`` equals
    ``head_sample(nearest_upsample(h_upper))``.  If ``head_sample`` is
    seeded from the v56c ``head`` weights via
    :func:`copy_per_patch_head_to_sample`, the initial output is
    pointwise identical to v56c's per-patch logits replicated by
    ``patch_size``.  Training then refines.
    """

    def __init__(self, *, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        d_model = self.head.in_features
        n_classes = self.head.out_features
        # Drop the inherited reg head — boundary precision comes from
        # sample-resolution softmax in this variant.
        if hasattr(self, "reg_head"):
            del self.reg_head
        # Residual refinement at sample resolution. Second conv is
        # zero-initialised so the refine output is zero at load time.
        self.refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        # Per-sample classification head (replaces the per-patch
        # ``head``). Seed by copying ``head`` weights — see
        # ``copy_per_patch_head_to_sample``.
        self.head_sample = nn.Linear(d_model, n_classes)
        with torch.no_grad():
            self.head_sample.weight.copy_(self.head.weight)
            self.head_sample.bias.copy_(self.head.bias)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_sample_res_1ch"
        self.model_config["use_reg"] = False
        self.model_config["n_reg"] = 0
        self.model_config["sample_res_frame"] = True

    def copy_per_patch_head_to_sample(self) -> None:
        """Re-seed ``head_sample`` from the per-patch ``head``.  Call
        this after :func:`load_model_from_ckpt` when initialising from a
        v56c-style ckpt whose state dict only contains ``head.*``.
        """
        with torch.no_grad():
            self.head_sample.weight.copy_(self.head.weight)
            self.head_sample.bias.copy_(self.head.bias)

    def forward(self, x, lead_id):
        sig, _qrs_pp_unused = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)
        h_upper = self.upper_transformer(h_lower)        # (B, n_patches, d)

        # Sample-resolution path: nearest upsample + zero-init refine
        # residual + per-sample head.
        h_T = h_upper.transpose(1, 2)                    # (B, d, n_patches)
        h_up = torch.nn.functional.interpolate(
            h_T, scale_factor=self.patch_size, mode="nearest")
        h_up = h_up + self.refine(h_up)                  # (B, d, T)
        h_up = h_up.transpose(1, 2)                      # (B, T, d)
        cls_logits = self.head_sample(h_up)              # (B, T, n_classes)
        return cls_logits, None, None, None


class FrameClassifierTransformerSampleResConvTok1Ch(
        FrameClassifierTransformerSampleRes1Ch):
    """v57d-D2 — sample-resolution head with a **convolutional tokenizer**
    that replaces the Linear ``patch_embed`` entirely.

    Ablation question: is the learned Linear patch_embed
    (reshape-to-patches → Linear) necessary, or does a strided Conv1d
    tokenizer (which keeps local translation structure) work as well or
    better for a single-lead model?

    Tokenizer: ``Conv1d(1, d_model, kernel_size=2*patch_size,
    stride=patch_size, padding=patch_size//2)`` maps the raw 500 Hz
    signal ``(B, T)`` directly to ``(B, n_patches, d_model)`` at the
    50 Hz grid — no reshape, no Linear patch_embed.  ``patch_embed`` is
    deleted; everything downstream (lower CNN, upper transformer,
    sample-resolution upsample + refine + head_sample) is unchanged, so
    the v56c backbone still transfers (only the tokenizer is new/random).

    Single-lead: ``use_lead_emb`` defaults to False here.
    """

    def __init__(self, *, mid_split: int = 4, lower_kernel: int = 7,
                 use_lead_emb: bool = False, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel,
                         use_lead_emb=use_lead_emb, **kwargs)
        d_model = self.head.in_features
        ps = self.patch_size
        # Strided conv tokenizer: out_len = (T + 2*pad - k)/stride + 1.
        # k=2*ps, stride=ps, pad=ps//2 -> out_len = T/ps (== n_patches).
        self.conv_tok = nn.Conv1d(1, d_model, kernel_size=2 * ps,
                                  stride=ps, padding=ps // 2)
        if hasattr(self, "patch_embed"):
            del self.patch_embed
        if hasattr(self, "stem_conv1"):
            del self.stem_conv1
            del self.stem_conv2
            self.conv_stem = False
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_sample_res_convtok_1ch"
        self.model_config["tokenizer"] = "conv"

    def forward(self, x, lead_id):
        sig, _ = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        # Conv tokenizer -> (B, d_model, n_patches) -> (B, n_patches, d_model)
        h = self.conv_tok(sig.unsqueeze(1))
        if h.shape[-1] != n_patches:                       # guard rounding
            h = h[..., :n_patches]
        h = h.transpose(1, 2)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)
        h_upper = self.upper_transformer(h_lower)

        h_T = h_upper.transpose(1, 2)
        h_up = torch.nn.functional.interpolate(
            h_T, scale_factor=self.patch_size, mode="nearest")
        h_up = h_up + self.refine(h_up)
        h_up = h_up.transpose(1, 2)
        cls_logits = self.head_sample(h_up)
        return cls_logits, None, None, None


class FrameClassifierTransformerSampleResConvTokMH1Ch(
        FrameClassifierTransformerSampleResConvTok1Ch):
    """v59b — the layered-codec **multi-head** on the E0 backbone:
    conv-tokenizer + sample-resolution (500 Hz output) + single-lead +
    NO reg head, with three **per-sample** heads sharing the encoder:

      * ``head_sample``   -> frame  (B, T, n_classes)   [other/P/QRS/T]
      * ``beat_sample``   -> beat   (B, T, 6)           [none/sinus/vpc/paced/fusion/unknown]
      * ``rhythm_sample`` -> rhythm (B, T, 6)           [sinus/avb/paced/afib/bbb/ventricular]

    All three read the same sample-resolution feature ``h_up`` (the E0
    upsample+refine output), so the multi-head adds only three thin
    Linear heads over the shared 500 Hz representation. beat/rhythm are
    zero-init (safe defaults) and train from synth; frame + backbone
    transfer from E0/v57c. Forward returns ``(frame, beat, rhythm)`` —
    note 3-tuple (no reg slot), consistent with the no-reg decision.
    """

    def __init__(self, *, beat_n_classes: int = 6, rhythm_n_classes: int = 6,
                 mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        d = self.head_sample.in_features
        self.beat_sample = nn.Linear(d, beat_n_classes)
        self.rhythm_sample = nn.Linear(d, rhythm_n_classes)
        for h in (self.beat_sample, self.rhythm_sample):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_sample_res_convtok_mh_1ch"
        self.model_config["beat_n_classes"] = int(beat_n_classes)
        self.model_config["rhythm_n_classes"] = int(rhythm_n_classes)

    def forward(self, x, lead_id):
        sig, _ = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        h = self.conv_tok(sig.unsqueeze(1))
        if h.shape[-1] != n_patches:
            h = h[..., :n_patches]
        h = h.transpose(1, 2)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)
        h_upper = self.upper_transformer(h_lower)
        h_T = h_upper.transpose(1, 2)
        h_up = torch.nn.functional.interpolate(
            h_T, scale_factor=self.patch_size, mode="nearest")
        h_up = h_up + self.refine(h_up)
        h_up = h_up.transpose(1, 2)                          # (B, T, d)
        return (self.head_sample(h_up),                      # frame
                self.beat_sample(h_up),                      # beat
                self.rhythm_sample(h_up))                    # rhythm


class FrameClassifierTransformerNoAux2ChBoxIn(FrameClassifierTransformerNoAux2Ch):
    """v54o — noaux 2-ch where the qrs_box channel is **fed into patch
    embedding** alongside the signal (vs. the parent which discards it).

    Ablation goal: isolate the contribution of the qrs_box feature when
    used *only* as an input prior, without the mid-stack aux supervision
    or priors_inject_proj of the dual-aux variant.

    Compared to:
      * ``FrameClassifierTransformerNoAux2Ch`` (parent): same backbone,
        but parent only uses ``sig`` in ``patch_embed`` and discards
        ``qrs_box`` entirely. This subclass concatenates both channels
        patch-wise so ``patch_embed`` sees 2 × patch_size features.
      * ``FrameClassifierTransformerDualAux2Ch`` (sibling, v54a): uses
        ``qrs_box`` via aux heads + mid-concat injection. This subclass
        skips both — only the input pathway carries the prior.

    Input contract: ``(B, 2, T)`` same as parent, but BOTH channels are
    consumed. Output: ``(cls, reg, None, None)`` 4-tuple.

    Not compatible with conv_stem=True (the stem currently expects 1-ch).
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        if self.conv_stem:
            raise ValueError(
                "FrameClassifierTransformerNoAux2ChBoxIn does not support "
                "conv_stem=True (1-ch only). Set conv_stem=False.")
        d_model = self.model_config["d_model"]
        # Replace the 1-ch patch_embed (Linear(patch_size, d_model)) with
        # a 2-ch version that takes both [sig_patches | box_patches].
        self.patch_embed = nn.Linear(self.patch_size * 2, d_model)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_noaux_2ch_boxin"

    def forward(self, x, lead_id):
        sig, qrs_pp = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        # Stack [sig|box] patch-wise → (B, n_patches, 2 * patch_size)
        sig_p = sig.view(B, n_patches, self.patch_size)
        box_p = qrs_pp.view(B, n_patches, self.patch_size)
        patches = torch.cat([sig_p, box_p], dim=-1)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        # CNN lower (residual + LayerNorm) — same as parent.
        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        # Upper transformer; no aux, no mid-inject.
        h_upper = self.upper_transformer(h_lower)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, None, None


class FrameClassifierCnnOnly2Ch(FrameClassifierTransformerNoAux2Ch):
    """v54l — Pure-CNN encoder (NO transformer). All ``n_layers`` are
    patch-level Conv1d residual blocks (same kernel + LayerNorm shape as
    the existing CNN-lower stack). Used as the no-attention baseline to
    measure how much of v54c/i/h's accuracy actually comes from the
    transformer upper vs the CNN lower mixer.

    Same input contract as ``FrameClassifierTransformerNoAux2Ch``
    (2-ch [B, 2, T] with qrs_box discarded). Forward returns
    ``(cls, reg, None, None)`` to stay compatible with
    ``train_one_epoch_reg_aux``.
    """

    def __init__(self, lower_kernel: int = 7, **kwargs):
        n_total = int(kwargs.get("n_layers", 8))
        # Hack: parent constructor needs a mid_split <= n_layers to build
        # the upper transformer stack. Pass 1 (placeholder), then tear
        # down upper_transformer + lower_convs/norms and rebuild lower
        # with ``n_total`` Conv1d blocks below.
        kwargs["mid_split"] = 1
        super().__init__(lower_kernel=lower_kernel, **kwargs)

        d_model = self.model_config["d_model"]
        del self.lower_convs
        del self.lower_norms
        if hasattr(self, "upper_transformer"):
            del self.upper_transformer

        self.lower_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=lower_kernel,
                      padding=lower_kernel // 2)
            for _ in range(n_total)
        ])
        self.lower_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_total)
        ])

        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_cnn_only_2ch"
        self.model_config["mid_split"] = n_total  # everything is "lower"

    def forward(self, x, lead_id):
        sig, _qrs_pp_unused = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        cls_logits = self.head(h_lower)
        reg_offsets = self.reg_head(h_lower)
        return cls_logits, reg_offsets, None, None


class FrameClassifierTransformerNoAux2ChRankEmb(FrameClassifierTransformerNoAux2Ch):
    """v54e — NoAux2Ch with discrete rank-embedding lookup as the input stem.

    Replaces the float-rank → ``patch_size``-wide Linear patch embedding
    with a learnable lookup table: rank-quantized int index →
    nn.Embedding(n_rank_bins, rank_emb_dim) → reshape to per-patch tokens
    of width ``patch_size × rank_emb_dim`` → Linear(..., d_model).

    Hypothesis: continuous rank-norm + linear projection treats every
    rank position uniformly. An embedding table lets each rank bin learn
    its own representation — useful if certain rank levels (R-peak max,
    baseline mid, etc.) carry distinct semantics worth specialized
    encoding. ECG R-peaks always land at rank ≈ +1 and baseline at rank
    ≈ 0, so per-bin embeddings could short-circuit "amplitude position"
    detection earlier in the stack.

    Input contract: the dataset must yield int (typically int16) for
    ch0 in the int16-scaled rank format ([-RANK_INT_SCALE, +RANK_INT_SCALE]
    produced by `_RankNormalizedDataset(output_int=True)`); ch1 (qrs_box)
    is also int16 0/1 but discarded by the NoAux variant. Internally the
    int range is binned to ``n_rank_bins`` levels for the lookup table.

    Param overhead vs the parent (patch_embed Linear(5, 128) ≈ 0.6K):
      * embedding: n_rank_bins × rank_emb_dim
      * patch_embed: patch_size × rank_emb_dim → d_model

    Default (n_rank_bins=1024, rank_emb_dim=16) at d_model=128 / patch_size=5:
    16K + 10K ≈ 26K extra params (vs ~0.99M base = +2.6%).
    """

    def __init__(
        self, *, n_rank_bins: int = 1024, rank_emb_dim: int = 16, **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rank_bins = int(n_rank_bins)
        self.rank_emb_dim = int(rank_emb_dim)
        d_model = self.model_config["d_model"]
        self.rank_emb = torch.nn.Embedding(self.n_rank_bins, self.rank_emb_dim)
        # Replace patch_embed: from Linear(patch_size, d_model) (parent's
        # conv_stem=False path) to Linear(patch_size * rank_emb_dim, d_model)
        # — each token concatenates ``patch_size`` rank-embedding vectors.
        self.patch_embed = torch.nn.Linear(
            self.patch_size * self.rank_emb_dim, d_model,
        )
        # Bookkeeping in the model config so checkpoints record the variant.
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_transformer_noaux_2ch_rankembed"
        self.model_config["n_rank_bins"] = self.n_rank_bins
        self.model_config["rank_emb_dim"] = self.rank_emb_dim

    @staticmethod
    def _split_signal_qrs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Same shape check as parent, but keep sig in its original (int)
        # dtype — the rank-embedding lookup needs an integer index. qrs_pp
        # is cast to float since the noaux branch never uses it but other
        # callers expect a float tensor.
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(
                f"2ch rank-embed expects [B, 2, T], got {tuple(x.shape)}"
            )
        sig, qrs_pp = x[:, 0], x[:, 1]
        if not qrs_pp.is_floating_point():
            qrs_pp = qrs_pp.float()
        return sig, qrs_pp

    def forward(self, x, lead_id):
        sig, _qrs_pp_unused = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size

        # Normalize ch0 to the int16 rank range first, then linearly bin into
        # [0, n_rank_bins-1]. Two upstream conventions are supported:
        #   * train path: dataset yields int16 in [-RANK_INT_SCALE, +RANK_INT_SCALE]
        #     (`_RankNormalizedDataset(output_int=True)`). Use the int values.
        #   * eval path: `score_all_1ch(..., input_norm="rank")` applies the
        #     float rank wrapper, yielding values in [-1, +1]. Re-scale to
        #     the same int range so the model sees a single index distribution.
        # Cast the scale to int explicitly because the parent's class attribute
        # is a float (for the non-embed `.float() / RANK_INT_SCALE` cast).
        scale = int(self._RANK_INT_SCALE)
        full_range = 2 * scale + 1                  # 65535
        if sig.is_floating_point():
            sig_int = (sig * scale).round().long()
        else:
            sig_int = sig.long()
        sig_idx = ((sig_int + scale) * self.n_rank_bins) // full_range
        sig_idx = sig_idx.clamp_(0, self.n_rank_bins - 1)

        # Lookup → patchify → project
        h = self.rank_emb(sig_idx)                                  # [B, T, e]
        patches = h.reshape(B, n_patches, self.patch_size * self.rank_emb_dim)
        h = self.patch_embed(patches)                               # [B, P, d]

        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        # CNN lower (residual + LayerNorm) — same as parent.
        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        h_upper = self.upper_transformer(h_lower)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, None, None


class FrameClassifierMambaDualAux2Ch(FrameClassifierMambaDualAux):
    """v46 — Mamba dual aux on 2-channel input (sig, qrs_box).

    Drops the pace prior dimension from FrameClassifierMambaDualAux. The
    mid-stack concat shrinks from d_model + 6 to d_model + 5 (QRS prior
    1-d + aux_qrs softmax 2-d + aux_p softmax 2-d). All other components
    (CNN lower stack, Mamba upper, dual aux heads) are inherited.

    Input layout: [B, 2, T] where x[:,0] = signal, x[:,1] = QRS box
    indicator (1 between detect_qrs(return_boundaries=True)'s on/off).
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7,
                  d_state: int = 16, d_conv: int = 4, expand: int = 2,
                  **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel,
                          d_state=d_state, d_conv=d_conv, expand=expand,
                          **kwargs)
        d_model = self.model_config["d_model"]
        # qrs(1) + aux_qrs_softmax(2) + aux_p_softmax(2) = +5
        self.priors_inject_proj = nn.Linear(d_model + 5, d_model)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "mamba_dual_aux_2ch"
        self.model_config["n_input_channels"] = 2

    @staticmethod
    def _split_signal_qrs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(
                f"2ch dual-aux expects [B, 2, T] input, got {tuple(x.shape)}"
            )
        return x[:, 0], x[:, 1]

    def forward(self, x, lead_id):
        sig, qrs_pp = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)

        aux_qrs_logits = self.aux_head(h_lower)
        aux_p_logits   = self.aux_head_p(h_lower)
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs   = torch.softmax(aux_p_logits,   dim=-1)

        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = qrs_per_patch.unsqueeze(-1)  # [B, P, 1]

        h_upper_in = self.priors_inject_proj(
            torch.cat([h_lower, rules, aux_qrs_probs, aux_p_probs], dim=-1),
        )
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_qrs_logits, aux_p_logits


class FrameClassifierFullTransformerDualAux2Ch(FrameClassifierTransformerDualAux2Ch):
    """v45i — Pure transformer (lower + upper both ``nn.TransformerEncoder``)
    dual aux on 2-ch (sig, qrs_box). Replaces the conv1d lower stack of v45g
    with another transformer encoder of the same depth as the conv lower
    (``mid_split`` layers). Everything else (patch embed, pos enc, lead emb,
    dual aux heads, priors_inject_proj, upper transformer) is inherited
    unchanged from v45g.

    Hypothesis: the conv1d lower stack provides a fixed local receptive
    field that may be the bottleneck for atypical ECG morphology (paced /
    BBB / extreme amplitude). A transformer lower gets self-attention from
    the patch embedding onward and can learn whatever local↔global pattern
    the task needs. Cost: more params per layer (attention vs conv).

    Runs natively on Windows (no mamba_ssm needed) — same portability
    advantage as v45g but spends more capacity in the lower stack.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(mid_split=mid_split, lower_kernel=lower_kernel, **kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config.get("n_heads", 4)
        ff = self.model_config.get("ff", 4 * d_model)
        dropout = self.model_config.get("dropout", 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.lower_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(mid_split),
        )
        # Free the inherited conv lower components (held in ParameterList
        # / ModuleList on the parent) so they don't waste optimizer state.
        del self.lower_convs
        del self.lower_norms
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_full_transformer_dual_aux_2ch"

    def forward(self, x, lead_id):
        sig, qrs_pp = self._split_signal_qrs(x)
        B, T = sig.shape
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        h_lower = self.lower_transformer(h)

        aux_qrs_logits = self.aux_head(h_lower)
        aux_p_logits   = self.aux_head_p(h_lower)
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs   = torch.softmax(aux_p_logits,   dim=-1)

        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = qrs_per_patch.unsqueeze(-1)

        h_upper_in = self.priors_inject_proj(
            torch.cat([h_lower, rules, aux_qrs_probs, aux_p_probs], dim=-1),
        )
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_qrs_logits, aux_p_logits


class FrameClassifierTriStageDualAux(nn.Module):
    """v41 (12L tri-stage transformer) + DUAL aux heads (QRS + P) at L8.

    Same as FrameClassifierViTRegTriStageInjectQRSPace except the aux at
    the 2/3-depth split has TWO 2-class heads (QRS-binary and P-binary)
    instead of one. Both softmaxes feed into the upper stack via concat:
    d + qrs_softmax(2) + p_softmax(2) = d+4.

    Tests whether the dual-aux trick (which gave QTDB +0.016 on the
    8L Mamba arch) compounds with v41's 12L depth (which gave LUDB
    +0.005 over v37b).

    Forward returns (cls, reg, aux_qrs, aux_p) — 4-tuple. Use the same
    `fit_reg_aux` trainer (which detects `aux_target='dual_binary'`).
    """

    def __init__(self, lower_split: int = 4, mid_split: int = 8, **kwargs):
        # Lazy import to avoid circular dependency.
        from openecg.stage2.model import FrameClassifierViTRegTriStageInjectQRSPace
        super().__init__()
        # Build the parent module and steal its modules — composition over
        # inheritance to avoid clashing with our 4-tuple return signature.
        self._parent = FrameClassifierViTRegTriStageInjectQRSPace(
            lower_split=lower_split, mid_split=mid_split, **kwargs,
        )
        d_model = self._parent.model_config["d_model"]

        # Add P-binary aux head alongside the parent's QRS-binary head.
        self.aux_head_p = nn.Linear(d_model, 2)
        # Override aux_inject_proj from d+2 (single softmax) → d+4 (dual).
        self._parent.aux_inject_proj = nn.Linear(d_model + 4, d_model)

        # Mark for trainer dispatch.
        self.aux_target = "dual_binary"

        # Mirror parent's model_config + arch tag.
        self.model_config = dict(self._parent.model_config)
        self.model_config["arch"] = "vit_reg_tri_stage_inject_qrs_pace_dual"
        self.model_config["aux_target"] = "dual_binary"

        # Re-route Module.__getattr__ to fall through to parent for
        # things like .patch_embed etc. Done via direct attribute set:
        # parameters() will recurse into both _parent and aux_head_p.
        self.patch_size = self._parent.patch_size
        self.use_lead_emb = self._parent.use_lead_emb
        self.conv_stem = self._parent.conv_stem
        self.head = self._parent.head
        self.reg_head = self._parent.reg_head

    def forward(self, x, lead_id):
        p = self._parent
        sig, qrs_pp, pace_pp = p._split_signal_qrs_pace(x)
        B, T = sig.shape
        n_patches = T // p.patch_size
        if p.conv_stem:
            h = torch.nn.functional.gelu(p.stem_conv1(sig.unsqueeze(1)))
            h = torch.nn.functional.gelu(p.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, p.patch_size * 32)
        else:
            patches = sig.view(B, n_patches, p.patch_size)
        h = p.patch_embed(patches)
        if p.pos_enc is not None:
            h = h + p.pos_enc[:, :n_patches]
        if p.use_lead_emb:
            h = h + p.lead_emb(lead_id).unsqueeze(1)

        # Stage 1: lower (raw signal only).
        h_lower = p.lower_transformer(h)

        # Rule prior injection at 1/3.
        qrs_per_patch  = qrs_pp.view(B, n_patches, p.patch_size).amax(dim=2)
        pace_per_patch = pace_pp.view(B, n_patches, p.patch_size).amax(dim=2)
        rules = torch.stack([qrs_per_patch, pace_per_patch], dim=-1)
        h_mid_in = p.rule_inject_proj(torch.cat([h_lower, rules], dim=-1))

        # Stage 2: mid (rule-aware).
        h_mid = p.mid_transformer(h_mid_in)

        # Dual aux at 2/3.
        aux_qrs_logits = p.aux_head(h_mid)              # [B, P, 2]
        aux_p_logits   = self.aux_head_p(h_mid)         # [B, P, 2]
        aux_qrs_probs = torch.softmax(aux_qrs_logits, dim=-1)
        aux_p_probs   = torch.softmax(aux_p_logits,   dim=-1)
        h_upper_in = p.aux_inject_proj(
            torch.cat([h_mid, aux_qrs_probs, aux_p_probs], dim=-1),
        )

        # Stage 3: upper.
        h_upper = p.upper_transformer(h_upper_in)
        cls_logits  = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_qrs_logits, aux_p_logits


class FrameClassifierMambaPure(nn.Module):
    """Pure 8L Mamba on 1-ch patched signal. No CNN lower, no aux, no rule
    prior — just patch embed → 8L Mamba → cls + reg heads.

    Ablation against FrameClassifierMambaUpper which keeps the v40c
    bells (3-ch input, CNN lower, aux concat, rule injection). Tests
    whether the v40c structural priors are *necessary* on top of Mamba
    or whether plain Mamba can match.

    Forward: (sig [B, T], lead_id [B]) → (cls [B, P, 4], reg [B, P, 6]).
    Returns just (cls, reg) — use fit_reg trainer.
    """

    def __init__(self, patch_size: int = 5, d_model: int = 128,
                  n_layers: int = 8, dropout: float = 0.1,
                  use_lead_emb: bool = False, pos_type: str = "learnable",
                  conv_stem: bool = False,
                  d_state: int = 16, d_conv: int = 4, expand: int = 2,
                  n_reg: int = 6, n_classes: int = 4,
                  # Compatibility with v40c kwargs (ignored):
                  n_heads: int | None = None, ff: int | None = None):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except Exception as exc:
            raise RuntimeError(f"mamba_ssm not available: {exc}") from exc

        self.patch_size = int(patch_size)
        self.use_lead_emb = bool(use_lead_emb)
        self.conv_stem = bool(conv_stem)

        self.patch_embed = nn.Linear(patch_size, d_model)
        # Positional embedding (matches v9 KWARGS pos_type='learnable').
        self.pos_enc = nn.Parameter(
            torch.zeros(1, 2500 // patch_size, d_model)
        )
        if pos_type == "learnable":
            nn.init.trunc_normal_(self.pos_enc, std=0.02)
        if use_lead_emb:
            self.lead_emb = nn.Embedding(12, d_model)

        blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.transformer = _MambaStack(blocks, norms)

        self.head = nn.Linear(d_model, n_classes)
        self.reg_head = nn.Linear(d_model, n_reg)

        self.model_config = dict(
            arch="mamba_pure",
            patch_size=patch_size, d_model=d_model, n_layers=n_layers,
            dropout=dropout, use_lead_emb=use_lead_emb,
            pos_type=pos_type, conv_stem=conv_stem,
            mamba_d_state=d_state, mamba_d_conv=d_conv, mamba_expand=expand,
            n_input_channels=1,
        )

    def forward(self, x: torch.Tensor, lead_id: torch.Tensor):
        # Accept either [B, T] (1-ch) or [B, 3, T] (3-ch — ignore rule chans).
        if x.dim() == 3:
            x = x[:, 0]
        B, T = x.shape
        n_patches = T // self.patch_size
        patches = x.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h), self.reg_head(h)
