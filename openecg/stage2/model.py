"""Stage 2 FrameClassifier: Conv + Transformer + Linear -> per-frame 4-class logits."""

import math
from pathlib import Path

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
        max_seq_len=None,
    ):
        super().__init__()
        # Auto-size pos_enc to accommodate 2500-frame inputs at any
        # patch_size (factor of 2500). Override only if explicitly set.
        if max_seq_len is None:
            max_seq_len = max(512, 2500 // patch_size)
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
    """v13 Phase 1 / v16: ViT backbone with an auxiliary head tapped at an
    intermediate transformer layer.

    Splits the transformer into a lower stack (default 4 layers) and an upper
    stack (remaining layers). The aux head supervises the lower stack
    directly — turning the clinical workflow ("first identify QRS, then
    locate P/T relative to it") into an architectural inductive bias.

    `aux_target` controls what the lower stack is forced to commit to:
      * ``"all"`` (default, v13/v15 behaviour) — full 4-class supervision.
        The aux head is essentially deep supervision; the lower stack ends
        up learning the same task as the main head, just earlier.
      * ``"qrs_binary"`` (v16) — 2-class (QRS vs rest). The lower stack must
        commit to "is this frame inside a QRS?" *without* having to also
        decide P / T. This is the strict reading of the README's claim
        that aux is "QRS-aware".
      * ``"p_binary"`` — 2-class (P vs rest). Symmetric counterpart for
        atrial-activity-first hierarchies.

    For v15-style concat (FrameClassifierViTRegAuxConcat), the aux output
    dim flows into the upper stack's input projection, so changing
    `aux_target` also changes the projection's input width.

    Forward returns
        cls_logits  [B, n_patches, n_classes]        (final 4-class output)
        reg_offsets [B, n_patches, n_reg]            (boundary regression)
        aux_logits  [B, n_patches, aux_n_classes]    (intermediate output)
    """

    AUX_TARGETS = ("all", "qrs_binary", "p_binary")

    def __init__(self, aux_layer_split: int = 4, aux_target: str = "all", **kwargs):
        super().__init__(**kwargs)
        if aux_target not in self.AUX_TARGETS:
            raise ValueError(
                f"aux_target must be one of {self.AUX_TARGETS}, got {aux_target!r}"
            )
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_main_classes = self.model_config["n_classes"]
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
        aux_n_classes = n_main_classes if aux_target == "all" else 2
        self.aux_target = aux_target
        self.aux_n_classes = aux_n_classes
        self.aux_head = nn.Linear(d_model, aux_n_classes)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_aux"
        self.model_config["aux_layer_split"] = n_lower
        self.model_config["aux_target"] = aux_target

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


class FrameClassifierViTRegAuxConcat(FrameClassifierViTRegAux):
    """v15 Phase 2: aux output is fed forward into the upper transformer
    input via a learned projection.

    The aux head's QRS-aware logits (after softmax) are concatenated with
    the lower-stack features and projected back to d_model before entering
    the upper stack. The upper stack's attention can therefore use the
    explicit QRS-confidence channel as input — the architectural form of
    the clinical "find paced/wide QRS near where a QRS already is" hint.

    Forward returns (cls_logits, reg_offsets, aux_logits) just like the
    parent so existing trainers / inference helpers stay compatible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        # aux_n_classes is set by parent based on aux_target; binary aux
        # heads project a 2-D channel into the upper stack instead of 4.
        self.aux_to_upper_proj = nn.Linear(d_model + self.aux_n_classes, d_model)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_aux_concat"

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
        aux_probs = torch.softmax(aux_logits, dim=-1)
        h_upper_in = self.aux_to_upper_proj(torch.cat([h_lower, aux_probs], dim=-1))
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_logits


class FrameClassifierViTRegMultiTask(FrameClassifierViTRegAuxConcat):
    """v17: v16 (QRS-binary aux + concat) augmented with window-level
    multi-task heads on top of the upper transformer features.

    Two backbone heads are always present (the divide-and-conquer pivots):
      * `rr_regular`  binary  — is the RR pattern regular?
      * `qrs_wide`    binary  — is the typical QRS ≥ 120 ms?

    Optional richer heads (off by default; enable with non-zero kwargs):
      * `rhythm` (n_rhythm_classes) — 3-class sinus / AF / other.
      * `avb`    (n_avb_classes)    — 4-class none / 1° / 2° / 3°.

    Each window-level head is a Linear over the time-mean of the upper
    transformer output (one prediction per 10 s window). Forward returns
    a 4-tuple (cls_logits, reg_offsets, aux_logits, window_logits) where
    `window_logits` is a dict keyed by task name; tasks with disabled
    heads are absent from the dict. Inference helpers that only read
    `out[:2]` (e.g. predict_frames_with_reg) keep working unchanged.
    """

    def __init__(self, n_rhythm_classes: int = 0, n_avb_classes: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        d = self.model_config["d_model"]
        self.head_rr_regular = nn.Linear(d, 2)
        self.head_qrs_wide = nn.Linear(d, 2)
        self.n_rhythm_classes = int(n_rhythm_classes)
        self.n_avb_classes = int(n_avb_classes)
        self.head_rhythm = (nn.Linear(d, self.n_rhythm_classes)
                            if self.n_rhythm_classes > 0 else None)
        self.head_avb = (nn.Linear(d, self.n_avb_classes)
                         if self.n_avb_classes > 0 else None)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_multitask"
        self.model_config["n_rhythm_classes"] = self.n_rhythm_classes
        self.model_config["n_avb_classes"] = self.n_avb_classes

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
        aux_probs = torch.softmax(aux_logits, dim=-1)
        h_upper_in = self.aux_to_upper_proj(torch.cat([h_lower, aux_probs], dim=-1))
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        h_window = h_upper.mean(dim=1)                     # [B, d_model]
        window_logits: dict[str, torch.Tensor] = {
            "rr_regular": self.head_rr_regular(h_window),  # [B, 2]
            "qrs_wide":   self.head_qrs_wide(h_window),    # [B, 2]
        }
        if self.head_rhythm is not None:
            window_logits["rhythm"] = self.head_rhythm(h_window)
        if self.head_avb is not None:
            window_logits["avb"] = self.head_avb(h_window)
        return cls_logits, reg_offsets, aux_logits, window_logits


class FrameClassifierViTRegMultiIn(FrameClassifierViTReg):
    """v22 input-channel-prior model.

    Identical to FrameClassifierViTReg except the ECG-sample input is
    extended from 1 channel to ``n_input_channels`` channels (signal +
    rule-based priors such as pacer slope and QRS-position indicator).
    The ViT patch embedding layer's input dimension is widened from
    ``patch_size`` to ``n_input_channels * patch_size`` (or the conv-stem
    first conv is widened from 1 to ``n_input_channels`` input channels).

    Forward expects ``x`` of shape [B, C, T] where C == n_input_channels.
    Returns the same (cls_logits, reg_offsets) tuple as the parent so
    downstream training and eval helpers stay compatible.
    """

    def __init__(self, n_input_channels: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_input_channels = int(n_input_channels)
        if self.n_input_channels < 1:
            raise ValueError("n_input_channels must be >= 1")
        if self.n_input_channels != 1:
            patch_size = self.model_config["patch_size"]
            d_model = self.model_config["d_model"]
            if self.conv_stem:
                old = self.stem_conv1
                new = nn.Conv1d(
                    in_channels=self.n_input_channels,
                    out_channels=old.out_channels,
                    kernel_size=old.kernel_size[0],
                    padding=old.padding[0],
                )
                self.stem_conv1 = new
            else:
                self.patch_embed = nn.Linear(
                    self.n_input_channels * patch_size, d_model,
                )
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_multiin"
        self.model_config["n_input_channels"] = self.n_input_channels

    def forward(self, x, lead_id):
        if x.dim() == 2:
            # Backward compat path: [B, T] treated as 1-channel input.
            if self.n_input_channels != 1:
                raise ValueError(
                    f"model has n_input_channels={self.n_input_channels} but received "
                    f"x with shape {tuple(x.shape)} (no channel dim)"
                )
            x = x.unsqueeze(1)                              # [B, 1, T]
        elif x.dim() != 3:
            raise ValueError(f"expected 2D or 3D input, got shape {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.n_input_channels:
            raise ValueError(
                f"got C={C} but model has n_input_channels={self.n_input_channels}"
            )
        n_patches = T // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(x))
            h = torch.nn.functional.gelu(self.stem_conv2(h))   # [B, 32, T]
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            # Each patch concatenates ``patch_size`` samples across all channels.
            x_p = x.reshape(B, C, n_patches, self.patch_size).transpose(1, 2)
            patches = x_p.reshape(B, n_patches, C * self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        cls_logits = self.head(h)
        reg_offsets = self.reg_head(h)
        return cls_logits, reg_offsets


class _QRSInjectionMixin:
    """Helpers shared by v24a / v24b classes that inject a rule-based per-patch
    QRS prior at a fixed network depth.

    Forward expects ``x`` of shape ``[B, 2, T]`` where channel 0 is the
    z-normed ECG signal and channel 1 is a (point-only) QRS indicator at the
    same sample rate. Inside forward, channel 1 is max-pooled to per-patch
    binary [B, n_patches] and then expanded to a 2-D one-hot-style tensor
    [(1-q), q] mirroring v16's softmax(QRS-binary) shape so the projection
    layer keeps the same input width (d_model + 2)."""

    @staticmethod
    def _split_signal_qrs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(
                f"QRS-injection model expects [B, 2, T] input, got {tuple(x.shape)}"
            )
        return x[:, 0], x[:, 1]

    @staticmethod
    def _qrs_per_patch_2d(qrs_pp: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Max-pool per-patch and expand to ``[B, n_patches, 2]`` tensor
        ``[(1-q), q]``.
        """
        B, T = qrs_pp.shape
        n_patches = T // patch_size
        per_patch = qrs_pp.view(B, n_patches, patch_size).amax(dim=2)   # [B, P]
        return torch.stack([1.0 - per_patch, per_patch], dim=-1)         # [B, P, 2]


class FrameClassifierViTRegPatchInjectQRS(FrameClassifierViTReg, _QRSInjectionMixin):
    """v24a: rule-based QRS prior concatenated *immediately after patch_embed*.

    Signal goes through ``patch_embed`` exactly as in v12_reg; the rule-based
    QRS channel (max-pooled to per-patch) is then concatenated as 2 extra dims
    and projected back to d_model. The full 8-layer transformer follows.

    This isolates "early-mid injection (after patch_embed but before transformer)"
    from "raw-input-channel injection" (v22). Returns the same (cls_logits,
    reg_offsets) tuple as v12_reg.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        self.qrs_inject_proj = nn.Linear(d_model + 2, d_model)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_patch_inject_qrs"

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
        h = self.patch_embed(patches)                                   # [B, P, d]
        qrs_2d = self._qrs_per_patch_2d(qrs_pp, self.patch_size)        # [B, P, 2]
        h = self.qrs_inject_proj(torch.cat([h, qrs_2d], dim=-1))         # [B, P, d]
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        cls_logits = self.head(h)
        reg_offsets = self.reg_head(h)
        return cls_logits, reg_offsets


class FrameClassifierViTRegMidInjectQRS(FrameClassifierViTReg, _QRSInjectionMixin):
    """v24b: rule-based QRS prior concatenated at the mid-stack split.

    Architecturally identical to v16's concat path (lower stack of
    ``mid_split`` layers, then concat 2-D QRS prior, project back to d_model,
    then upper stack of remaining layers) — but the prior is the rule-based
    detect_qrs output instead of a learned aux head's softmax. Tests the
    hypothesis that v16's mid-stack concat slot is the structural advantage,
    independent of whether the prior is learned or hand-crafted.

    Returns the same (cls_logits, reg_offsets) tuple as v12_reg.
    """

    def __init__(self, mid_split: int = 4, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_lower = int(mid_split)
        n_upper = n_total - n_lower
        if not (0 < n_lower < n_total):
            raise ValueError(
                f"mid_split={mid_split} must be in (0, {n_total})"
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
        self.qrs_inject_proj = nn.Linear(d_model + 2, d_model)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_mid_inject_qrs"
        self.model_config["mid_split"] = n_lower

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
        qrs_2d = self._qrs_per_patch_2d(qrs_pp, self.patch_size)
        h_upper_in = self.qrs_inject_proj(torch.cat([h_lower, qrs_2d], dim=-1))
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets


class FrameClassifierViTRegMultiInjectQRS(FrameClassifierViTReg, _QRSInjectionMixin):
    """v26: rule-based QRS prior injected at multiple transformer-layer boundaries.

    The single 8-layer transformer is split into ``len(inject_at) + 1`` stacks.
    A 1-D per-patch QRS indicator is concatenated (feature dim) to the running
    hidden state at each split and projected back to ``d_model`` via a per-split
    Linear. Concat dim is 1 (not 2 as in v24b) since the rule-based prior is
    a single binary indicator per patch — Linear(d+1) is mathematically
    equivalent to Linear(d+2) with [1-q, q] but uses fewer parameters.

    inject_at: tuple of ints in ``(0, n_layers)`` — concat the prior after the
    Nth layer (1-indexed). e.g. ``(2, 6)`` for an 8-layer model = inject after
    layer 2 and after layer 6. ``(1, 2, 3, 4, 5, 6, 7)`` = inject after every
    layer except the last (= every-layer injection).
    """

    def __init__(self, inject_at: tuple[int, ...] = (2, 6), **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        inject_at = tuple(sorted(set(int(i) for i in inject_at)))
        for i in inject_at:
            if not (0 < i < n_total):
                raise ValueError(
                    f"inject_at entry {i} must be in (0, {n_total}); got {inject_at}"
                )

        def _make_stack(n: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=n)

        boundaries = (0,) + inject_at + (n_total,)
        sizes = [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]
        self.stacks = nn.ModuleList([_make_stack(sz) for sz in sizes])
        # One projection per inject point (d_model + 1 → d_model).
        self.qrs_inject_projs = nn.ModuleList([
            nn.Linear(d_model + 1, d_model) for _ in inject_at
        ])
        del self.transformer
        self.inject_at = inject_at
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_multi_inject_qrs"
        self.model_config["inject_at"] = list(inject_at)

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

        # Per-patch 1-D QRS indicator (max-pool across patch_size samples).
        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)  # [B, P]
        qrs_1d = qrs_per_patch.unsqueeze(-1)                                     # [B, P, 1]

        # Walk stacks; inject between consecutive stacks.
        n_stacks = len(self.stacks)
        for i, stack in enumerate(self.stacks):
            h = stack(h)
            if i < n_stacks - 1:                                                # inject before next stack
                h = self.qrs_inject_projs[i](torch.cat([h, qrs_1d], dim=-1))

        cls_logits = self.head(h)
        reg_offsets = self.reg_head(h)
        return cls_logits, reg_offsets


class FrameClassifierViTRegMidInjectQRSPace(FrameClassifierViTReg, _QRSInjectionMixin):
    """v33: rule-based QRS *and* pacing-spike priors injected at mid-stack.

    Architecturally identical to v24b (`FrameClassifierViTRegMidInjectQRS`)
    except the input has a third channel — the pacing-spike 1-point indicator
    from ``openecg.detect_pacings`` — and both per-patch priors (QRS + pacing)
    are concatenated as 1-D each (2 extra dims total) before the
    Linear(d_model+2 → d_model) projection.

    Input shape: ``[B, 3, T]`` where channel 0 is z-normed signal,
    channel 1 is QRS 1-point indicator at target fs, channel 2 is pacing
    1-point indicator at target fs.
    """

    def __init__(self, mid_split: int = 4, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_lower = int(mid_split)
        n_upper = n_total - n_lower
        if not (0 < n_lower < n_total):
            raise ValueError(
                f"mid_split={mid_split} must be in (0, {n_total})"
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
        # Two 1-D priors concatenated — same projection size as v24b's [1-q, q]
        # 2-d concat (mathematically equivalent, see v24b vs v26 1-d analysis).
        self.priors_inject_proj = nn.Linear(d_model + 2, d_model)
        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_mid_inject_qrs_pace"
        self.model_config["mid_split"] = n_lower

    @staticmethod
    def _split_signal_qrs_pace(x: torch.Tensor) -> tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 3:
            raise ValueError(
                f"QRS+pacing-injection model expects [B, 3, T] input, "
                f"got {tuple(x.shape)}"
            )
        return x[:, 0], x[:, 1], x[:, 2]

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
        h_lower = self.lower_transformer(h)

        # Per-patch 1-D priors (max-pool across patch_size samples).
        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        pace_per_patch = pace_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        priors = torch.stack([qrs_per_patch, pace_per_patch], dim=-1)         # [B, P, 2]

        h_upper_in = self.priors_inject_proj(torch.cat([h_lower, priors], dim=-1))
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets


class FrameClassifierViTRegHybridAuxQRSPace(FrameClassifierViTReg, _QRSInjectionMixin):
    """v38: CNN-Transformer hybrid with GT-supervised QRS-binary aux head and
    rule-based QRS+pacing priors at mid-stack.

    Combines three architectural levers:
      * Hybrid backbone — lower stack is a 1D-conv tower (length-preserving),
        upper stack is a transformer. CNN gives strong local-pattern bias for
        P/QRS edges; transformer keeps the long-range rhythm context.
      * GT aux supervision — at the mid-stack split, an aux head is trained
        on QRS-binary frame labels (= v16 trick). Its softmax is concat'd to
        the upper-stack input.
      * 3-channel rule prior — signal + detect_qrs 1-pt + detect_pacings 1-pt
        per-patch indicators concat'd alongside the aux softmax (= v33 trick).

    The mid-stack concat width is therefore d_model + 4 (= q + p + 2-d aux
    softmax), projected back to d_model.

    Forward returns (cls_logits, reg_offsets, aux_logits) so the existing
    ``fit_reg_aux`` trainer applies without modification. Kwargs accept
    larger d_model / n_heads / ff than the v9 defaults.
    """

    def __init__(self, mid_split: int = 4, lower_kernel: int = 7, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_lower = int(mid_split)
        n_upper = n_total - n_lower
        if not (0 < n_lower < n_total):
            raise ValueError(
                f"mid_split={mid_split} must be in (0, {n_total})"
            )

        # 1D conv lower stack — length-preserving, on the patch sequence.
        # Each block: Conv1d(d, d, k, padding) + GELU + LayerNorm.
        pad = lower_kernel // 2
        conv_layers = []
        for _ in range(n_lower):
            conv_layers.append(nn.Conv1d(d_model, d_model,
                                           kernel_size=lower_kernel, padding=pad))
        self.lower_convs = nn.ModuleList(conv_layers)
        self.lower_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_lower)],
        )

        # Upper stack stays a transformer.
        upper_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.upper_transformer = nn.TransformerEncoder(upper_layer, num_layers=n_upper)
        del self.transformer

        # Aux head supervised by QRS-binary GT (trainer derives it via
        # _aux_targets_from_main with aux_target='qrs_binary').
        self.aux_target = "qrs_binary"
        self.aux_n_classes = 2
        self.aux_head = nn.Linear(d_model, self.aux_n_classes)

        # Mid-stack inject projection: cat(h_lower, qrs(1), pace(1), aux_softmax(2)).
        self.priors_inject_proj = nn.Linear(d_model + 4, d_model)

        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_hybrid_aux_qrs_pace"
        self.model_config["mid_split"] = n_lower
        self.model_config["lower_kernel"] = int(lower_kernel)
        self.model_config["aux_target"] = self.aux_target

    @staticmethod
    def _split_signal_qrs_pace(x: torch.Tensor) -> tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 3:
            raise ValueError(
                f"Hybrid model expects [B, 3, T] input, got {tuple(x.shape)}"
            )
        return x[:, 0], x[:, 1], x[:, 2]

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
        h = self.patch_embed(patches)                                    # [B, P, d]
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)

        # Lower CNN stack: convs operate on (B, d, P) — transpose, conv, transpose.
        h_lower = h
        for conv, norm in zip(self.lower_convs, self.lower_norms):
            residual = h_lower
            h_lower = h_lower.transpose(1, 2)                            # [B, d, P]
            h_lower = conv(h_lower)
            h_lower = h_lower.transpose(1, 2)                            # [B, P, d]
            h_lower = torch.nn.functional.gelu(h_lower)
            h_lower = norm(h_lower + residual)                           # post-norm residual

        # Aux head + softmax for concat.
        aux_logits = self.aux_head(h_lower)                              # [B, P, 2]
        aux_probs = torch.softmax(aux_logits, dim=-1)

        # Per-patch rule priors.
        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)  # [B, P]
        pace_per_patch = pace_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = torch.stack([qrs_per_patch, pace_per_patch], dim=-1)     # [B, P, 2]

        h_upper_in = self.priors_inject_proj(
            torch.cat([h_lower, rules, aux_probs], dim=-1),
        )
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_logits


class FrameClassifierViTRegTriStageInjectQRSPace(FrameClassifierViTReg,
                                                    _QRSInjectionMixin):
    """v41: 12-layer tri-stage transformer.

    Stages (default split for 12-layer model):
      * L1..L4   (lower): raw signal only
      * L5..L8   (mid):   rule prior (QRS + pacing) injected after L4
      * L9..L12  (upper): learnable aux (QRS-binary, softmax) injected after L8

    Rationale: rule prior gives the mid stack a "where the QRS likely is" hint
    derived from a signal-processing detector that has zero training cost; the
    aux head — supervised by GT QRS-binary frame labels — then commits the
    upper stack to a learned segmentation that can correct the rule prior's
    FP/FN.

    Returns (cls_logits, reg_offsets, aux_logits) so ``fit_reg_aux`` applies.

    Input shape: ``[B, 3, T]`` (signal, QRS 1-pt indicator, pacing 1-pt
    indicator) — same as v33/v38.
    """

    def __init__(self, lower_split: int = 4, mid_split: int = 8, **kwargs):
        super().__init__(**kwargs)
        d_model = self.model_config["d_model"]
        n_heads = self.model_config["n_heads"]
        ff = self.model_config["ff"]
        dropout = self.model_config["dropout"]
        n_total = self.model_config["n_layers"]
        n_lower = int(lower_split)
        n_mid = int(mid_split) - int(lower_split)
        n_upper = n_total - int(mid_split)
        if not (0 < n_lower and 0 < n_mid and 0 < n_upper
                 and n_lower + n_mid + n_upper == n_total):
            raise ValueError(
                f"lower_split={lower_split} / mid_split={mid_split} invalid for "
                f"n_layers={n_total}: need 0 < lower < mid < n_layers."
            )

        def _make_stack(n: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=n)

        self.lower_transformer = _make_stack(n_lower)
        self.mid_transformer = _make_stack(n_mid)
        self.upper_transformer = _make_stack(n_upper)
        del self.transformer

        # Inject 1: rule prior (QRS + pacing) — 2 extra dims.
        self.rule_inject_proj = nn.Linear(d_model + 2, d_model)
        # Inject 2: learnable aux (QRS-binary softmax) — 2 extra dims.
        self.aux_target = "qrs_binary"
        self.aux_n_classes = 2
        self.aux_head = nn.Linear(d_model, self.aux_n_classes)
        self.aux_inject_proj = nn.Linear(d_model + self.aux_n_classes, d_model)

        self.model_config = dict(self.model_config)
        self.model_config["arch"] = "vit_reg_tri_stage_inject_qrs_pace"
        self.model_config["lower_split"] = n_lower
        self.model_config["mid_split"] = int(mid_split)
        self.model_config["aux_target"] = self.aux_target

    @staticmethod
    def _split_signal_qrs_pace(x: torch.Tensor) -> tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 3:
            raise ValueError(
                f"Tri-stage model expects [B, 3, T] input, got {tuple(x.shape)}"
            )
        return x[:, 0], x[:, 1], x[:, 2]

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

        # Stage 1: lower (raw signal only).
        h_lower = self.lower_transformer(h)

        # Inject rule prior at 1/3 point.
        qrs_per_patch = qrs_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        pace_per_patch = pace_pp.view(B, n_patches, self.patch_size).amax(dim=2)
        rules = torch.stack([qrs_per_patch, pace_per_patch], dim=-1)         # [B, P, 2]
        h_mid_in = self.rule_inject_proj(torch.cat([h_lower, rules], dim=-1))

        # Stage 2: mid (with rule prior).
        h_mid = self.mid_transformer(h_mid_in)

        # Inject learnable aux at 2/3 point. Aux head reads the mid stack output
        # (which has had a chance to refine the rule prior) and produces the
        # QRS-binary softmax that the upper stack receives.
        aux_logits = self.aux_head(h_mid)                                    # [B, P, 2]
        aux_probs = torch.softmax(aux_logits, dim=-1)
        h_upper_in = self.aux_inject_proj(torch.cat([h_mid, aux_probs], dim=-1))

        # Stage 3: upper.
        h_upper = self.upper_transformer(h_upper_in)
        cls_logits = self.head(h_upper)
        reg_offsets = self.reg_head(h_upper)
        return cls_logits, reg_offsets, aux_logits


_ARCH_REGISTRY: dict[str, type[nn.Module]] = {
    "frame_classifier":         FrameClassifier,
    "vit":                      FrameClassifierViT,
    "vit_reg":                  FrameClassifierViTReg,
    "vit_reg_aux":              FrameClassifierViTRegAux,
    "vit_reg_aux_concat":       FrameClassifierViTRegAuxConcat,
    "vit_reg_multitask":        FrameClassifierViTRegMultiTask,
    "vit_reg_multiin":          FrameClassifierViTRegMultiIn,
    "vit_reg_patch_inject_qrs": FrameClassifierViTRegPatchInjectQRS,
    "vit_reg_mid_inject_qrs":   FrameClassifierViTRegMidInjectQRS,
    "vit_reg_multi_inject_qrs": FrameClassifierViTRegMultiInjectQRS,
    "vit_reg_mid_inject_qrs_pace": FrameClassifierViTRegMidInjectQRSPace,
    "vit_reg_hybrid_aux_qrs_pace": FrameClassifierViTRegHybridAuxQRSPace,
    "vit_reg_tri_stage_inject_qrs_pace": FrameClassifierViTRegTriStageInjectQRSPace,
}


def _register_variant_arches():
    """Lazy-register v42 architectural variants — avoids circular import."""
    from openecg.stage2.model_variants import (
        FrameClassifierMambaDualAux, FrameClassifierMambaPure,
        FrameClassifierMambaUpper, FrameClassifierSparseAttnUpper,
        FrameClassifierTriStageDualAux,
    )
    _ARCH_REGISTRY["vit_reg_hybrid_aux_qrs_pace_mamba"] = FrameClassifierMambaUpper
    _ARCH_REGISTRY["vit_reg_hybrid_aux_qrs_pace_sparseattn"] = FrameClassifierSparseAttnUpper
    _ARCH_REGISTRY["vit_reg_hybrid_aux_qrs_pace_mamba_dual"] = FrameClassifierMambaDualAux
    _ARCH_REGISTRY["vit_reg_tri_stage_inject_qrs_pace_dual"] = FrameClassifierTriStageDualAux
    _ARCH_REGISTRY["mamba_pure"] = FrameClassifierMambaPure
    # v52+ noaux 2-channel variants — used by v54i / v54m. Register so
    # `load_model_from_ckpt` can rebuild from ckpt metadata.
    from openecg.stage2.model_variants import (  # noqa: F401  (already imported above for some)
        FrameClassifierTransformerDualAux2Ch,
        FrameClassifierTransformerDualAuxNoReg2Ch,
        FrameClassifierTransformerNoAux2Ch,
        FrameClassifierTransformerNoAux1Ch,
        FrameClassifierTransformerNoAux2ChBoxIn,
        FrameClassifierTransformerNoAux2ChRankEmb,
        FrameClassifierCnnOnly2Ch,
    )
    _ARCH_REGISTRY["vit_transformer_dualaux_2ch"]          = FrameClassifierTransformerDualAux2Ch
    _ARCH_REGISTRY["vit_transformer_dualaux_noreg_2ch"]    = FrameClassifierTransformerDualAuxNoReg2Ch
    _ARCH_REGISTRY["vit_transformer_noaux_2ch"]            = FrameClassifierTransformerNoAux2Ch
    _ARCH_REGISTRY["vit_transformer_noaux_1ch"]            = FrameClassifierTransformerNoAux1Ch
    _ARCH_REGISTRY["vit_transformer_noaux_2ch_boxin"]      = FrameClassifierTransformerNoAux2ChBoxIn
    _ARCH_REGISTRY["vit_transformer_noaux_2ch_rankembed"]  = FrameClassifierTransformerNoAux2ChRankEmb
    _ARCH_REGISTRY["vit_cnn_only_2ch"]                     = FrameClassifierCnnOnly2Ch


def load_model_from_ckpt(ckpt_path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Construct the matching FrameClassifier* variant from a saved checkpoint.

    Reads `model_config['arch']` to dispatch to the correct class. Returns
    (model, blob) where blob is the full checkpoint dict so callers can
    inspect metrics/extra without reloading.

    Old FrameClassifier checkpoints (no `arch` field) are dispatched via
    presence of `lead_emb`/conv keys.
    """
    from openecg.stage2.train import load_checkpoint_blob
    blob = load_checkpoint_blob(Path(ckpt_path))
    cfg = dict(blob.get("model_config") or {})
    arch = cfg.pop("arch", "frame_classifier")
    # Register v42 variants on demand — they live in a separate module
    # to keep model.py focused on the canonical architectures.
    if (arch.endswith("_mamba") or arch.endswith("_sparseattn")
            or arch == "mamba_pure" or arch.startswith("vit_transformer_")
            or arch == "vit_cnn_only_2ch"):
        _register_variant_arches()
    if arch not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unknown arch '{arch}' in {ckpt_path}; "
            f"known: {sorted(_ARCH_REGISTRY)}"
        )
    cls = _ARCH_REGISTRY[arch]
    # The hybrid / tri-stage archs set `aux_target` internally as metadata only;
    # their ctors do not accept it (and the kwarg cascades to the parent ViT,
    # which also rejects it). Drop it for archs that hard-code the value.
    if arch in ("vit_reg_hybrid_aux_qrs_pace",
                 "vit_reg_tri_stage_inject_qrs_pace",
                 "vit_reg_hybrid_aux_qrs_pace_mamba",
                 "vit_reg_hybrid_aux_qrs_pace_sparseattn",
                 "vit_reg_hybrid_aux_qrs_pace_mamba_dual",
                 "vit_reg_tri_stage_inject_qrs_pace_dual"):
        cfg.pop("aux_target", None)
    # Translate mamba-prefixed config keys to ctor kwarg names.
    if "_mamba" in arch or arch == "mamba_pure":
        if "mamba_d_state" in cfg: cfg["d_state"] = cfg.pop("mamba_d_state")
        if "mamba_d_conv"  in cfg: cfg["d_conv"]  = cfg.pop("mamba_d_conv")
        if "mamba_expand"  in cfg: cfg["expand"]  = cfg.pop("mamba_expand")
        cfg.pop("mamba_version", None)
    if arch == "vit_reg_hybrid_aux_qrs_pace_sparseattn":
        # attn_window is a ctor kwarg, leave as-is.
        pass
    # `aux_target` is metadata on the noaux 2-ch variants too (drop before ctor).
    if arch in ("vit_transformer_noaux_2ch",
                 "vit_transformer_noaux_1ch",
                 "vit_transformer_noaux_2ch_boxin",
                 "vit_transformer_noaux_2ch_rankembed",
                 "vit_cnn_only_2ch"):
        cfg.pop("aux_target", None)
        cfg.pop("use_aux", None)
    # n_input_channels is metadata-only on hybrid archs (input shape derived).
    cfg.pop("n_input_channels", None)
    model = cls(**cfg)
    model.load_state_dict(blob["model_state"])
    model = model.to(device).train(False)
    return model, blob
