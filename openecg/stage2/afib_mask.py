"""AFib-window P-wave masking utilities (strict don't-care).

When ``openecg.is_afib(window) == True`` we want P-wave information to be
fully invisible to BOTH the loss (train) and metric (val/test) — neither
the model's P output nor the GT's P label should contribute to any
TP/FP/FN counter or gradient.

Strict don't-care (no logsumexp fold — that approach leaks the P logit
into OTHER and therefore into QRS/T loss):

* Logits:        ``logits[..., P] = -large`` — model's P output is
                 silently overwritten before any softmax / argmax /
                 CE call. The remaining (OTHER, QRS, T) channels compete
                 cleanly. Any value the model put on P has zero effect.
* Hard labels:   ``P -> IGNORE_INDEX`` on AFib rows — frames whose GT is
                 P are dropped from CE entirely (no loss / no gradient).
                 Non-P labels (OTHER/QRS/T) keep their supervision.
* Soft targets:  P-column added to OTHER-column, P-column zeroed.
                 (KL works over distributions, so collapsing P mass into
                 OTHER + suppressed P column is the soft-equivalent of the
                 hard-label rule.)
* Reg mask:      ``mask[..., p_on:p_off+1] = False`` on AFib rows.
* Metric pred:   AFib frames where ``pred == P`` OR ``true == P`` are
                 dropped from all per-class counters by routing them
                 through ``true = IGNORE_INDEX`` (``frame_f1`` already
                 excludes those). No P TP/FP/FN, and OTHER/QRS/T counters
                 ignore any P-related decision.

Layout convention: class index is the LAST dim of any logits/soft-target
tensor; time index is the second-to-last for ``[B, T, C]`` style.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import is_afib
from openecg import eval as ee


SUPER_OTHER = ee.SUPER_OTHER
SUPER_P = ee.SUPER_P
IGNORE_INDEX = ee.IGNORE_INDEX
REG_P_COLS = (0, 1)  # p_on, p_off in reg_targets.REG_CHANNELS

# Large negative finite logit used to suppress the P channel. -1e4 in fp32
# gives exp(-1e4) underflowing to 0 in softmax while remaining well clear
# of -inf, so gradients stay finite (the channel just gets zero share).
_LARGE_NEG = -1.0e4


# ---------------------------------------------------------------- core ----


def compute_afib_flags(sigs: torch.Tensor, fs: int) -> torch.Tensor:
    """Per-window AFib decision on a CPU pass over ``openecg.is_afib``.

    ``sigs``: ``[B, S]`` or ``[B, C, S]`` — channel 0 is the raw signal.
    Returns a CPU bool tensor of shape ``[B]``.
    """
    if sigs.ndim == 3:
        sigs_1d = sigs[:, 0, :]
    elif sigs.ndim == 2:
        sigs_1d = sigs
    else:
        raise ValueError(f"compute_afib_flags expected 2D/3D input, got {sigs.shape}")
    arr = sigs_1d.detach().cpu().numpy().astype(np.float64)
    out = np.zeros(arr.shape[0], dtype=bool)
    for i, s in enumerate(arr):
        try:
            out[i] = bool(is_afib(s, int(fs)))
        except Exception:
            out[i] = False
    return torch.from_numpy(out)


def _flag_broadcast(flag: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape ``flag`` [B] to ``[B, 1, ..., 1]`` with ``ndim`` dims."""
    return flag.view(-1, *([1] * (ndim - 1))).to(dtype=torch.bool)


def pfold_logits(
    logits: torch.Tensor,
    afib_flag: torch.Tensor,
    super_p: int = SUPER_P,
    super_other: int = SUPER_OTHER,  # kept for sig stability; unused now
) -> torch.Tensor:
    """Suppress the P channel on AFib rows: ``logits[..., P] = -LARGE``.

    No folding into OTHER — that would propagate the model's P logit into
    OTHER's softmax mass and corrupt QRS/T loss. Strict don't-care
    requires the P logit to leave the competition entirely; assigning a
    very negative finite value (``_LARGE_NEG = -1e4``) makes softmax
    contribution and gradient w.r.t. the original P logit both zero in
    fp32.

    ``logits``: any shape ending in class dim, e.g. ``[B, T, C]`` or
    ``[B, C]``. ``afib_flag``: ``[B]`` bool.
    Returns a new tensor; input is not modified in place.
    """
    del super_other  # not used in the suppress-only variant
    if not afib_flag.any():
        return logits
    out = logits.clone()
    p = out[..., super_p]
    flag = _flag_broadcast(afib_flag.to(logits.device), p.ndim).expand_as(p)
    out[..., super_p] = torch.where(
        flag, torch.full_like(p, _LARGE_NEG), p
    )
    return out


def pfold_labels(
    labels: torch.Tensor,
    afib_flag: torch.Tensor,
    super_p: int = SUPER_P,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Replace ``labels == P`` with ``IGNORE_INDEX`` on AFib rows so those
    frames drop out of CE / frame_f1 entirely (no loss, no metric count).

    Mapping P -> OTHER instead would still penalise the model for not
    predicting OTHER, which is not strict don't-care. IGNORE removes the
    frame from both sides.

    ``labels``: ``[B, T]`` long (also handles ``[B]``).
    ``afib_flag``: ``[B]`` bool.
    Existing IGNORE_INDEX positions are left untouched.
    """
    if not afib_flag.any():
        return labels
    out = labels.clone()
    flag = _flag_broadcast(afib_flag.to(labels.device), out.ndim).expand_as(out)
    target = flag & (out == super_p)
    out[target] = ignore_index
    return out


def pfold_soft_target(
    soft: torch.Tensor,
    afib_flag: torch.Tensor,
    super_p: int = SUPER_P,
    super_other: int = SUPER_OTHER,
) -> torch.Tensor:
    """Fold P-column into OTHER-column on AFib rows of a soft target.

    ``soft``: ``[B, T, C]`` non-negative (rows may not normalise to 1).
    """
    if not afib_flag.any():
        return soft
    out = soft.clone()
    flag = _flag_broadcast(
        afib_flag.to(soft.device), out[..., super_other].ndim,
    ).expand_as(out[..., super_other])
    out[..., super_other] = torch.where(
        flag, out[..., super_other] + out[..., super_p], out[..., super_other],
    )
    out[..., super_p] = torch.where(
        flag, torch.zeros_like(out[..., super_p]), out[..., super_p],
    )
    return out


def pfold_reg_mask(reg_mask: torch.Tensor, afib_flag: torch.Tensor) -> torch.Tensor:
    """Zero p_on / p_off columns of a ``[B, T, 6]`` regression mask on AFib rows."""
    if not afib_flag.any():
        return reg_mask
    out = reg_mask.clone()
    flag = afib_flag.to(reg_mask.device).view(-1, 1, 1)
    for col in REG_P_COLS:
        out[..., col] = out[..., col] & ~flag.squeeze(-1).bool()
    return out


# ----------------------------------------------------------- metric side ----


def pfold_predictions_arrays(
    pred: np.ndarray,
    true: np.ndarray,
    afib_per_frame: np.ndarray,
    super_p: int = SUPER_P,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop AFib frames whose pred or true is P from all per-class
    counters by setting ``true[drop] = IGNORE_INDEX``. ``frame_f1``
    already excludes IGNORE_INDEX rows, so neither side counts toward
    P / OTHER / QRS / T TP/FP/FN.

    All three inputs are 1-D arrays of equal length. Called inside
    ``run_eval_*`` right before ``openecg.eval.frame_f1``.
    """
    pred = np.asarray(pred).copy()
    true = np.asarray(true).copy()
    afib = np.asarray(afib_per_frame, dtype=bool)
    if afib.shape != pred.shape:
        raise ValueError(
            f"afib_per_frame shape {afib.shape} must equal pred shape {pred.shape}"
        )
    drop = afib & ((pred == super_p) | (true == super_p))
    true[drop] = ignore_index
    return pred, true


def expand_window_flag_to_frames(
    afib_per_window: np.ndarray, frames_per_window: int,
) -> np.ndarray:
    """Repeat a per-window AFib flag to a per-frame mask of length
    ``len(afib_per_window) * frames_per_window``.
    """
    afib = np.asarray(afib_per_window, dtype=bool)
    return np.repeat(afib, int(frames_per_window))


def filter_p_boundaries(
    boundaries: dict, gt: dict, is_af: bool,
) -> tuple[dict, dict]:
    """Drop ``p_on`` / ``p_off`` from both predicted and GT boundary dicts
    when the window/record is AFib. Mirrors the eval-only logic in
    ``scripts/eval_pwave_afib_masked._filter_p``.
    """
    if not is_af:
        return boundaries, gt
    bnd = {k: v for k, v in boundaries.items() if not k.startswith("p_")}
    gtk = {k: v for k, v in gt.items() if not k.startswith("p_")}
    return bnd, gtk


# ----------------------------------------------------------- dataset ----


class AFibAwareDataset(Dataset):
    """Wrap a base dataset so each item exposes its AFib flag.

    Base must yield a tuple whose first element is the signal tensor
    (``[S]`` or ``[C, S]``). The wrapper:

      1. Runs ``openecg.is_afib`` on channel 0 of the native signal (cached
         per index — AFib decision is deterministic).
      2. Optionally GT-folds the labels: ``label==SUPER_P -> IGNORE_INDEX``
         on AFib items (legacy ``--mask`` behaviour from v44, kept off by
         default — strict logit-fold is preferred and happens in the loss).
      3. Returns ``(*base_item, afib_flag)`` where ``afib_flag`` is a
         scalar bool tensor.

    Wrap order matters when ``RegLabelDataset`` is also used: place
    ``AFibAwareDataset`` BEFORE ``RegLabelDataset`` so the reg-target
    builder still sees real P labels (we mask boundaries via the loss,
    not via label rewriting). ``fold_labels=True`` is the legacy
    asymmetric mode and should be combined with ``RegLabelDataset``
    AFTER this wrapper so reg targets inherit the IGNORE positions.
    """

    def __init__(
        self,
        base,
        fs: int,
        *,
        fold_labels: bool = False,
        p_class: int = SUPER_P,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.base = base
        self.fs = int(fs)
        self.fold_labels = bool(fold_labels)
        self.p_class = int(p_class)
        self.ignore_index = int(ignore_index)
        self._cache: dict[int, bool] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _decide_afib(self, idx: int, sig_arr: np.ndarray) -> bool:
        if idx in self._cache:
            return self._cache[idx]
        try:
            v = bool(is_afib(sig_arr, self.fs))
        except Exception:
            v = False
        self._cache[idx] = v
        return v

    def __getitem__(self, idx: int):
        item = self.base[idx]
        sig = item[0]
        sig_arr = (
            sig.detach().cpu().numpy() if hasattr(sig, "detach") else np.asarray(sig)
        )
        if sig_arr.ndim == 2:
            sig_arr = sig_arr[0]
        flag = self._decide_afib(idx, sig_arr)

        if self.fold_labels and flag and len(item) >= 3:
            labels = item[2]
            lbl_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach") else np.asarray(labels)
            )
            lbl_np = lbl_np.copy()
            lbl_np[lbl_np == self.p_class] = self.ignore_index
            labels_new = (
                torch.from_numpy(lbl_np)
                if hasattr(item[2], "detach") else lbl_np
            )
            item = (item[0], item[1], labels_new, *item[3:])

        return (*item, torch.tensor(flag, dtype=torch.bool))

    def label_counts(self):
        return (
            self.base.label_counts()
            if hasattr(self.base, "label_counts") else None
        )


__all__ = [
    "compute_afib_flags",
    "pfold_logits",
    "pfold_labels",
    "pfold_soft_target",
    "pfold_reg_mask",
    "pfold_predictions_arrays",
    "expand_window_flag_to_frames",
    "filter_p_boundaries",
    "AFibAwareDataset",
    "SUPER_OTHER",
    "SUPER_P",
    "IGNORE_INDEX",
]
