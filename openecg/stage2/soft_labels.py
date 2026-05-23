"""Soft labels at frame transitions to soften per-frame CE at boundaries.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.1
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as ee


def soft_boundary_labels(
    labels: np.ndarray,
    alpha: float = 0.7,
    n_classes: int = 4,
    ignore_index: int = ee.IGNORE_INDEX,
) -> np.ndarray:
    """Convert a hard label sequence to per-frame soft targets.

    For every transition (i, i+1) with labels[i] != labels[i+1] (and
    neither equal to ignore_index):
        soft[i  ] = alpha · onehot(labels[i  ]) + (1-alpha) · onehot(labels[i+1])
        soft[i+1] = (1-alpha) · onehot(labels[i  ]) + alpha · onehot(labels[i+1])
    Frames in multiple transitions take the later application.
    Rows for ignore_index frames are all zero (signals "skip in loss").
    """
    labels = np.asarray(labels, dtype=np.int64)
    T = len(labels)
    soft = np.zeros((T, n_classes), dtype=np.float32)
    for i in range(T):
        c = int(labels[i])
        if c != ignore_index and 0 <= c < n_classes:
            soft[i, c] = 1.0
    for i in range(T - 1):
        a, b = int(labels[i]), int(labels[i + 1])
        if a == ignore_index or b == ignore_index or a == b:
            continue
        if not (0 <= a < n_classes and 0 <= b < n_classes):
            continue
        soft[i] = 0.0
        soft[i + 1] = 0.0
        soft[i, a] = alpha
        soft[i, b] = 1.0 - alpha
        soft[i + 1, a] = 1.0 - alpha
        soft[i + 1, b] = alpha
    return soft


class SoftLabelDataset(Dataset):
    """Wraps a hard-label Dataset.

    The base dataset must yield (sig, lead_id, hard_labels[T] long). This
    wrapper instead yields (sig, lead_id, soft_labels[T, n_classes] float32)
    by applying soft_boundary_labels to each item.
    """

    def __init__(self, base, alpha: float = 0.7, n_classes: int = 4,
                 ignore_index: int = ee.IGNORE_INDEX):
        self.base = base
        self.alpha = float(alpha)
        self.n_classes = int(n_classes)
        self.ignore_index = int(ignore_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sig, lead_id, labels = self.base[idx]
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
        soft = soft_boundary_labels(
            labels_np, alpha=self.alpha, n_classes=self.n_classes,
            ignore_index=self.ignore_index,
        )
        return sig, lead_id, torch.from_numpy(soft)

    def label_counts(self):
        return self.base.label_counts()


def t_boundary_soft_target_batched(
    labels: torch.Tensor, n_classes: int = 4, alpha: float = 0.7,
    radius: int = 1, t_class: int = 3, ignore_index: int = ee.IGNORE_INDEX,
) -> torch.Tensor:
    """Build a soft-target tensor that softens **only T_on / T_off** frames.

    QTDB worst-case analysis (v54a) showed several records where the model
    misses every T boundary because annotator-level T_on/T_off uncertainty
    is wider than the 20 ms-per-frame quantization grid. Hard CE penalises
    those fuzzy frames identically to a clean miss; softening lets the
    model learn the boundary as a probability ramp.

    For every transition where exactly one side is the T class (BG/QRS↔T,
    1-D windowing) and the other is non-T non-ignore:
        * The two frames straddling the transition are blended:
            soft[i  , T_class] = alpha,  soft[i  , other_class] = 1-alpha
            soft[i+1, T_class] = 1-alpha (if other side stays T), etc.
        * Optionally extends the blend ``radius`` frames outward (default 1,
          i.e. only the two straddling frames; radius=2 also adds the two
          neighbours with a half-strength blend).
    Other transitions (P/QRS boundaries) keep hard one-hot.
    IGNORE_INDEX rows stay all-zero so the kl_cross_entropy mask drops them.

    Args:
        labels: ``[B, T]`` long tensor.
        n_classes: usually 4 (BG / P / QRS / T) for the v45 head.
        alpha: weight of the dominant (this-frame) class. ``alpha=1.0``
               reduces to one-hot (pure hard label).
        radius: number of frames per side of the transition to soften.
                1 (default) softens the 2 straddling frames; 2 also softens
                the 1-out neighbours.
        t_class: integer T class index (3 by default).
    Returns: ``[B, T, n_classes]`` float32.
    """
    if labels.dim() != 2:
        raise ValueError(f"expected [B, T] labels, got {tuple(labels.shape)}")
    B, T = labels.shape
    device = labels.device
    soft = torch.zeros((B, T, n_classes), dtype=torch.float32, device=device)
    # One-hot for non-ignore frames.
    valid_mask = (labels != ignore_index) & (labels >= 0) & (labels < n_classes)
    flat_labels = labels.clone()
    flat_labels[~valid_mask] = 0
    soft.scatter_(2, flat_labels.unsqueeze(-1), 1.0)
    soft[~valid_mask] = 0.0
    if alpha >= 1.0 - 1e-6:
        return soft

    # Find T-boundary transitions: labels[:, i] != labels[:, i+1] and at least
    # one side equals t_class.
    left = labels[:, :-1]
    right = labels[:, 1:]
    is_transition = (left != right) & (left != ignore_index) & (right != ignore_index)
    is_t_boundary = is_transition & ((left == t_class) | (right == t_class))
    # Only the two straddling frames (radius=1). Extend for radius > 1.
    for r in range(radius):
        # Frame i (positions [0..T-2])
        bi, ti = torch.where(is_t_boundary)
        if bi.numel() == 0:
            continue
        i0 = ti - r
        i1 = ti + 1 + r
        # Clamp inside the window and bail on duplicates safely.
        mask = (i0 >= 0) & (i1 < T)
        bi = bi[mask]; i0 = i0[mask]; i1 = i1[mask]
        if bi.numel() == 0:
            continue
        a = labels[bi, i0]
        b = labels[bi, i1]
        valid_pair = (a != ignore_index) & (b != ignore_index) & (a != b) \
                     & (a >= 0) & (a < n_classes) & (b >= 0) & (b < n_classes)
        bi = bi[valid_pair]; i0 = i0[valid_pair]; i1 = i1[valid_pair]
        if bi.numel() == 0:
            continue
        a = labels[bi, i0]
        b = labels[bi, i1]
        # Distance-weighted blend: further from the transition → closer to
        # hard. r=0 (straddling) gets full alpha blend; r=1 uses
        # 0.5 * (1-alpha) extra blend so the ramp tapers.
        local_alpha = alpha + (1.0 - alpha) * (r / max(radius, 1)) * 0.5
        # Wipe the rows we are about to rewrite.
        soft[bi, i0] = 0.0
        soft[bi, i1] = 0.0
        soft[bi, i0, a] = local_alpha
        soft[bi, i0, b] = 1.0 - local_alpha
        soft[bi, i1, a] = 1.0 - local_alpha
        soft[bi, i1, b] = local_alpha
    return soft
