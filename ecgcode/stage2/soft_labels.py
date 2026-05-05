"""Soft labels at frame transitions to soften per-frame CE at boundaries.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.1
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ecgcode import eval as ee


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
