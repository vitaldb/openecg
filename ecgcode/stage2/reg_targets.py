"""Boundary regression targets for FrameClassifierViTReg.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.2
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ecgcode import eval as ee


REG_CHANNELS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")
_REG_INDEX = {  # (super_class, edge) -> channel index
    (ee.SUPER_P, "on"): 0, (ee.SUPER_P, "off"): 1,
    (ee.SUPER_QRS, "on"): 2, (ee.SUPER_QRS, "off"): 3,
    (ee.SUPER_T, "on"): 4, (ee.SUPER_T, "off"): 5,
}


def boundary_regression_targets(
    hard_labels: np.ndarray,
    samples_per_frame: int = 5,
    window_frames: int = 5,
    ignore_index: int = ee.IGNORE_INDEX,
):
    """Build per-frame regression targets and active masks from hard labels.

    Convention (matches extract_boundaries):
      - off boundary sample = transition_frame * spf - 1
      - on  boundary sample = transition_frame * spf

    Returns (targets[T, 6] float32, mask[T, 6] bool).
    target[f, k] = signed sample offset (boundary - frame_start) if a boundary
    of channel k lies within ±window_frames of f, else 0. mask[f, k] = True iff
    such a boundary exists. Frames where hard_labels[f] == ignore_index get
    mask all-False regardless.
    """
    labels = np.asarray(hard_labels, dtype=np.int64)
    T = len(labels)
    targets = np.zeros((T, 6), dtype=np.float32)
    mask = np.zeros((T, 6), dtype=bool)
    if T == 0:
        return targets, mask

    boundaries: list[tuple[int, int]] = []
    prev = int(labels[0])
    for f in range(1, T):
        cur = int(labels[f])
        if prev == ignore_index or cur == ignore_index or cur == prev:
            prev = cur
            continue
        if (prev, "off") in _REG_INDEX:
            boundaries.append((_REG_INDEX[(prev, "off")], f * samples_per_frame - 1))
        if (cur, "on") in _REG_INDEX:
            boundaries.append((_REG_INDEX[(cur, "on")], f * samples_per_frame))
        prev = cur

    radius = window_frames * samples_per_frame
    for ch, b_sample in boundaries:
        for f in range(T):
            f_sample = f * samples_per_frame
            offset = b_sample - f_sample
            if abs(offset) <= radius:
                if (not mask[f, ch]) or abs(offset) < abs(targets[f, ch]):
                    targets[f, ch] = float(offset)
                    mask[f, ch] = True

    for f in range(T):
        if int(labels[f]) == ignore_index:
            mask[f, :] = False
            targets[f, :] = 0.0
    return targets, mask


class RegLabelDataset(Dataset):
    """Wrap a base dataset to additionally yield reg targets + mask.

    Base must yield (sig, lead_id, hard_labels[T]). This wrapper yields
    (sig, lead_id, hard_labels, reg_targets[T,6], reg_mask[T,6]).
    """

    def __init__(self, base, samples_per_frame: int = 5,
                 window_frames: int = 5,
                 ignore_index: int = ee.IGNORE_INDEX):
        self.base = base
        self.samples_per_frame = int(samples_per_frame)
        self.window_frames = int(window_frames)
        self.ignore_index = int(ignore_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sig, lead_id, labels = self.base[idx]
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
        targets, mask = boundary_regression_targets(
            labels_np, samples_per_frame=self.samples_per_frame,
            window_frames=self.window_frames, ignore_index=self.ignore_index,
        )
        return (sig, lead_id, labels,
                torch.from_numpy(targets), torch.from_numpy(mask))

    def label_counts(self):
        return self.base.label_counts()
