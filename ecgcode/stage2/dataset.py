"""LUDB Stage 2 dataset: signal+lead_id to frame labels (4-class supercategory)."""

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from ecgcode import eval as ee
from ecgcode import ludb

FS_NATIVE = 500
FS_INPUT = 250
FRAME_MS = 20
N_CLASSES = 4


class LUDBFrameDataset(Dataset):
    """Eager-load LUDB train/val sequences. Memory: ~30 MB for 1908 sequences.

    __getitem__ returns: (signal[2500] float32, lead_id scalar long, labels[500] long).

    `mask_unlabeled_edges`: when True, frames before the first GT boundary
    (minus margin) and after the last GT boundary (plus margin) are set to
    IGNORE_INDEX so the model is not penalized for correctly detecting
    edge beats that the cardiologist did not annotate. LUDB cardiologists
    skip ~1.4s at start and ~1.3s at end on average (per record-lead).
    """

    def __init__(self, record_ids, mask_unlabeled_edges=False, edge_margin_ms=100):
        self.items = []
        self.cache = {}

        margin_250 = int(round(edge_margin_ms * FS_INPUT / 1000.0))
        spf = int(round(FRAME_MS * FS_INPUT / 1000.0))  # 5
        for rid in record_ids:
            try:
                record = ludb.load_record(rid)
            except Exception:
                continue
            for lead_idx, lead in enumerate(ludb.LEADS_12):
                sig_500 = record[lead]
                sig_250 = scipy_signal.decimate(sig_500, 2, zero_phase=True)
                mean = float(sig_250.mean())
                std = float(sig_250.std()) + 1e-6
                sig_250 = ((sig_250 - mean) / std).astype(np.float32)
                try:
                    gt_ann = ludb.load_annotations(rid, lead)
                except Exception:
                    continue
                labels = ee.gt_to_super_frames(
                    gt_ann, n_samples=len(sig_500), fs=FS_NATIVE, frame_ms=FRAME_MS
                ).astype(np.int64)
                if mask_unlabeled_edges:
                    rng = ludb.labeled_range(rid, lead)
                    if rng is not None:
                        first_250 = max(0, rng[0] // 2 - margin_250)
                        last_250 = rng[1] // 2 + margin_250
                        first_frame = first_250 // spf
                        last_frame = (last_250 + spf - 1) // spf
                        labels[:first_frame] = ee.IGNORE_INDEX
                        labels[last_frame + 1:] = ee.IGNORE_INDEX
                self.cache[(rid, lead)] = (sig_250, lead_idx, labels)
                self.items.append((rid, lead))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rid, lead = self.items[idx]
        sig, lead_idx, labels = self.cache[(rid, lead)]
        return (
            torch.from_numpy(sig),
            torch.tensor(lead_idx, dtype=torch.long),
            torch.from_numpy(labels),
        )

    def label_counts(self):
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for (_, _, labels) in self.cache.values():
            for c in range(N_CLASSES):
                counts[c] += int((labels == c).sum())
        return counts


def compute_class_weights(counts):
    """Soft inverse-sqrt class weights, normalized so sum == n_classes."""
    n = len(counts)
    weights = 1.0 / np.sqrt(counts + 1e-6)
    weights = weights / weights.sum() * n
    return weights.astype(np.float64)


class BoundaryMaskedDataset(Dataset):
    """Wrapper that masks the first/last `mask_frames` of each sequence's
    labels with `ignore_index`. The model's transformer attention can only see
    the 10s window, so boundary frames have one-sided context and unreliable
    predictions; masking them removes that noise from the training signal.

    Default: mask 100 frames each side (= 2s @ 50Hz frame rate, 20% of window).
    Loss must use `ignore_index=ee.IGNORE_INDEX` (=255). Eval should skip the
    same boundary regions when extracting predicted boundaries.
    """

    def __init__(self, base, mask_frames=100, ignore_index=None):
        from ecgcode import eval as ee
        self.base = base
        self.mask_frames = int(mask_frames)
        self.ignore_index = ee.IGNORE_INDEX if ignore_index is None else int(ignore_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sig, lead_id, labels = self.base[idx]
        if self.mask_frames > 0:
            labels = labels.clone()
            labels[:self.mask_frames] = self.ignore_index
            labels[-self.mask_frames:] = self.ignore_index
        return sig, lead_id, labels

    def label_counts(self):
        # Underlying base dataset's class counts (don't subtract masked frames;
        # class weights remain stable across mask choices).
        return self.base.label_counts()

    @property
    def items(self):
        return self.base.items


class LUDBFrameDatasetAugmented(LUDBFrameDataset):
    """Legacy LUDB-only augmentation.

    Kept for old v1.1 experiments. Time shift is edge-filled and frame-aligned
    so ECG events do not wrap around the 10s window.
    """

    def __init__(self, record_ids, noise_sigma=0.05, max_shift=20, amp_scale_range=0.2, seed=None):
        super().__init__(record_ids)
        self.noise_sigma = noise_sigma
        self.max_shift = max_shift
        self.amp_scale_range = amp_scale_range
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        sig, lead_id, labels = super().__getitem__(idx)
        sig_np = sig.numpy()
        labels_np = labels.numpy()

        if self.max_shift > 0:
            from ecgcode.stage2.augment import time_shift_aligned
            max_shift_ms = int(round(self.max_shift * 1000.0 / FS_INPUT))
            sig_np, labels_np = time_shift_aligned(
                sig_np, labels_np, fs_sig=FS_INPUT, frame_ms=FRAME_MS,
                max_shift_ms=max_shift_ms, rng=self.rng,
            )

        # Amplitude scaling
        amp_scale = 1.0 + float(self.rng.uniform(-self.amp_scale_range, self.amp_scale_range))
        sig_np = sig_np * amp_scale

        # Gaussian noise
        noise = self.rng.normal(0, self.noise_sigma, size=sig_np.shape).astype(np.float32)
        sig_np = sig_np + noise

        return (
            torch.from_numpy(sig_np.astype(np.float32)),
            lead_id,
            torch.from_numpy(labels_np.astype(np.int64)),
        )
