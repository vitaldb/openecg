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
    """

    def __init__(self, record_ids):
        self.items = []
        self.cache = {}

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


class LUDBFrameDatasetAugmented(LUDBFrameDataset):
    """Same data, with training-time augmentation: gaussian noise + time shift + amplitude scaling."""

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

        # Time shift (samples at 250Hz; labels at 50Hz so shift by samples//5)
        shift = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
        if shift != 0:
            sig_np = np.roll(sig_np, shift)
            label_shift = shift // 5  # 5 samples = 1 frame at 50Hz
            if label_shift != 0:
                labels_np = np.roll(labels_np, label_shift)

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
