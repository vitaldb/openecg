# ecgcode/stage2/multi_dataset.py
"""Combined LUDB + QTDB + ISP dataset for v3 training.

Each sequence: 10s window @ 250Hz (2500 samples), z-normalized, with lead_id and frame_labels (50Hz, 500 frames).
"""

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from ecgcode import eval as ee
from ecgcode import isp, ludb, qtdb

FRAME_MS = 20
WINDOW_SAMPLES = 2500       # 10s @ 250Hz
WINDOW_FRAMES = 500
N_CLASSES = 4

# QTDB lead names -> LUDB lead-id (LEADS_12 = i, ii, iii, avr, avl, avf, v1..v6).
# Anything not in this map is skipped to avoid polluting the lead embedding.
QTDB_LEAD_TO_LUDB_ID = {
    "I": 0, "i": 0,
    "MLII": 1, "ML II": 1, "II": 1, "ii": 1,
    "III": 2, "iii": 2,
    "aVR": 3, "AVR": 3, "avr": 3,
    "aVL": 4, "AVL": 4, "avl": 4,
    "aVF": 5, "AVF": 5, "avf": 5,
    "V1": 6, "v1": 6,
    "V2": 7, "v2": 7,
    "V3": 8, "v3": 8,
    "V4": 9, "v4": 9,
    "V5": 10, "v5": 10,
    "V6": 11, "v6": 11,
}


def _normalize(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


def _decimate_to_250(sig, fs_native):
    factor = fs_native // 250
    if factor == 1:
        return sig.astype(np.float64)
    return scipy_signal.decimate(sig, factor, zero_phase=True)


class CombinedFrameDataset(Dataset):
    """Eager-load combined train/val from multiple datasets.

    `sources` is a list of strings: 'ludb_train', 'ludb_val', 'qtdb', 'isp_train', 'isp_test'.
    """

    LEAD_TO_ID = {lead: i for i, lead in enumerate(ludb.LEADS_12)}

    def __init__(self, sources: list[str]):
        self.items = []           # list of (source, key) for debugging
        self.cache = []           # list of (sig_250, lead_idx, labels)

        for src in sources:
            if src == "ludb_train":
                self._load_ludb(ludb.load_split()["train"])
            elif src == "ludb_val":
                self._load_ludb(ludb.load_split()["val"])
            elif src == "qtdb":
                self._load_qtdb()
            elif src == "isp_train":
                self._load_isp("train")
            elif src == "isp_test":
                self._load_isp("test")
            else:
                raise ValueError(f"unknown source: {src}")

    def _add(self, sig_250, lead_idx, labels, src_key):
        self.cache.append((sig_250, lead_idx, labels))
        self.items.append(src_key)

    def _load_ludb(self, record_ids):
        for rid in record_ids:
            try:
                record = ludb.load_record(rid)
            except Exception:
                continue
            for lead_idx, lead in enumerate(ludb.LEADS_12):
                sig_500 = record[lead]
                sig_250 = _decimate_to_250(sig_500, 500)
                sig_n = _normalize(sig_250)
                try:
                    gt_ann = ludb.load_annotations(rid, lead)
                except Exception:
                    continue
                # Truncate or pad to WINDOW_SAMPLES
                if len(sig_n) >= WINDOW_SAMPLES:
                    sig_n = sig_n[:WINDOW_SAMPLES]
                else:
                    continue
                labels = ee.gt_to_super_frames(
                    gt_ann, n_samples=len(sig_500), fs=500, frame_ms=FRAME_MS
                ).astype(np.int64)
                if len(labels) >= WINDOW_FRAMES:
                    labels = labels[:WINDOW_FRAMES]
                else:
                    continue
                self._add(sig_n, lead_idx, labels, ("ludb", rid, lead))

    def _load_qtdb(self):
        n_loaded = 0
        n_skipped = 0
        for rid in qtdb.records_with_q1c():
            try:
                record = qtdb.load_record(rid)
                ann = qtdb.load_q1c(rid)
            except Exception:
                continue
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=250)
            if win is None:
                continue
            start, end = win
            if end > 225000:
                end = 225000
                start = end - WINDOW_SAMPLES

            # Build per-frame labels for the window (independent of lead).
            win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
            n_samples = WINDOW_SAMPLES
            sample_labels = np.full(n_samples, ee.SUPER_OTHER, dtype=np.uint8)
            for on, off in zip(win_ann["p_on"], win_ann["p_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_P
            for on, off in zip(win_ann["qrs_on"], win_ann["qrs_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_QRS
            for on, off in zip(win_ann["t_on"], win_ann["t_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_T
            samples_per_frame = WINDOW_SAMPLES // WINDOW_FRAMES   # =5
            labels = np.zeros(WINDOW_FRAMES, dtype=np.int64)
            for f in range(WINDOW_FRAMES):
                seg = sample_labels[f * samples_per_frame:(f + 1) * samples_per_frame]
                vals, counts = np.unique(seg, return_counts=True)
                labels[f] = int(vals[np.argmax(counts)])

            # Use ALL leads in the record that map cleanly to a LUDB lead-id.
            # QTDB records typically have 2 leads (e.g., MLII + V5).
            for lead_name in record.keys():
                if lead_name not in QTDB_LEAD_TO_LUDB_ID:
                    n_skipped += 1
                    continue
                lead_idx = QTDB_LEAD_TO_LUDB_ID[lead_name]
                sig = record[lead_name][start:end]
                if len(sig) < WINDOW_SAMPLES:
                    continue
                sig_n = _normalize(sig)
                self._add(sig_n, lead_idx, labels.copy(), ("qtdb", rid, lead_name))
                n_loaded += 1
        print(f"QTDB: loaded {n_loaded} sequences (skipped {n_skipped} with unmappable leads)")

    def _load_isp(self, split: str):
        rec_ids = isp.load_split()[split]
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split=split)
                ann_super = isp.load_annotations_as_super(rid, split=split)
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = record[lead]
                sig_250 = _decimate_to_250(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) >= WINDOW_SAMPLES:
                    sig_n = sig_n[:WINDOW_SAMPLES]
                else:
                    continue
                labels = ee.gt_to_super_frames(
                    ann_super, n_samples=len(sig_1000), fs=1000, frame_ms=FRAME_MS
                ).astype(np.int64)
                if len(labels) >= WINDOW_FRAMES:
                    labels = labels[:WINDOW_FRAMES]
                else:
                    continue
                self._add(sig_n, lead_idx, labels, ("isp", rid, lead))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        sig, lead_idx, labels = self.cache[idx]
        return (
            torch.from_numpy(sig),
            torch.tensor(lead_idx, dtype=torch.long),
            torch.from_numpy(labels),
        )

    def label_counts(self):
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for (_, _, labels) in self.cache:
            for c in range(N_CLASSES):
                counts[c] += int((labels == c).sum())
        return counts

    def source_counts(self):
        from collections import Counter
        return Counter(item[0] for item in self.items)


class CombinedFrameDatasetAugmented(CombinedFrameDataset):
    """Same as CombinedFrameDataset but applies ECG-specific signal-domain augmentation.

    Augmentations are pure signal-domain (powerline / sine / white noise / amplitude scale)
    so labels are unchanged. Per SemiSegECG (arXiv 2507.18323), we avoid horizontal flip
    and baseline shift since they harm delineation models.
    """

    def __init__(self, sources, n_ops=2, seed=42):
        super().__init__(sources)
        self.n_ops = n_ops
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        from ecgcode.stage2.augment import randaugment_ecg
        sig, lead_idx, labels = super().__getitem__(idx)
        sig_np = sig.numpy()
        sig_np = randaugment_ecg(sig_np, fs=250, n_ops=self.n_ops, rng=self.rng)
        return (
            torch.from_numpy(sig_np.astype(np.float32)),
            lead_idx,
            labels,
        )
