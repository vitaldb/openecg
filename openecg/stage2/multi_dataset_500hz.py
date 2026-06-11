"""500 Hz native CombinedFrameDataset — for the v57c+ foundation model.

Source-by-source fs handling:
  * LUDB  — native 500 Hz, used as-is.
  * QTDB  — native 250 Hz, upsampled to 500 Hz via polyphase (×2).
  * ISP   — native 1000 Hz, decimated to 500 Hz (×½).

Frame grid stays at 50 Hz (frame_ms = 20) — same as the 250 Hz path —
so the model's per-patch output still spans 20 ms. The only change vs
``multi_dataset.py`` is the input sample rate (and therefore window
size and per-patch sample count).

Constants:
  * ``WINDOW_SAMPLES = 5000``   (10 s @ 500 Hz)
  * ``WINDOW_FRAMES  = 500``    (unchanged — same 50 Hz grid)
  * Caller-side ``patch_size = 10`` (vs 5 for the 250 Hz path) — keeps
    n_patches = 500.

This module is a parallel implementation, not a replacement: the
existing 250 Hz pipeline (multi_dataset.py) remains for v56c/v56d/v57a
backward compatibility.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from openecg import eval as ee
from openecg import isp, ludb, qtdb


FRAME_MS = 20
WINDOW_SAMPLES = 5000           # 10 s @ 500 Hz
WINDOW_FRAMES = 500             # 50 Hz frame grid
TARGET_FS = 500
N_CLASSES = 4                    # other / P / QRS / T

QTDB_LEAD_TO_LUDB_ID = {
    # mirror multi_dataset.py's mapping; lead-name normalised to lower.
    "MLII": 1, "II": 1, "ECG1": 0, "ECG2": 1, "V1": 6, "V2": 7,
    "V4": 9, "V5": 10,
}


def _normalize(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


def _resample_to_500(sig: np.ndarray, fs_native: int) -> np.ndarray:
    """Bring a 1-D signal to 500 Hz exactly via polyphase resampling."""
    if fs_native == 500:
        return sig.astype(np.float64)
    if fs_native == 1000:
        # ½ decimation — zero-phase Chebyshev like the 250 Hz path uses.
        return scipy_signal.decimate(sig, 2, zero_phase=True)
    if fs_native == 250:
        return scipy_signal.resample_poly(sig, up=2, down=1).astype(np.float64)
    if fs_native > 500 and fs_native % 500 == 0:
        return scipy_signal.decimate(sig, fs_native // 500, zero_phase=True)
    # Arbitrary fs — generic polyphase
    from math import gcd
    g = gcd(int(fs_native), 500)
    return scipy_signal.resample_poly(sig, up=500 // g,
                                       down=int(fs_native) // g
                                       ).astype(np.float64)


class CombinedFrameDataset500Hz(Dataset):
    """500 Hz native combined train/val from LUDB / QTDB / ISP.

    Source strings accepted: ``ludb_train``, ``ludb_val``, ``qtdb``,
    ``isp_train``, ``isp_test``. Identical interface to
    :class:`openecg.stage2.multi_dataset.CombinedFrameDataset` but with
    500 Hz signals.
    """

    def __init__(self, sources: list[str], qtdb_q1c_pu_merge: bool = True,
                 qtdb_min_anns_per_window: int = 4):
        self._qtdb_q1c_pu_merge = qtdb_q1c_pu_merge
        self._qtdb_min_anns_per_window = qtdb_min_anns_per_window
        self.items: list = []
        self.cache: list[tuple[np.ndarray, int, np.ndarray]] = []

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

    def _add(self, sig, lead_idx, labels, src_key):
        self.cache.append((sig, lead_idx, labels))
        self.items.append(src_key)

    def _load_ludb(self, record_ids):
        n_ok = 0
        for rid in record_ids:
            try:
                record = ludb.load_record(rid)            # native 500 Hz
            except Exception:
                continue
            for lead_idx, lead in enumerate(ludb.LEADS_12):
                sig_500 = np.asarray(record[lead])
                try:
                    gt_ann = ludb.load_annotations(rid, lead)
                except Exception:
                    continue
                if len(sig_500) >= WINDOW_SAMPLES:
                    sig_500 = sig_500[:WINDOW_SAMPLES]
                else:
                    continue
                sig_n = _normalize(sig_500)
                labels = ee.gt_to_super_frames(
                    gt_ann, n_samples=len(sig_500), fs=500, frame_ms=FRAME_MS
                ).astype(np.int64)
                if len(labels) >= WINDOW_FRAMES:
                    labels = labels[:WINDOW_FRAMES]
                else:
                    continue
                self._add(sig_n, lead_idx, labels, ("ludb", rid, lead))
                n_ok += 1
        print(f"LUDB-500Hz: loaded {n_ok} sequences")

    def _load_qtdb(self):
        n_loaded = 0; n_skipped = 0; n_sparse = 0
        for rid in qtdb.records_with_q1c():
            try:
                record = qtdb.load_record(rid)            # native 250 Hz
                if self._qtdb_q1c_pu_merge:
                    ann_250 = qtdb.load_q1c_pu_merged(rid, pu_lead=0)
                else:
                    ann_250 = qtdb.load_q1c(rid)
            except Exception:
                continue
            # Scale annotation positions to 500 Hz.
            ann = {k: [int(s * 2) for s in v] for k, v in ann_250.items()}
            full_len_500 = (len(next(iter(record.values()))) * 2)
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=500)
            if win is None:
                continue
            start, end = win
            # Clamp to record length (in 500 Hz samples; 225000 samples @ 250 Hz
            # => 450000 @ 500 Hz).
            cap = full_len_500
            if end > cap:
                end = cap
                start = end - WINDOW_SAMPLES
            n_in_win = sum(1 for k in ("p_on","p_off","qrs_on","qrs_off","t_on","t_off")
                           for s in ann.get(k, []) if start <= s < end)
            if n_in_win < self._qtdb_min_anns_per_window:
                n_sparse += 1
                continue
            win_ann = {k: [s - start for s in v if start <= s < end]
                       for k, v in ann.items()}
            n_samples = WINDOW_SAMPLES
            sample_labels = np.full(n_samples, ee.SUPER_OTHER, dtype=np.uint8)
            for on, off in zip(win_ann["p_on"], win_ann["p_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_P
            for on, off in zip(win_ann["qrs_on"], win_ann["qrs_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_QRS
            for on, off in zip(win_ann["t_on"], win_ann["t_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_T
            samples_per_frame = WINDOW_SAMPLES // WINDOW_FRAMES   # = 10
            labels = np.zeros(WINDOW_FRAMES, dtype=np.int64)
            for f in range(WINDOW_FRAMES):
                seg = sample_labels[f * samples_per_frame:(f + 1) * samples_per_frame]
                vals, counts = np.unique(seg, return_counts=True)
                labels[f] = int(vals[np.argmax(counts)])

            for lead_name in record.keys():
                if lead_name not in QTDB_LEAD_TO_LUDB_ID:
                    n_skipped += 1
                    continue
                lead_idx = QTDB_LEAD_TO_LUDB_ID[lead_name]
                sig_250 = np.asarray(record[lead_name])
                sig_500 = _resample_to_500(sig_250, fs_native=250)
                seg = sig_500[start:end]
                if len(seg) < WINDOW_SAMPLES:
                    continue
                sig_n = _normalize(seg[:WINDOW_SAMPLES])
                self._add(sig_n, lead_idx, labels.copy(),
                          ("qtdb", rid, lead_name))
                n_loaded += 1
        print(f"QTDB-500Hz: loaded {n_loaded} sequences  (skipped {n_skipped} "
              f"unmappable, {n_sparse} sparse-window)")

    def _load_isp(self, split: str):
        rec_ids = isp.load_split()[split]
        n_ok = 0
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split=split)    # native 1000 Hz
                ann_super = isp.load_annotations_as_super(rid, split=split)
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = np.asarray(record[lead])
                sig_500 = _resample_to_500(sig_1000, fs_native=1000)
                sig_n = _normalize(sig_500)
                # Pad / truncate to WINDOW_SAMPLES.
                if len(sig_n) >= WINDOW_SAMPLES:
                    sig_n = sig_n[:WINDOW_SAMPLES]
                else:
                    pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                labels = ee.gt_to_super_frames(
                    ann_super, n_samples=len(sig_1000), fs=1000, frame_ms=FRAME_MS
                ).astype(np.int64)
                if len(labels) >= WINDOW_FRAMES:
                    labels = labels[:WINDOW_FRAMES]
                else:
                    pad = np.full(WINDOW_FRAMES - len(labels), ee.SUPER_OTHER,
                                  dtype=labels.dtype)
                    labels = np.concatenate([labels, pad])
                self._add(sig_n, lead_idx, labels, ("isp", rid, lead))
                n_ok += 1
        print(f"ISP-500Hz ({split}): loaded {n_ok} sequences")

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


SAMPLES_PER_FRAME = WINDOW_SAMPLES // WINDOW_FRAMES   # = 10

LUDB_LEAD_NAMES = ("i", "ii", "iii", "avr", "avl", "avf",
                    "v1", "v2", "v3", "v4", "v5", "v6")
LEAD_NAME_TO_ID = {name: i for i, name in enumerate(LUDB_LEAD_NAMES)}


def _synth_labels_to_frame_array_500hz(labels: dict, paced_pattern: bool
                                        ) -> np.ndarray:
    """500 Hz analogue of synth_dataset._labels_to_frame_array.

    ``labels`` is the dict returned by
    ``openecg.synth.generate_avb_window`` at fs=500 — keys
    ``p_on / p_off / qrs_on / qrs_off / t_on / t_off`` are lists of
    sample indices (already in 500-Hz coordinates). Returns the
    per-frame label array at 50 Hz (WINDOW_FRAMES = 500) via
    majority-of-10-samples voting.
    """
    sample_labels = np.full(WINDOW_SAMPLES, ee.SUPER_OTHER, dtype=np.uint8)
    qrs_class = ee.SUPER_PACED_QRS if paced_pattern else ee.SUPER_QRS
    # Same overwrite order as synth_dataset._labels_to_frame_array
    # (T first, P second, QRS last so QRS wins over P at overlap and
    # P wins over T).
    for cls_id, on_key, off_key in (
        (ee.SUPER_T,   "t_on",   "t_off"),
        (ee.SUPER_P,   "p_on",   "p_off"),
        (qrs_class,    "qrs_on", "qrs_off"),
    ):
        for on, off in zip(labels.get(on_key, []), labels.get(off_key, [])):
            lo = max(0, int(on))
            hi = min(WINDOW_SAMPLES, int(off) + 1)
            if hi > lo:
                sample_labels[lo:hi] = cls_id

    frame_labels = np.zeros(WINDOW_FRAMES, dtype=np.int64)
    for f in range(WINDOW_FRAMES):
        seg = sample_labels[f * SAMPLES_PER_FRAME:(f + 1) * SAMPLES_PER_FRAME]
        vals, counts = np.unique(seg, return_counts=True)
        frame_labels[f] = int(vals[np.argmax(counts)])
    return frame_labels


class SyntheticAVBDataset500Hz(Dataset):
    """500 Hz version of :class:`openecg.stage2.synth_dataset.SyntheticAVBDataset`.

    Constructor signature matches the 250 Hz original; the only
    behavioural change is fs=500 throughout (the underlying TemplateBank
    must also be built at fs=500 — pass one via ``bank``).
    """

    def __init__(self, bank, leads=("ii", "v1", "i", "v5", "v2"),
                 scenarios=("mobitz1", "mobitz2", "complete", "paced"),
                 n_windows: int = 2000, base_seed: int | None = 12345):
        from openecg import synth
        self._synth = synth
        self.bank = bank
        self.leads = tuple(leads)
        self.scenarios = tuple(scenarios)
        self.n_windows = int(n_windows)
        self.base_seed = base_seed

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if self.base_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.base_seed + idx)
        scenario = self.scenarios[idx % len(self.scenarios)]
        lead = self.leads[(idx // len(self.scenarios)) % len(self.leads)]
        sig, labels, meta = self._synth.generate_avb_window(
            self.bank, lead, scenario, rng,
            fs=TARGET_FS, duration_s=10.0,
        )
        frame_labels = _synth_labels_to_frame_array_500hz(
            labels, paced_pattern=meta["is_wide_paced_pattern"],
        )
        return (
            torch.from_numpy(sig.astype(np.float32)),
            torch.tensor(LEAD_NAME_TO_ID[lead], dtype=torch.long),
            torch.from_numpy(frame_labels),
        )

    def label_counts(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        counts = np.zeros(N_CLASSES + 1, dtype=np.int64)   # +1 for paced_QRS
        n_sample = 100
        for k in range(n_sample):
            scenario = self.scenarios[k % len(self.scenarios)]
            lead = self.leads[(k // len(self.scenarios)) % len(self.leads)]
            _, labels, meta = self._synth.generate_avb_window(
                self.bank, lead, scenario, rng,
                fs=TARGET_FS, duration_s=10.0,
            )
            frames = _synth_labels_to_frame_array_500hz(
                labels, paced_pattern=meta["is_wide_paced_pattern"],
            )
            for c in range(counts.size):
                counts[c] += int((frames == c).sum())
        scale = self.n_windows / max(1, n_sample)
        return np.maximum((counts * scale).astype(np.int64), 1)


def _synth_labels_to_sample_array_500hz(labels: dict, paced_pattern: bool
                                         ) -> np.ndarray:
    """Per-sample (500 Hz, 5000-long) SUPER label array for one synth window.

    Same overwrite order as :func:`_synth_labels_to_frame_array_500hz`
    (T, P, QRS) but returns the full-resolution sample array instead of
    the 50 Hz frame downsample — for sample-resolution training.
    """
    sample_labels = np.full(WINDOW_SAMPLES, ee.SUPER_OTHER, dtype=np.uint8)
    qrs_class = ee.SUPER_PACED_QRS if paced_pattern else ee.SUPER_QRS
    for cls_id, on_key, off_key in (
        (ee.SUPER_T,   "t_on",   "t_off"),
        (ee.SUPER_P,   "p_on",   "p_off"),
        (qrs_class,    "qrs_on", "qrs_off"),
    ):
        for on, off in zip(labels.get(on_key, []), labels.get(off_key, [])):
            lo = max(0, int(on))
            hi = min(WINDOW_SAMPLES, int(off) + 1)
            if hi > lo:
                sample_labels[lo:hi] = cls_id
    return sample_labels


class SyntheticAVBSampleDataset500Hz(SyntheticAVBDataset500Hz):
    """Sample-resolution (500 Hz) variant of :class:`SyntheticAVBDataset500Hz`.

    Returns ``(sig[5000], lead_id, sample_labels[5000])`` — labels are
    per-sample so the per-sample softmax head trains against the synth's
    own native 500 Hz boundaries (no 50 Hz frame quantisation).
    """

    def __getitem__(self, idx):
        if self.base_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.base_seed + idx)
        scenario = self.scenarios[idx % len(self.scenarios)]
        lead = self.leads[(idx // len(self.scenarios)) % len(self.leads)]
        sig, labels, meta = self._synth.generate_avb_window(
            self.bank, lead, scenario, rng, fs=TARGET_FS, duration_s=10.0,
        )
        sample_labels = _synth_labels_to_sample_array_500hz(
            labels, paced_pattern=meta["is_wide_paced_pattern"],
        )
        return (
            torch.from_numpy(sig.astype(np.float32)),
            torch.tensor(LEAD_NAME_TO_ID[lead], dtype=torch.long),
            torch.from_numpy(sample_labels.astype(np.int64)),
        )


class CachedSampleDataset500Hz(Dataset):
    """Load a precomputed 500 Hz sample-resolution cache (.npz).

    The cache is built locally by ``scripts/build_500hz_cache.py`` and
    shipped to the training pod. Each item is
    ``(sig[5000] float32, lead_id long, sample_labels[5000] int64)`` —
    note the labels are **per-sample** (500 Hz), not per-frame, so the
    per-sample softmax head trains against native-resolution boundaries.

    Signals are z-normalised in the cache; wrap with the training-side
    rank-normaliser (rank_normalize is order-based, so z-norm-then-rank
    == rank, preserving the v56c input contract).
    """

    def __init__(self, npz_path: str):
        blob = np.load(npz_path)
        self.signals = blob["signals"]      # (N, 5000) float32
        self.lead_ids = blob["lead_ids"]    # (N,) int16
        self.labels = blob["labels"]        # (N, 5000) uint8
        assert self.signals.shape[1] == WINDOW_SAMPLES, \
            f"expected {WINDOW_SAMPLES}-sample signals, got {self.signals.shape}"

    def __len__(self):
        return int(self.signals.shape[0])

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx].astype(np.float32)),
            torch.tensor(int(self.lead_ids[idx]), dtype=torch.long),
            torch.from_numpy(self.labels[idx].astype(np.int64)),
        )

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for c in range(N_CLASSES):
            counts[c] = int((self.labels == c).sum())
        return counts


__all__ = [
    "CombinedFrameDataset500Hz", "SyntheticAVBDataset500Hz",
    "SyntheticAVBSampleDataset500Hz", "CachedSampleDataset500Hz",
    "WINDOW_SAMPLES", "WINDOW_FRAMES", "FRAME_MS", "TARGET_FS",
    "SAMPLES_PER_FRAME", "LEAD_NAME_TO_ID",
    "_resample_to_500",
]
