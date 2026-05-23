"""Unified PyTorch Dataset that draws labeled (signal, class_id) windows
from multiple sources for training the class-conditional diffusion
model. Each source contributes its rule-mapped class_id (see
``arrhythmia_classes``), and a per-source sampling weight controls
the rare-class oversampling.

Sources (in priority order; rare classes get heavy oversample weight):
  * Lydus (167K, common rhythms — NSR/AFib/AVB/BBB/paced/VPC)
  * Synarrdb (70K parametric — VT_mono/VT_poly/CAVB/Mobitz/paced/afib)
  * CUDB     (real VT/VF episodes — Phase B)
  * VFDB     (real VT/VF — Phase B)
  * Long-QT DB (real LQT/TdP-prone — Phase B)

All windows are returned as (signal[1, 2500] float32 z-normed,
class_id int) at 250 Hz. Per-source class-id mapping is rule-based
on metadata, no model inference involved at dataset load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from openecg.stage2.arrhythmia_classes import (
    CLASS_NAMES, N_CLASSES, TARGET_DISTRIBUTION,
    SYNARRDB_SCENARIO_TO_CLASS,
    lydus_record_to_class,
)


FS_TARGET = 250
WINDOW_SAMPLES = 2500
LEAD_TARGET = "ii"


class LydusClassConditionalDataset(Dataset):
    """Lydus rows → (lead-II signal, class_id). Skips rows that
    ``lydus_record_to_class`` can't map (Nonspecific, Other, etc).
    """

    def __init__(self):
        import duckdb
        from openecg import lydus
        # NOTE: do not store the lydus module on self (not picklable for
        # multiprocessing DataLoader). Import lazily inside __getitem__.
        self.lead_idx = lydus.LEADS_8.index(LEAD_TARGET)
        db_path = lydus._root() / "lydus_ecg.duckdb"
        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute(
            "SELECT npz_idx, rhythm, avb, bbb, premature_beat, pacing, "
            "vrate, qrsd, qtc, dx, conclusion FROM records"
        ).fetchall()
        con.close()
        cols = ["npz_idx", "rhythm", "avb", "bbb", "premature_beat", "pacing",
                "vrate", "qrsd", "qtc", "dx", "conclusion"]
        self.items: list[tuple[int, int]] = []  # (npz_idx, class_id)
        for r in rows:
            d = dict(zip(cols, r))
            c = lydus_record_to_class(d)
            if c is not None:
                self.items.append((int(d["npz_idx"]), c))
        print(f"LydusClassConditional: {len(self.items)} labeled rows "
              f"(skipped {len(rows) - len(self.items)})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from openecg import lydus  # lazy import for worker pickling
        npz_idx, class_id = self.items[idx]
        sig_500 = lydus.load_signal(npz_idx, self.lead_idx, fs_target=500)
        sig_500 = np.nan_to_num(sig_500, nan=0.0, posinf=0.0, neginf=0.0)
        sig_250 = scipy_signal.decimate(sig_500, 2, zero_phase=True)
        sig_250 = sig_250[:WINDOW_SAMPLES]
        if len(sig_250) < WINDOW_SAMPLES:
            sig_250 = np.concatenate([
                sig_250,
                np.zeros(WINDOW_SAMPLES - len(sig_250), dtype=sig_250.dtype),
            ])
        m, s = float(sig_250.mean()), float(sig_250.std()) + 1e-6
        sig_250 = ((sig_250 - m) / s).astype(np.float32)
        sig_250 = np.nan_to_num(sig_250, nan=0.0, posinf=0.0, neginf=0.0)
        sig_250 = np.clip(sig_250, -10.0, 10.0)  # cap z-norm outliers
        return torch.from_numpy(sig_250).unsqueeze(0), class_id

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for _, c in self.items:
            counts[c] += 1
        return counts


class SynarrdbClassConditionalDataset(Dataset):
    """Synarrdb (dist_clean) rows → (decimated lead-II signal, class_id)
    via ``SYNARRDB_SCENARIO_TO_CLASS``. Reads from the same npz +
    duckdb that ``openecg.stage2.synarrdb_dataset.SynarrdbDataset``
    uses.
    """

    def __init__(self, npz_path: Path | str, duckdb_path: Path | str,
                  split: str = "train"):
        import duckdb
        npz = np.load(str(npz_path), mmap_mode="r")
        signals_all = npz["signals"]
        keys_all = npz["keys"]
        key_to_idx = {str(k): i for i, k in enumerate(keys_all)}
        con = duckdb.connect(str(duckdb_path), read_only=True)
        rows = con.execute(
            "SELECT key, scenario, lead FROM windows "
            "WHERE split = ? AND lead = ?",
            [split, LEAD_TARGET],
        ).fetchall()
        con.close()
        self.items: list[tuple[int, int]] = []
        for key, scenario, _lead in rows:
            cid = SYNARRDB_SCENARIO_TO_CLASS.get(scenario)
            if cid is None:
                continue
            row_idx = key_to_idx.get(str(key))
            if row_idx is None:
                continue
            self.items.append((int(row_idx), cid))
        self._signals_all = signals_all
        print(f"SynarrdbClassConditional[{split}]: {len(self.items)} labeled windows")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        row_idx, class_id = self.items[idx]
        sig_500 = self._signals_all[row_idx].astype(np.float32) / 1000.0  # mV
        sig_500 = np.nan_to_num(sig_500, nan=0.0, posinf=0.0, neginf=0.0)
        sig_250 = scipy_signal.decimate(sig_500, 2, zero_phase=True)
        sig_250 = sig_250[:WINDOW_SAMPLES]
        if len(sig_250) < WINDOW_SAMPLES:
            sig_250 = np.concatenate([
                sig_250,
                np.zeros(WINDOW_SAMPLES - len(sig_250), dtype=sig_250.dtype),
            ])
        m, s = float(sig_250.mean()), float(sig_250.std()) + 1e-6
        sig_250 = ((sig_250 - m) / s).astype(np.float32)
        sig_250 = np.nan_to_num(sig_250, nan=0.0, posinf=0.0, neginf=0.0)
        sig_250 = np.clip(sig_250, -10.0, 10.0)
        return torch.from_numpy(sig_250).unsqueeze(0), class_id

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for _, c in self.items:
            counts[c] += 1
        return counts


class MultiSourceClassConditional(Dataset):
    """Concatenates multiple per-source datasets. ``__getitem__`` simply
    routes to the right sub-dataset; the global sampling weight (for
    rare-class oversampling) is built by ``build_balanced_sampler``.
    """

    def __init__(self, sources: list[Dataset]):
        self.sources = sources
        offsets = [0]
        for ds in sources:
            offsets.append(offsets[-1] + len(ds))
        self._offsets = offsets
        self._total = offsets[-1]
        # Materialise the (class_id, source_idx) array for sampler weighting
        all_labels = []
        for s_idx, ds in enumerate(sources):
            for c in ds.label_counts().repeat(1).tolist():
                pass  # placeholder
            counts = ds.label_counts()
            print(f"  source {s_idx} ({type(ds).__name__}) class counts: "
                  + ", ".join(
                      f"{CLASS_NAMES[c]}={counts[c]}"
                      for c in range(N_CLASSES) if counts[c] > 0
                  ))
        # Build the per-item class array by iterating sources
        self.labels = np.zeros(self._total, dtype=np.int32)
        for s_idx, ds in enumerate(sources):
            base = offsets[s_idx]
            if hasattr(ds, "items"):
                for j, (_, c) in enumerate(ds.items):
                    self.labels[base + j] = c
            else:
                for j in range(len(ds)):
                    self.labels[base + j] = ds[j][1]

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        # Binary search for source
        for s_idx, off in enumerate(self._offsets):
            if idx < self._offsets[s_idx + 1]:
                return self.sources[s_idx][idx - off]
        raise IndexError(idx)

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for c in self.labels:
            counts[int(c)] += 1
        return counts

    def build_balanced_sampler(
        self,
        target_distribution: Optional[dict] = None,
        n_samples_per_epoch: Optional[int] = None,
    ) -> WeightedRandomSampler:
        """Return a ``WeightedRandomSampler`` whose per-class probability
        matches ``target_distribution`` (defaults to
        ``arrhythmia_classes.TARGET_DISTRIBUTION``). Each window's
        sampling weight is ``target[c] / count[c]``.
        """
        target = target_distribution or TARGET_DISTRIBUTION
        counts = self.label_counts()
        # Avoid div-by-zero for classes with no real examples
        per_class_w = np.zeros(N_CLASSES, dtype=np.float64)
        for c in range(N_CLASSES):
            if counts[c] > 0:
                per_class_w[c] = target.get(c, 0.0) / counts[c]
        weights = per_class_w[self.labels]
        if n_samples_per_epoch is None:
            n_samples_per_epoch = self._total
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=int(n_samples_per_epoch),
            replacement=True,
        )


__all__ = [
    "LydusClassConditionalDataset",
    "SynarrdbClassConditionalDataset",
    "MultiSourceClassConditional",
]
