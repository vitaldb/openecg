# openecg/stage2/synarrdb_dataset.py
"""Torch Dataset reader for the synarrdb on-disk dataset.

Synarrdb (https://github.com/vitaldb/synarrdb) is the upstream
synthetic arrhythmia ECG dataset that supersedes the openecg in-process
``SyntheticAVBDataset``. It ships as a pair of files produced by
``python -m synarrdb.build``:

  * ``synarrdb_500hz.npz``  — int16 ``signals[N, 5000]`` (mV·1000) and
    fixed-width unicode ``keys[N]``.
  * ``synarrdb.duckdb``     — per-window metadata (rhythm + AV ratio +
    template kind + …), sample-level on/off boundary lists, the 80/10/10
    stratified ``split`` column, and a 5000-byte uint8 ``frame_labels``
    BLOB per row whose values match openecg.eval.SUPER_* exactly.

This Dataset materialises a single split (train / val / test) from
both files into RAM and yields the same
``(signal[2500] float32, lead_id long, frame_labels[500] int64)``
tuple as ``LUDBFrameDataset`` and ``SyntheticAVBDataset`` so it
composes into ConcatDataset for v16/v18/v30c training without any
adapter layer.

The signal pipeline at ``__getitem__`` mirrors ``LUDBFrameDataset``:

  1. ``int16 / 1000.0`` → mV.
  2. ``scipy.signal.decimate(_, 2, zero_phase=True)`` → 250 Hz.
  3. z-normalisation to mean=0 / std=1.

The frame-label pipeline aggregates the 5000-sample uint8 array to
500 frames (one frame = 20 ms = 10 samples at 500 Hz) by majority
vote per 10-sample window. Class IDs already align with
openecg.eval.SUPER_OTHER / SUPER_P / SUPER_QRS / SUPER_T /
SUPER_PACED_QRS so no remapping is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from openecg import eval as ee
from openecg import ludb

LEAD_NAME_TO_ID = {name: i for i, name in enumerate(ludb.LEADS_12)}

FS_NATIVE = 500
FS_INPUT = 250
WINDOW_SAMPLES = 2500     # 10 s @ 250 Hz
WINDOW_FRAMES = 500       # 20 ms frames at 250 Hz
SAMPLES_PER_FRAME_500HZ = 10   # 1 frame = 20 ms = 10 samples @ 500 Hz
N_CLASSES = 5


def _aggregate_frames(frame_labels_5000: np.ndarray) -> np.ndarray:
    """Reduce a 5000-sample uint8 frame array to 500 frames via the
    mode (most common class) within each 10-sample window.

    The mode is preferred over a simple majority because PACED_QRS
    frames in a single 10-sample window can co-occur with QRS frames
    from the previous beat at very high heart rates (vt_poly), and a
    "majority" tiebreak would hide those. ``np.unique`` + argmax of
    counts implements the mode unambiguously.
    """
    out = np.zeros(WINDOW_FRAMES, dtype=np.int64)
    arr = frame_labels_5000.reshape(WINDOW_FRAMES, SAMPLES_PER_FRAME_500HZ)
    for f in range(WINDOW_FRAMES):
        seg = arr[f]
        vals, counts = np.unique(seg, return_counts=True)
        out[f] = int(vals[np.argmax(counts)])
    return out


class SynarrdbDataset(Dataset):
    """Materialised reader for one split of a synarrdb on-disk build.

    Args:
        npz_path: path to ``synarrdb_500hz.npz``.
        duckdb_path: path to ``synarrdb.duckdb``.
        split: one of ``"train"`` / ``"val"`` / ``"test"``.
        leads: optional filter on the ``lead`` column. Defaults to
            ``("ii",)`` which is synarrdb v1's only lead.
        scenarios: optional filter on the ``scenario`` column (e.g.
            ``("nsr", "vt_mono")`` for an ablation).
        max_windows: cap the number of windows kept (useful for fast
            sanity runs / unit tests). ``None`` keeps all of the split.

    Memory budget at full size: 56 000 train windows × 5 000 samples
    × 2 bytes (signals) ≈ 560 MB, plus the 5 000-byte frame_labels
    array per window ≈ 280 MB → ~840 MB total. Comparable to the
    Lydus loader.
    """

    def __init__(
        self,
        npz_path: str | Path,
        duckdb_path: str | Path,
        split: str = "train",
        leads: tuple[str, ...] = ("ii",),
        scenarios: tuple[str, ...] | None = None,
        max_windows: int | None = None,
    ):
        import duckdb

        npz_path = Path(npz_path)
        duckdb_path = Path(duckdb_path)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        # --- Load the npz signals and key index ---
        # Memory-map the npz: even at 700 MB it loads quickly and the
        # subsequent integer-indexing copies only the rows we need.
        npz = np.load(npz_path, mmap_mode="r")
        signals_all = npz["signals"]            # int16[N, 5000], memory-mapped
        keys_all = npz["keys"]                  # str[N]
        key_to_idx = {str(k): i for i, k in enumerate(keys_all)}

        # --- Pull metadata + frame BLOBs for the requested split ---
        con = duckdb.connect(str(duckdb_path), read_only=True)
        clauses = [f"split = '{split}'"]
        if leads:
            lead_list = ", ".join(f"'{l}'" for l in leads)
            clauses.append(f"lead IN ({lead_list})")
        if scenarios:
            scen_list = ", ".join(f"'{s}'" for s in scenarios)
            clauses.append(f"scenario IN ({scen_list})")
        where = " AND ".join(clauses)
        sql = (
            f"SELECT key, lead, is_wide_qrs, frame_labels "
            f"FROM windows WHERE {where} ORDER BY key"
        )
        if max_windows is not None:
            sql += f" LIMIT {int(max_windows)}"
        rows = con.execute(sql).fetchall()
        con.close()

        if not rows:
            raise RuntimeError(
                f"synarrdb: no rows for split={split!r} leads={leads} "
                f"scenarios={scenarios}"
            )

        # --- Materialise each row into RAM ---
        # We copy the signal slice out of the memory map so the npz
        # file handle can be closed and GC'd; the ~840 MB total fits
        # comfortably in RAM.
        # Detect native fs from npz signal width — slim builds ship 250 Hz
        # (2500 samples) signals so kgpu uploads stay <1 GiB.
        self._sig_width = int(signals_all.shape[-1])
        if self._sig_width not in (5000, 2500):
            raise ValueError(
                f"synarrdb npz unsupported window width {self._sig_width}; "
                f"expected 5000 (500 Hz) or 2500 (250 Hz)"
            )
        n = len(rows)
        sig_buf = np.empty((n, self._sig_width), dtype=np.int16)
        frame_buf = np.empty((n, 5000), dtype=np.uint8)
        lead_ids = np.empty(n, dtype=np.int64)
        for i, (key, lead, _is_wide, blob) in enumerate(rows):
            row_idx = key_to_idx.get(str(key))
            if row_idx is None:
                raise RuntimeError(f"synarrdb: key {key!r} missing from npz")
            sig_buf[i] = signals_all[row_idx]
            frame_buf[i] = np.frombuffer(blob, dtype=np.uint8, count=5000)
            lead_ids[i] = LEAD_NAME_TO_ID.get(lead, -1)

        if (lead_ids < 0).any():
            unknown = {rows[i][1] for i in np.where(lead_ids < 0)[0]}
            raise ValueError(
                f"synarrdb: leads {unknown} not in openecg.ludb.LEADS_12"
            )

        self.signals = sig_buf
        self.frames = frame_buf
        self.lead_ids = lead_ids
        self.split = split
        self.leads_filter = tuple(leads)
        self.scenarios_filter = tuple(scenarios) if scenarios else None

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        sig_native = self.signals[idx].astype(np.float32) / 1000.0   # mV
        # Slim builds already store at 250 Hz (2500 samples) — skip decimate.
        if self._sig_width == 5000:
            sig_250 = scipy_signal.decimate(sig_native, 2, zero_phase=True)
        else:
            sig_250 = sig_native
        mean = float(sig_250.mean())
        std = float(sig_250.std()) + 1e-6
        sig_250 = ((sig_250 - mean) / std).astype(np.float32)
        labels = _aggregate_frames(self.frames[idx])
        return (
            torch.from_numpy(sig_250),
            torch.tensor(int(self.lead_ids[idx]), dtype=torch.long),
            torch.from_numpy(labels),
        )

    def label_counts(self) -> np.ndarray:
        """Per-class frame counts across the materialised split.

        Used by ``stage2.train`` to compute class-balanced loss
        weights. Computed exactly (no sampling) since the frames are
        already in RAM.
        """
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        # The frames buffer is at 500 Hz (5000 samples). The model
        # operates on 500 frames, but the per-class ratios are the
        # same at either resolution, so we count over the raw buffer.
        for c in range(N_CLASSES):
            counts[c] = int((self.frames == c).sum())
        return np.maximum(counts, 1)
