"""WFDB rhythm-segment loaders for the rare-arrhythmia source DBs
(VFDB, CUDB, SDDB). Each record has rhythm-change annotations of
the form ``(XXX`` at sample N; we walk the annotations to slice
the record into segments, then window each segment to 10 s @ 250 Hz
and assign a class_id via ``WFDB_TAG_TO_CLASS``.

All three DBs are 250 Hz @ 2-channel (or 3-channel) ECG; we take the
first channel and z-norm per window.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from openecg.stage2.arrhythmia_classes import (
    CLASS_NSR, CLASS_AFIB, CLASS_AFLUTTER,
    CLASS_SINUS_BRADY, CLASS_SINUS_TACHY,
    CLASS_AVB_HIGH, CLASS_PACED, CLASS_VPC,
    CLASS_VT_MONO, CLASS_VT_POLY, CLASS_VF,
    CLASS_SVT, CLASS_OTHER, N_CLASSES,
)


FS_TARGET = 250
WINDOW_SAMPLES = 2500


# Rhythm-tag → class_id. Tags are the strings inside parentheses in
# WFDB aux_note (e.g. "(VT" → "VT"). Tags not in this map are skipped.
WFDB_TAG_TO_CLASS = {
    # Sinus
    "N":     CLASS_NSR,
    "NSR":   CLASS_NSR,
    "SBR":   CLASS_SINUS_BRADY,
    "ST":    CLASS_SINUS_TACHY,
    # Junctional / nodal
    "NOD":   CLASS_SINUS_BRADY,   # nodal as slow non-sinus regular
    # Atrial
    "AFIB":  CLASS_AFIB,
    "AF":    CLASS_AFIB,
    "AFL":   CLASS_AFLUTTER,
    "SVTA":  CLASS_SVT,
    "SVT":   CLASS_SVT,
    # Conduction / block
    "BII":   CLASS_AVB_HIGH,      # 2° block
    "BIII":  CLASS_AVB_HIGH,      # 3° block
    "AB":    CLASS_VPC,           # atrial bigeminy → treat as ectopy
    "B":     CLASS_VPC,           # bigeminy (often ventricular)
    "BI":    CLASS_VPC,           # bigeminy abbrev
    "T":     CLASS_VPC,           # trigeminy (rare)
    # Premature/ectopic
    "VPC":   CLASS_VPC,
    "PVC":   CLASS_VPC,
    "HGEA":  CLASS_VPC,           # high-grade ectopy
    # Ventricular tach/fib
    "VT":    CLASS_VT_MONO,
    "VFL":   CLASS_VT_POLY,       # ventricular flutter = polymorphic
    "VFIB":  CLASS_VF,
    "VF":    CLASS_VF,
    "VER":   CLASS_VT_MONO,       # ventricular escape rhythm
    # Paced
    "P":     CLASS_PACED,
    "PM":    CLASS_PACED,
    # Skip
    "NOISE": None,
    "ASYS":  None,                # asystole — no signal, skip
}


def _parse_rhythm_segments(rec_base: str) -> List[Tuple[int, int, str]]:
    """Parse a WFDB record's atr annotation file into rhythm segments.
    Returns list of (start_sample, end_sample, rhythm_tag)."""
    import wfdb
    ann = wfdb.rdann(rec_base, "atr")
    rhythm_changes = []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note or []):
        aux_clean = (aux or "").rstrip("\x00")
        if aux_clean.startswith("(") and len(aux_clean) > 1:
            tag = aux_clean[1:]
            rhythm_changes.append((int(s), tag))
    # End sample
    sig_len = None
    try:
        rec = wfdb.rdheader(rec_base)
        sig_len = int(rec.sig_len)
    except Exception:
        pass
    # Pair consecutive (start, next_start) → segment of one rhythm
    segments = []
    for i, (s, tag) in enumerate(rhythm_changes):
        end = (rhythm_changes[i + 1][0] if i + 1 < len(rhythm_changes)
               else (sig_len if sig_len is not None else s + WINDOW_SAMPLES))
        if end > s + 100:  # at least 100 samples (0.4 s @ 250Hz)
            segments.append((s, end, tag))
    return segments


def _load_record_channel(rec_base: str, ch: int = 0) -> Tuple[np.ndarray, int]:
    """Load one channel of a WFDB record + native sample rate."""
    import wfdb
    rec = wfdb.rdrecord(rec_base)
    sig = rec.p_signal[:, ch].astype(np.float64)
    return sig, int(rec.fs)


class WFDBRareDataset(Dataset):
    """Generic rhythm-segment dataset for VFDB/CUDB/SDDB records.

    ``record_dir`` should contain ``*.dat / *.hea / *.atr`` triplets.
    For each record:
      1. Parse rhythm-change annotations into segments
      2. For each segment with a recognised class_id, slice into
         non-overlapping 10 s windows (at native fs, then decimate to
         250 Hz if needed)
      3. z-norm each window
      4. Yield (signal[1, 2500], class_id)

    All accepted windows are cached in RAM at init time — these DBs
    are small (<50 MB each).
    """

    def __init__(self, record_dir: Path | str):
        record_dir = Path(record_dir)
        atrs = sorted(glob.glob(str(record_dir / "*.atr")))
        if not atrs:
            raise FileNotFoundError(
                f"No *.atr files found under {record_dir}; "
                f"download the PhysioNet DB first."
            )

        self.windows: list[tuple[np.ndarray, int]] = []
        self._record_dir = record_dir
        skipped_records = 0
        for atr in atrs:
            base = atr[:-4]
            try:
                segments = _parse_rhythm_segments(base)
                if not segments:
                    skipped_records += 1
                    continue
                sig, fs = _load_record_channel(base, ch=0)
                # Decimate to 250 Hz if needed
                if fs != FS_TARGET:
                    factor = int(fs // FS_TARGET)
                    if factor > 1:
                        sig = scipy_signal.decimate(sig, factor, zero_phase=True)
                    else:
                        sig = scipy_signal.resample(
                            sig, int(round(len(sig) * FS_TARGET / fs))
                        )
                # Adjust segment indices to target fs
                scale = FS_TARGET / fs
                for s, e, tag in segments:
                    cid = WFDB_TAG_TO_CLASS.get(tag)
                    if cid is None:
                        continue
                    s_t = int(s * scale)
                    e_t = int(e * scale)
                    pos = s_t
                    while pos + WINDOW_SAMPLES <= e_t and pos + WINDOW_SAMPLES <= len(sig):
                        w = sig[pos:pos + WINDOW_SAMPLES]
                        # Drop windows that are all-NaN, all-zero, or have
                        # vanishing std (asystole / noise / corrupted).
                        if (not np.isfinite(w).all()
                                or float(np.nanstd(w)) < 1e-3):
                            pos += WINDOW_SAMPLES
                            continue
                        m, std = float(w.mean()), float(w.std()) + 1e-6
                        w = ((w - m) / std).astype(np.float32)
                        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                        w = np.clip(w, -10.0, 10.0)
                        self.windows.append((w, cid))
                        pos += WINDOW_SAMPLES
            except Exception as e:
                print(f"  WARN: failed {base}: {e}")
                skipped_records += 1
        print(f"WFDBRareDataset[{record_dir.name}]: "
              f"{len(self.windows)} windows from "
              f"{len(atrs) - skipped_records}/{len(atrs)} records "
              f"(skipped {skipped_records})")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w, c = self.windows[idx]
        return torch.from_numpy(w).unsqueeze(0), c

    @property
    def items(self):
        # Compatible with MultiSourceClassConditional's introspection
        return [(i, c) for i, (_, c) in enumerate(self.windows)]

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for _, c in self.windows:
            counts[c] += 1
        return counts


__all__ = ["WFDBRareDataset", "WFDB_TAG_TO_CLASS"]
