# ecgcode/qtdb.py
"""QTDB (Physionet QT Database) loader for cross-DB validation.

Expects ECGCODE_QTDB_ZIP env var pointing to QTDB zip file.
Extracts to ECGCODE_QTDB_CACHE (default: ~/.cache/ecgcode/qtdb).
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import wfdb

QTDB_INNER_DIR = "qt-database-1.0.0"


def _zip_path() -> Path:
    p = os.environ.get("ECGCODE_QTDB_ZIP")
    if not p:
        raise FileNotFoundError(
            "Set ECGCODE_QTDB_ZIP env var to QTDB zip file path. "
            "Download from https://physionet.org/content/qtdb/"
        )
    return Path(p)


def _cache_path() -> Path:
    p = os.environ.get("ECGCODE_QTDB_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "ecgcode" / "qtdb"


def ensure_extracted() -> Path:
    cache = _cache_path()
    inner = cache / QTDB_INNER_DIR
    if inner.exists():
        return inner
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_zip_path()) as z:
        z.extractall(cache)
    return inner


def all_record_ids() -> list[str]:
    """Return all QTDB record IDs (e.g., 'sel100', 'sel102', ...)."""
    inner = ensure_extracted()
    records_file = inner / "RECORDS"
    return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]


def records_with_q1c() -> list[str]:
    """Records that have a q1c (cardiologist annotator 1) annotation file."""
    inner = ensure_extracted()
    out = []
    for rid in all_record_ids():
        if (inner / f"{rid}.q1c").exists():
            out.append(rid)
    return out


def load_record(record_id: str) -> dict[str, np.ndarray]:
    """Load QTDB record. Returns {lead_name: signal[225000]} (15 min @ 250Hz)."""
    inner = ensure_extracted()
    record = wfdb.rdrecord(str(inner / record_id))
    return {
        record.sig_name[i]: record.p_signal[:, i].astype(np.float64)
        for i in range(record.n_sig)
    }


def load_q1c(record_id: str) -> dict[str, list[int]]:
    """Load q1c cardiologist annotations.
    Returns dict with keys p_on, p_peak, p_off, qrs_on, qrs_peak, qrs_off, t_on, t_peak, t_off."""
    inner = ensure_extracted()
    ann = wfdb.rdann(str(inner / record_id), "q1c")
    out = {"p_on": [], "p_peak": [], "p_off": [],
           "qrs_on": [], "qrs_peak": [], "qrs_off": [],
           "t_on": [], "t_peak": [], "t_off": []}
    for i, sym in enumerate(ann.symbol):
        s = int(ann.sample[i])
        if sym == "p":
            out["p_peak"].append(s)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["p_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["p_off"].append(int(ann.sample[i + 1]))
        elif sym == "N":
            out["qrs_peak"].append(s)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["qrs_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["qrs_off"].append(int(ann.sample[i + 1]))
        elif sym == "t":
            out["t_peak"].append(s)
            if i > 0 and ann.symbol[i - 1] == "(":
                out["t_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["t_off"].append(int(ann.sample[i + 1]))
    return out


def annotated_window(ann_dict: dict, window_samples: int = 2500, fs: int = 250) -> tuple[int, int] | None:
    """Find a 10s (2500 samples @ 250Hz) window centered on the annotated region.
    Returns (start, end) sample indices, or None if no annotations."""
    all_samples = []
    for k, v in ann_dict.items():
        all_samples.extend(v)
    if not all_samples:
        return None
    mid = (min(all_samples) + max(all_samples)) // 2
    start = max(0, mid - window_samples // 2)
    end = start + window_samples
    return (start, end)


def load_annotations_as_super(record_id: str, window: tuple[int, int] | None = None) -> dict[str, list[int]]:
    """Wrapper around load_q1c that returns LUDB-style super dict.
    If window=(start, end), only returns annotations within window (samples re-zeroed)."""
    ann = load_q1c(record_id)
    if window is None:
        return ann
    start, end = window
    out = {k: [] for k in ann.keys()}
    for k, vals in ann.items():
        for s in vals:
            if start <= s < end:
                out[k].append(s - start)
    return out
