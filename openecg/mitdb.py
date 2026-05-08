# openecg/mitdb.py
"""MIT-BIH Arrhythmia Database loader.

Source: https://physionet.org/content/mitdb/1.0.0/  (Moody & Mark, 2001 —
*the* canonical R-peak / beat-classification benchmark; 48 records ×
30 min × 2-lead @ 360 Hz, manually annotated by 2 cardiologists).

We use this for QRS-detector validation (``scripts/validate_qrs_mitdb.py``)
since beat annotations are gold-standard and tolerance conventions are
fixed by AAMI EC57 (100 ms).

Zip discovery follows ``openecg.butpdb``:
  1. OPENECG_MITDB_ZIP env (explicit path).
  2. <OPENECG_DATASETS_DIR or G:/Shared drives/datasets/ecg>/
     mit-bih-arrhythmia-database-1.0.0.zip
  3. Download from PhysioNet, cached into the dataset folder above.

Cache extracts to OPENECG_MITDB_CACHE (default: ~/.cache/openecg/mitdb).
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import wfdb

INNER_DIR = "mit-bih-arrhythmia-database-1.0.0"
ZIP_FILENAME = f"{INNER_DIR}.zip"

PHYSIONET_URL = (
    "https://physionet.org/static/published-projects/mitdb/"
    f"{ZIP_FILENAME}"
)

DEFAULT_DATASETS_DIR = Path(r"G:\Shared drives\datasets\ecg")

# AAMI EC57-recommended evaluation set. Records 102, 104, 107, 217 are
# excluded by AAMI because they contain extensive paced rhythm — beat
# detection on these is contaminated by pacer spikes counted as beats.
AAMI_EXCLUDED: tuple[int, ...] = (102, 104, 107, 217)

# Beat-annotation symbol set (subset). A peak is a "beat" iff its symbol
# is in this set. Non-beat annotations (e.g., '+' rhythm change, '~'
# noise, '|' isolated artifact) must be skipped before R-peak matching.
BEAT_SYMBOLS: frozenset[str] = frozenset({
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q", "?",
})


def _datasets_dir() -> Path:
    p = os.environ.get("OPENECG_DATASETS_DIR")
    return Path(p).expanduser() if p else DEFAULT_DATASETS_DIR


def _download_zip(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[mitdb] downloading {PHYSIONET_URL} -> {target}")
    urllib.request.urlretrieve(PHYSIONET_URL, tmp)
    tmp.replace(target)
    return target


def _zip_path() -> Path:
    """Locate the MIT-BIH Arrhythmia zip.

    Order: OPENECG_MITDB_ZIP env -> <datasets_dir>/<ZIP_FILENAME> ->
    download from PhysioNet into <datasets_dir>.
    """
    env = os.environ.get("OPENECG_MITDB_ZIP")
    if env:
        return Path(env)
    candidate = _datasets_dir() / ZIP_FILENAME
    if candidate.exists():
        return candidate
    return _download_zip(candidate)


def _cache_path() -> Path:
    p = os.environ.get("OPENECG_MITDB_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "openecg" / "mitdb"


def ensure_extracted() -> Path:
    cache = _cache_path()
    inner = cache / INNER_DIR
    if inner.exists():
        return inner
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_zip_path()) as z:
        z.extractall(cache)
    return inner


def all_record_ids() -> list[int]:
    """Return the canonical 48 record IDs from the MIT-BIH RECORDS file."""
    inner = ensure_extracted()
    out = []
    for line in (inner / "RECORDS").read_text().splitlines():
        line = line.strip()
        if line:
            out.append(int(line))
    return sorted(out)


def aami_record_ids() -> list[int]:
    """44 records eligible for AAMI EC57 evaluation (excludes 102, 104,
    107, 217 — paced rhythm)."""
    return [r for r in all_record_ids() if r not in AAMI_EXCLUDED]


def _record_path(record_id: int) -> str:
    """WFDB-style path prefix (no extension) for a record."""
    inner = ensure_extracted()
    return str(inner / str(int(record_id)))


def load_record(record_id: int) -> dict:
    """Load one MIT-BIH Arrhythmia record. Returns
        {"fs": int=360, "leads": [name, name],
         "signal": (n_samples, 2) float32}.

    All MIT-BIH records are 360 Hz × 2-lead × 30 min (650,000 samples).
    Lead 0 is typically MLII (limb-II); lead 1 varies (V1/V2/V4/V5).
    """
    rec = wfdb.rdrecord(_record_path(record_id))
    return {
        "fs": int(rec.fs),
        "leads": [str(s) for s in rec.sig_name],
        "signal": rec.p_signal.astype(np.float32),
    }


def load_beats(record_id: int) -> dict[str, np.ndarray | list[str]]:
    """Beat annotations from the .atr file.

    Returns:
        {"sample": int64[N] — annotation sample indices,
         "symbol": list[str] — N annotation symbols.}

    Note: the .atr file mixes BEAT annotations ('N', 'V', 'L', ...) with
    non-beat annotations ('+' rhythm change, '~' noise, '|' isolated
    artifact, etc.). For R-peak evaluation, filter to ``BEAT_SYMBOLS``
    via ``load_qrs_peaks``.
    """
    ann = wfdb.rdann(_record_path(record_id), "atr")
    return {
        "sample": np.asarray(ann.sample, dtype=np.int64),
        "symbol": list(ann.symbol),
    }


def load_qrs_peaks(record_id: int) -> np.ndarray:
    """R-peak sample indices (only true beats, filtered by ``BEAT_SYMBOLS``).

    Use this as the ground-truth for QRS detector evaluation.
    """
    beats = load_beats(record_id)
    samples = beats["sample"]
    symbols = beats["symbol"]
    keep = [s for s, sym in zip(samples, symbols) if sym in BEAT_SYMBOLS]
    return np.asarray(keep, dtype=np.int64)
