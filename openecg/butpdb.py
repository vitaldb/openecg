# openecg/butpdb.py
"""BUT PDB (Brno University of Technology ECG Signal Database with Annotations
of P Wave) loader.

Source: https://physionet.org/content/but-pdb/1.0.0/  (50 records x 2 min,
2-lead, manual P-peak + QRS annotations by 2-expert consensus).

Records 1-38 are 360 Hz from MIT-BIH Arrhythmia; records 39-50 are 128 Hz
(SupraventricularArr or Long-Term AF). All are 2-lead. Provides P peak (no
on/off boundaries) and QRS annotations.

Pathology coverage (per the original README):
  BI   (1st-degree AV block):  record 22
  BII  (2nd-degree AV block):  records 1, 13
  BIII (3rd-degree AV block):  record 3
  AFIB, AFL, V, R, L, J, NOD, ...  see README for full list.

Set OPENECG_BUTPDB_ZIP to the dataset zip path. Cache extracts to
OPENECG_BUTPDB_CACHE (default: ~/.cache/openecg/butpdb).
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import wfdb

INNER_DIR = (
    "brno-university-of-technology-ecg-signal-database-with-annotations-"
    "of-p-wave-but-pdb-1.0.0"
)

# Pathology -> list of record IDs (parsed from README; canonical map for the
# AV-block cohort that motivates loading this dataset).
PATHOLOGY_RECORDS: dict[str, tuple[int, ...]] = {
    "BI":   (22,),
    "BII":  (1, 13),
    "BIII": (3,),
}

AVB_RECORDS: tuple[int, ...] = tuple(
    sorted(set(rid for rids in PATHOLOGY_RECORDS.values() for rid in rids))
)


def _zip_path() -> Path:
    p = os.environ.get("OPENECG_BUTPDB_ZIP")
    if not p:
        raise FileNotFoundError(
            "Set OPENECG_BUTPDB_ZIP env var to BUT PDB zip path "
            "(download: https://physionet.org/content/but-pdb/1.0.0/)"
        )
    return Path(p)


def _cache_path() -> Path:
    p = os.environ.get("OPENECG_BUTPDB_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "openecg" / "butpdb"


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
    """Return all 50 BUT PDB record IDs (1..50)."""
    inner = ensure_extracted()
    out = []
    for line in (inner / "RECORDS").read_text().splitlines():
        line = line.strip()
        if line:
            out.append(int(line))
    return sorted(out)


def _record_path(record_id: int) -> str:
    """WFDB-style path prefix (no extension) for a record."""
    inner = ensure_extracted()
    return str(inner / f"{record_id:02d}")


def load_record(record_id: int) -> dict:
    """Load one BUT PDB record. Returns
        {"fs": int, "leads": [name, name], "signal": (n_samples, 2) float32}.
    Records 1-38 are 360 Hz, 39-50 are 128 Hz. All are 2-lead; lead names are
    taken from the WFDB header (typically MLII / V1 / V2 / V5).
    """
    rec = wfdb.rdrecord(_record_path(record_id))
    return {
        "fs": int(rec.fs),
        "leads": [str(s) for s in rec.sig_name],
        "signal": rec.p_signal.astype(np.float32),
    }


def load_pwave_peaks(record_id: int) -> np.ndarray:
    """P-wave peak sample indices (single 1D array)."""
    ann = wfdb.rdann(_record_path(record_id), "pwave")
    return np.asarray(ann.sample, dtype=np.int64)


def load_qrs(record_id: int) -> dict[str, np.ndarray]:
    """QRS annotations. Returns {'sample': int64[N], 'symbol': list[str]}.

    Symbols mostly follow MIT-BIH conventions ('N', 'V', 'A', 'L', 'R', ...).
    """
    ann = wfdb.rdann(_record_path(record_id), "qrs")
    return {
        "sample": np.asarray(ann.sample, dtype=np.int64),
        "symbol": list(ann.symbol),
    }


def parse_pathology(record_id: int) -> tuple[str, ...]:
    """Pathology codes for a record (from header comments). Returns a tuple
    of upper-case codes; empty tuple if header has no comment line."""
    rec = wfdb.rdrecord(_record_path(record_id))
    if not rec.comments:
        return ()
    text = " ".join(rec.comments)
    # Codes are typically a comma-separated list inside quotes in the README.
    # Header comments tend to be looser; just split on non-word and uppercase.
    tokens = [t for t in re.split(r"[^A-Za-z]+", text) if t]
    return tuple(t.upper() for t in tokens)


def records_with_avb() -> tuple[int, ...]:
    """The 4 AV-block records (BI=22, BII={1,13}, BIII=3). The motivating
    cohort for using BUT PDB in this project."""
    return AVB_RECORDS
