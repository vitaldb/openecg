"""PTB-XL loader (Wagner et al. 2020) — 21,799 12-lead 10-s clinical ECGs.

Source: https://physionet.org/content/ptb-xl/1.0.3/

This loader exists to validate ``openecg.pacer.detect_pace`` on a
clinical-scale paced-vs-sinus cohort:

  * 294 records labelled with SCP-ECG code ``PACE`` (paced rhythm).
  * 18,054 records labelled with ``NORM`` or ``SR`` and NOT ``PACE``
    (normal sinus rhythm controls).

Records ship at both 100 Hz (``records100/``) and 500 Hz
(``records500/``); we use the 500 Hz files since that matches the LUDB
defaults the pacer detector was tuned on.

Zip discovery follows ``openecg.butpdb``: ``OPENECG_PTBXL_ZIP`` env →
``$OPENECG_DATASETS_DIR/ptb-xl-a-large-…-1.0.3.zip`` (defaults to
``G:/Shared drives/datasets/ecg``). No auto-download — the zip is 2.6 GB
and is expected to already be on disk for this validation use-case.
"""

from __future__ import annotations

import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import wfdb

INNER_DIR = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ZIP_FILENAME = f"{INNER_DIR}.zip"
DEFAULT_DATASETS_DIR = Path(r"G:\Shared drives\datasets\ecg")

# 12-lead names in PTB-XL .dat files (per the PhysioNet header).
LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6")

_SCP_KEY_RE = re.compile(r"'([A-Z0-9_]+)'\s*:")


def _datasets_dir() -> Path:
    p = os.environ.get("OPENECG_DATASETS_DIR")
    return Path(p).expanduser() if p else DEFAULT_DATASETS_DIR


def _zip_path() -> Path:
    env = os.environ.get("OPENECG_PTBXL_ZIP")
    if env:
        return Path(env)
    return _datasets_dir() / ZIP_FILENAME


def _cache_path() -> Path:
    p = os.environ.get("OPENECG_PTBXL_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "openecg" / "ptbxl"


@lru_cache(maxsize=1)
def _zip() -> zipfile.ZipFile:
    """Singleton zip handle; the .zip is read directly without bulk extraction
    to avoid materialising 2.6 GB on local disk for a sample-level use-case."""
    return zipfile.ZipFile(_zip_path())


def _scp_keys(s: object) -> set[str]:
    """Parse the SCP-ECG keys out of ``scp_codes`` rows, which are stored as
    Python-dict-style strings like ``"{'PACE': 100.0, 'NORM': 50.0}"``.
    Uses a regex over the keys (no eval / literal_eval needed)."""
    if not isinstance(s, str):
        return set()
    return set(_SCP_KEY_RE.findall(s))


@lru_cache(maxsize=1)
def metadata() -> pd.DataFrame:
    """Return the full ptbxl_database.csv with derived columns:
        is_paced  : True iff scp_codes contains 'PACE'
        is_sinus  : True iff scp_codes contains 'NORM' or 'SR' AND not 'PACE'
    """
    name = f"{INNER_DIR}/ptbxl_database.csv"
    with _zip().open(name) as f:
        df = pd.read_csv(f)
    codes = df["scp_codes"].apply(_scp_keys)
    df["is_paced"] = codes.apply(lambda k: "PACE" in k)
    df["is_sinus"] = codes.apply(
        lambda k: ("NORM" in k or "SR" in k) and "PACE" not in k
    )
    return df


def paced_ids() -> list[int]:
    """ECG IDs of all paced records (SCP code 'PACE'). 294 records."""
    return metadata().loc[lambda d: d["is_paced"], "ecg_id"].astype(int).tolist()


def sinus_ids() -> list[int]:
    """ECG IDs of all clean sinus controls (NORM or SR, NOT PACE).
    ~18 K records."""
    return metadata().loc[lambda d: d["is_sinus"], "ecg_id"].astype(int).tolist()


def load_record(ecg_id: int, fs: int = 500) -> dict:
    """Load one PTB-XL record at ``fs`` Hz (100 or 500). Returns
        {"fs": int, "leads": list[str], "signal": (n_samples, 12) float32}.

    Decompresses the WFDB .dat + .hea files from the zip into a cache dir;
    subsequent calls reuse the cached files.
    """
    if fs not in (100, 500):
        raise ValueError(f"PTB-XL is stored only at 100 or 500 Hz, got {fs}")
    suffix = "lr" if fs == 100 else "hr"
    sub = f"{(ecg_id // 1000) * 1000:05d}"
    rel = f"records{fs}/{sub}/{ecg_id:05d}_{suffix}"
    cache = _cache_path() / rel
    cache.parent.mkdir(parents=True, exist_ok=True)
    z = _zip()
    for ext in (".dat", ".hea"):
        out = cache.with_suffix(ext)
        if not out.exists():
            with z.open(f"{INNER_DIR}/{rel}{ext}") as src, out.open("wb") as dst:
                dst.write(src.read())
    rec = wfdb.rdrecord(str(cache))
    return {
        "fs": int(rec.fs),
        "leads": [str(s) for s in rec.sig_name],
        "signal": rec.p_signal.astype(np.float32),
    }


def iter_records(ecg_ids: list[int], fs: int = 500) -> Iterator[dict]:
    """Yield records by id, lazily decompressing each one."""
    for rid in ecg_ids:
        try:
            yield {"ecg_id": rid, **load_record(rid, fs=fs)}
        except Exception as exc:                             # pragma: no cover
            print(f"[ptbxl] skipping {rid}: {exc}")
