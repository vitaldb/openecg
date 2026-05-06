# openecg/isp.py
"""ISP ECG delineation dataset loader.

Source: https://zenodo.org/records/14679837 (475 records, 12-lead, ~10s @ 1000Hz,
2-cardiologist annotations of P/QRS/T onset+offset).

Target format: list of (class_id, onset_sample, offset_sample) where class 0=P, 1=QRS, 2=T.
"""

import csv
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import wfdb

ISP_INNER_DIR = "isp_delineation_dataset"
LEADS_12 = ("i", "ii", "iii", "avr", "avl", "avf",
            "v1", "v2", "v3", "v4", "v5", "v6")
FS_NATIVE = 1000

_TUPLE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")


def _zip_path() -> Path:
    p = os.environ.get("OPENECG_ISP_ZIP")
    if not p:
        raise FileNotFoundError("Set OPENECG_ISP_ZIP env var")
    return Path(p)


def _cache_path() -> Path:
    p = os.environ.get("OPENECG_ISP_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "openecg" / "isp"


def ensure_extracted() -> Path:
    cache = _cache_path()
    inner = cache / ISP_INNER_DIR
    if inner.exists():
        return inner
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_zip_path()) as z:
        z.extractall(cache)
    return inner


def _parse_target(s: str) -> list[tuple[int, int, int]]:
    """Parse target string like '[(0, 100, 150), (1, 200, 250), ...]' safely via regex."""
    return [(int(a), int(b), int(c)) for a, b, c in _TUPLE_RE.findall(s)]


def _load_csv(path: Path) -> dict[int, list[tuple[int, int, int]]]:
    """Returns {file_name (int): list of (class, onset, offset)}."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            fid = int(row["file_name"])
            out[fid] = _parse_target(row["target"])
    return out


def load_split() -> dict[str, list[int]]:
    """ISP's predefined train/test split. Returns {'train': [...], 'test': [...]}."""
    inner = ensure_extracted()
    train_ann = _load_csv(inner / "train_isp_delineation_data.csv")
    test_ann = _load_csv(inner / "test_isp_delineation_data.csv")
    return {"train": sorted(train_ann.keys()), "test": sorted(test_ann.keys())}


def _load_annotations(record_id: int, split: str) -> list[tuple[int, int, int]]:
    """split: 'train' or 'test'. Returns annotation list."""
    inner = ensure_extracted()
    csv_path = inner / f"{split}_isp_delineation_data.csv"
    ann = _load_csv(csv_path)
    return ann.get(record_id, [])


def load_record(record_id: int, split: str = "train") -> dict[str, np.ndarray]:
    """Load 12-lead ECG. Returns {lead_name: signal[9998 or so]} at 1000Hz."""
    inner = ensure_extracted()
    record_path = str(inner / f"{split}_data" / str(record_id))
    record = wfdb.rdrecord(record_path)
    return {lead: record.p_signal[:, i].astype(np.float64)
            for i, lead in enumerate(LEADS_12)}


def load_annotations_as_super(record_id: int, split: str = "train") -> dict[str, list[int]]:
    """Convert ISP annotation tuples to LUDB-style dict for use with gt_to_super_frames.
    Maps class 0->P, 1->QRS, 2->T."""
    raw = _load_annotations(record_id, split)
    out = {"p_on": [], "p_off": [],
           "qrs_on": [], "qrs_off": [],
           "t_on": [], "t_off": [],
           "p_peak": [], "qrs_peak": [], "t_peak": []}  # peaks unused but expected
    for cls, on, off in raw:
        if cls == 0:
            out["p_on"].append(on)
            out["p_off"].append(off)
        elif cls == 1:
            out["qrs_on"].append(on)
            out["qrs_off"].append(off)
        elif cls == 2:
            out["t_on"].append(on)
            out["t_off"].append(off)
    return out
