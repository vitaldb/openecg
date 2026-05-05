# ecgcode/ludb.py
"""LUDB (Lobachevsky University Database) loader and stratified split.

Expects ECGCODE_LUDB_ZIP env var pointing to the LUDB zip file.
Extracts to ECGCODE_LUDB_CACHE (default: ~/.cache/ecgcode/ludb).
"""

import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import wfdb

LEADS_12 = ("i", "ii", "iii", "avr", "avl", "avf",
            "v1", "v2", "v3", "v4", "v5", "v6")

LUDB_INNER_DIR = "lobachevsky-university-electrocardiography-database-1.0.1"


def _zip_path() -> Path:
    p = os.environ.get("ECGCODE_LUDB_ZIP")
    if not p:
        raise FileNotFoundError(
            "Set ECGCODE_LUDB_ZIP env var to LUDB zip file path. "
            "Download from https://physionet.org/content/ludb/1.0.1/"
        )
    return Path(p)


def _cache_path() -> Path:
    p = os.environ.get("ECGCODE_LUDB_CACHE")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".cache" / "ecgcode" / "ludb"


def ensure_extracted() -> Path:
    """Extract LUDB zip to cache (idempotent). Returns the inner data dir."""
    cache = _cache_path()
    inner = cache / LUDB_INNER_DIR
    if inner.exists():
        return inner
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_zip_path()) as z:
        z.extractall(cache)
    return inner


def all_record_ids() -> list[int]:
    """Return all 200 LUDB record IDs (1..200)."""
    inner = ensure_extracted()
    records_file = inner / "RECORDS"
    ids = []
    for line in records_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(int(Path(line).name))
    return sorted(ids)


def load_record(record_id: int) -> dict[str, np.ndarray]:
    """Load one LUDB record. Returns dict {lead_name: signal[5000]}.

    Signal shape: 5000 samples = 10s @ 500Hz, mV.
    """
    inner = ensure_extracted()
    record_path = str(inner / "data" / str(record_id))
    record = wfdb.rdrecord(record_path)
    return {lead: record.p_signal[:, i].astype(np.float64)
            for i, lead in enumerate(LEADS_12)}


def load_annotations(record_id: int, lead: str) -> dict[str, list[int]]:
    """Load LUDB cardiologist annotations for one record-lead.

    Returns dict with keys 'p_on', 'p_peak', 'p_off', 'qrs_on', 'qrs_peak',
    'qrs_off', 't_on', 't_peak', 't_off' (sample indices).
    """
    inner = ensure_extracted()
    ann_path = str(inner / "data" / str(record_id))
    ann = wfdb.rdann(ann_path, lead)
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


def labeled_range(record_id: int, lead: str) -> tuple[int, int] | None:
    """Return (first_sample, last_sample) at native 500Hz spanning all labeled
    boundaries for one record-lead, or None if no annotations.

    LUDB cardiologists skip the first/last partial beats (typical unlabeled
    edges: ~1.4s start, ~1.3s end out of 5000-sample 10s window). Use this
    range to mask edge frames during training and to scope the eval region so
    correctly detected but unlabeled edge beats don't penalize PPV.
    """
    ann = load_annotations(record_id, lead)
    all_pos = []
    for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
        all_pos.extend(ann.get(k, []))
    if not all_pos:
        return None
    return (int(min(all_pos)), int(max(all_pos)))


def load_metadata() -> list[dict]:
    """Read ludb.csv. Returns list of dicts with normalized rhythm field.

    Multi-line Rhythms (with embedded newlines) are normalized to first line.
    """
    inner = ensure_extracted()
    csv_path = inner / "ludb.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        rhythm_raw = r.get("Rhythms", "").strip()
        r["rhythm"] = rhythm_raw.split("\n")[0].strip()
        r["pacemaker"] = bool(r.get("Cardiac pacing", "").strip())
        r["id_int"] = int(r["ID"])
    return rows


def stratified_split(seed: int = 42, val_frac: float = 0.2) -> dict[str, list[int]]:
    """Rhythm-stratified record-level train/val split.

    Each rhythm class is split independently to preserve class balance.
    Reproducible via numpy default_rng(seed).
    """
    meta = load_metadata()
    by_rhythm: dict[str, list[int]] = defaultdict(list)
    for r in meta:
        by_rhythm[r["rhythm"]].append(r["id_int"])

    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    for rhythm, ids in sorted(by_rhythm.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n_val = round(len(ids_sorted) * val_frac)
        val_ids.extend(ids_sorted[:n_val])
        train_ids.extend(ids_sorted[n_val:])

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "seed": seed,
        "val_frac": val_frac,
    }


def save_split_json(out_path: Path | str = "data/splits/ludb_v1.json", seed: int = 42):
    """Generate stratified split and save to JSON for reproducibility lock-in."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split = stratified_split(seed=seed)
    out_path.write_text(json.dumps(split, indent=2))
    return split


def load_split(path: Path | str = "data/splits/ludb_v1.json") -> dict[str, list[int]]:
    """Load committed split JSON."""
    return json.loads(Path(path).read_text())
