# openecg/lydus.py
"""SNUH Lydus ECG loader (167K windows × 8-lead × 10s, native 500 Hz).

Provides a numpy-memmap signal store + a duckdb metadata table with rich
clinically-curated labels (rhythm, bbb, avb, premature_beat, qrsd, vrate,
arate, pri, axes, ...). Used as window-level multi-task supervision for
the v17 hierarchical model — the dataset has no frame-level wave
boundaries but provides window-level labels for "is RR regular" /
"is QRS wide" / rhythm class / AVB grade, all of which are the next layer
of supervision above the frame head.

Set OPENECG_LYDUS_DIR to the directory containing
  lydus_ecg.npz         (167199, 40000)   int16   8 leads × 5000 samples (10s @ 500Hz)
                        flattened lead-major: vals[r, l*5000:(l+1)*5000] = lead l
  lydus_ecg.duckdb      records table

The npz signal store is memory-mapped on first access; only the rows
referenced via `iter_windows` are read into RAM, so the whole 4 GB
file does not need to fit in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy.signal as scipy_signal

# Best-guess channel ordering for the 8-lead SNUH Lydus npz. The first three
# limb leads match SNUH's `learn_rhythm.py` (NCH=3 = lead I-III); the next five
# are precordial V1-V5 by amplitude characteristics. aVR/aVL/aVF/V6 are absent
# from this 8-lead montage. If a record clearly diverges from this layout the
# downstream model should fall back to the lead-agnostic (no lead_emb) path.
LEADS_8: tuple[str, ...] = ("i", "ii", "iii", "v1", "v2", "v3", "v4", "v5")

# Subset of leads that are also covered by the LUDB / ISP frame-labeled
# datasets, so the lead-embedding distribution stays consistent at training
# time. Lead III is technically present but the model has weak lead-III
# performance (axis-orthogonal P/T), so we keep it out of the v17 default mix.
SAFE_LEADS: tuple[str, ...] = ("i", "ii", "v1", "v2", "v5")

# QRS width threshold for the binary "wide QRS" task. 120 ms is the standard
# clinical cutoff (BBB / aberrancy boundary).
QRS_WIDE_MS = 120

# Rhythm categories whose RR pattern is intrinsically regular. Anything not in
# either set is left as `None` and masked out of the rr_regular loss.
RHYTHM_REGULAR: frozenset[str] = frozenset({
    "NSR", "S.brady", "S.tachy", "Junctional rhythm", "Possible pacing",
})
RHYTHM_IRREGULAR: frozenset[str] = frozenset({
    "A.fib", "A.flutter", "S.arrhythmia", "SVT", "Variable AVB",
})

# AVB grade (4-class). The Variable AVB / AV dissociation cohort that
# co-occurs with A.flutter is folded into 2°AVB for training stability;
# isolated AV dissociation is rare (2 cases) and folds into 3°AVB.
AVB_GRADE_MAP: dict[str, int] = {
    "NSP": 0,
    "1'AVB": 1,           # duckdb spelling
    "1AVB": 1,            # csv-era spelling
    "2'AVB": 2,
    "2AVB": 2,
    "Variable AVB": 2,    # clinically a variable Mobitz pattern
    "IVB": 2,             # incomplete VB grouped with 2°
    "3'AVB": 3,
    "3AVB": 3,
    "AV dissociation": 3,
}
N_AVB_GRADES = 4

# Rhythm 3-class. The 14 raw labels collapse cleanly into sinus / AF / other.
RHYTHM_CLASS_MAP: dict[str, int] = {
    "NSR": 0, "S.brady": 0, "S.tachy": 0, "S.arrhythmia": 0,
    "Other sinus rhythm": 0, "Junctional rhythm": 0,
    "Possible pacing": 0, "Nonspecific": 0,
    "A.fib": 1, "A.flutter": 1,
    "Others": 2, "SVT": 2, "Undetermined": 2, "WPW": 2,
}
N_RHYTHM_CLASSES = 3


FS_NATIVE = 500
N_LEADS = 8
N_SAMPLES_PER_LEAD = 5000                     # 10 s @ 500 Hz
N_SAMPLES_PER_ROW = N_LEADS * N_SAMPLES_PER_LEAD     # 40000


def _root() -> Path:
    p = os.environ.get("OPENECG_LYDUS_DIR")
    if not p:
        raise FileNotFoundError(
            "Set OPENECG_LYDUS_DIR env var to the lydus_ecg directory "
            "(should contain lydus_ecg.npz and lydus_ecg.duckdb)"
        )
    return Path(p)


@dataclass
class LydusWindow:
    """One window's metadata + lazy signal access. The signal array is
    materialized on `.signal_at_fs(fs)` and not stored on the dataclass."""
    npz_idx: int
    rid: int
    key: str
    rhythm: str
    bbb: str
    avb: str
    qrsd_ms: int          # measured QRS duration in milliseconds
    vrate_bpm: int        # measured ventricular rate
    arate_bpm: int        # measured atrial rate
    pri_ms: int | None    # PR interval; None when unmeasurable

    # Derived window-level labels (computed in load_window).
    qrs_wide: int                    # 0/1
    rr_regular: int | None           # 0/1, None when ambiguous (mask out)
    rhythm_class: int | None         # 0/1/2 sinus/AF/other
    avb_grade: int | None            # 0/1/2/3 none/1°/2°/3°


def _connect():
    """Open the duckdb in read-only mode."""
    import duckdb
    return duckdb.connect(str(_root() / "lydus_ecg.duckdb"), read_only=True)


# Process-global signal-store cache. The npz is ~4 GB at 500 Hz, so re-opening
# it on every __getitem__ is catastrophically slow on any non-local filesystem
# (Google Drive Stream's virtual FS in particular). We cache the (keys, vals)
# handle once.
_SIGNAL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def prewarm_signals(fs: int = FS_NATIVE, *, into_ram: bool = False) -> None:
    """Open the npz signal store ahead of time so the first __getitem__
    isn't penalised. With `into_ram=True`, copies the entire `vals` array
    into RAM (~4 GB) — recommended when the npz lives on Google Drive's
    virtual FS or any other slow random-access medium.

    ``fs`` must be ``FS_NATIVE`` (=500); resampling to a downstream target
    happens in :func:`load_signal`.
    """
    if fs in _SIGNAL_CACHE:
        return
    if fs != FS_NATIVE:
        raise ValueError(
            f"Lydus npz is now stored only at {FS_NATIVE} Hz; "
            f"resample inside load_signal() instead. Got fs={fs}."
        )
    path = _root() / "lydus_ecg.npz"
    if not path.exists():
        # Backward compat: older mirrors might still keep the suffix variant.
        legacy = _root() / "lydus_ecg_500hz.npz"
        if legacy.exists():
            path = legacy
        else:
            raise FileNotFoundError(
                f"Expected {path}; legacy {legacy} also missing."
            )
    data = np.load(path, mmap_mode="r")
    keys = np.asarray(data["keys"])
    vals = data["vals"]
    if vals.shape[1] != N_SAMPLES_PER_ROW:
        raise ValueError(
            f"Lydus vals row width must be {N_SAMPLES_PER_ROW} "
            f"(={N_LEADS}×{N_SAMPLES_PER_LEAD}); got {vals.shape}."
        )
    if into_ram:
        vals = np.ascontiguousarray(vals)
    _SIGNAL_CACHE[fs] = (keys, vals)


def _signal_store(fs: int = FS_NATIVE):
    """Return the cached (keys, vals) signal store, opening it if needed."""
    if fs not in _SIGNAL_CACHE:
        prewarm_signals(fs, into_ram=False)
    return _SIGNAL_CACHE[fs]


def _derive_labels(row) -> tuple[int, int | None, int | None, int | None]:
    """Return (qrs_wide, rr_regular, rhythm_class, avb_grade).

    `None` entries are masked out of the loss. Numeric labels (qrs_wide)
    are always available because qrsd has 100% coverage in the duckdb.
    """
    qrs_wide = 1 if int(row["qrsd"]) >= QRS_WIDE_MS else 0
    rh = str(row["rhythm"])
    if rh in RHYTHM_REGULAR:
        rr_regular = 1
    elif rh in RHYTHM_IRREGULAR:
        rr_regular = 0
    else:
        rr_regular = None
    rhythm_class = RHYTHM_CLASS_MAP.get(rh)
    avb_grade = AVB_GRADE_MAP.get(str(row["avb"]))
    return qrs_wide, rr_regular, rhythm_class, avb_grade


def load_metadata(only_with_labels: bool = True) -> "list[LydusWindow]":
    """Load all 167K rows. With `only_with_labels=True` (default), windows
    where every aux signal is masked are dropped — keeping ~99% of records.
    """
    con = _connect()
    rows = con.execute("""
        SELECT npz_idx, rid, rhythm, bbb, avb, qrsd, vrate, arate, pri
        FROM records
        WHERE npz_idx IS NOT NULL AND qrsd IS NOT NULL
        ORDER BY npz_idx
    """).fetchall()
    keys, _ = _signal_store(FS_NATIVE)
    out: list[LydusWindow] = []
    for npz_idx, rid, rhythm, bbb, avb, qrsd, vrate, arate, pri in rows:
        qrs_wide, rr_regular, rhythm_class, avb_grade = _derive_labels(
            {"qrsd": qrsd, "rhythm": rhythm, "avb": avb}
        )
        if only_with_labels and rr_regular is None and rhythm_class is None \
                and avb_grade is None:
            continue
        out.append(LydusWindow(
            npz_idx=int(npz_idx), rid=int(rid),
            key=str(keys[int(npz_idx)]),
            rhythm=str(rhythm), bbb=str(bbb), avb=str(avb),
            qrsd_ms=int(qrsd), vrate_bpm=int(vrate),
            arate_bpm=int(arate), pri_ms=int(pri) if pri is not None else None,
            qrs_wide=qrs_wide, rr_regular=rr_regular,
            rhythm_class=rhythm_class, avb_grade=avb_grade,
        ))
    return out


def load_signal(npz_idx: int, lead_idx: int, fs_target: int = 250) -> np.ndarray:
    """Load one (npz_idx, lead) signal at native ``FS_NATIVE`` (=500 Hz) and
    resample to ``fs_target``. Returns a float32 array of length
    ``int(round(fs_target * 10))``.

    The npz is stored as flattened (N_records, 8 × 5000) rows; this slices
    the lead's contiguous 5000-sample block out of the flat row.
    """
    if not (0 <= int(lead_idx) < N_LEADS):
        raise ValueError(f"lead_idx must be in [0, {N_LEADS}); got {lead_idx}")
    keys, vals = _signal_store(FS_NATIVE)
    lo = int(lead_idx) * N_SAMPLES_PER_LEAD
    hi = lo + N_SAMPLES_PER_LEAD
    sig_native = vals[int(npz_idx), lo:hi].astype(np.float32)
    if fs_target == FS_NATIVE:
        return sig_native
    if fs_target * 2 == FS_NATIVE:
        # 500 → 250: cheap zero-phase decimation.
        return scipy_signal.decimate(sig_native, 2, zero_phase=True).astype(np.float32)
    n_target = int(round(len(sig_native) * fs_target / FS_NATIVE))
    return scipy_signal.resample(sig_native, n_target).astype(np.float32)


def lead_id_for(channel_idx: int) -> int:
    """Map a Lydus npz channel index (0..7) to the LUDB 12-lead `lead_id` used
    by the model's lead embedding. Returns -1 if the channel falls outside
    the safe set (channel 2 = lead III) — caller should skip those leads
    or use a non-lead-emb model.
    """
    from openecg import ludb
    name = LEADS_8[channel_idx]
    if name not in ludb.LEADS_12:
        return -1
    return ludb.LEADS_12.index(name)


def split_records(seed: int = 42, val_frac: float = 0.05) -> dict[str, list[int]]:
    """Patient-disjoint train/val split on record id. The `rid` column is the
    duckdb primary key; record ids are assigned per visit, so the same
    patient-day appears once. We split on rid for simplicity (true
    patient-level split would require the original SNUH index).
    """
    con = _connect()
    rids = sorted(int(r[0]) for r in con.execute(
        "SELECT DISTINCT rid FROM records WHERE npz_idx IS NOT NULL "
        "AND qrsd IS NOT NULL").fetchall())
    rng = np.random.default_rng(seed)
    rng.shuffle(rids)
    n_val = max(1, int(len(rids) * val_frac))
    return {"val": rids[:n_val], "train": rids[n_val:]}
