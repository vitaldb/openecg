"""Validate ``openecg`` on the VitalDB Arrhythmia Database (PhysioNet
1.0.0, Lee et al. 2026 — https://physionet.org/content/vitaldb-arrhythmia/).

The dataset provides expert beat-type and rhythm annotations for 482
surgical-patient ECG recordings from the parent VitalDB project. The
annotation CSVs ship from PhysioNet (~25 MB total); the waveforms are
streamed from the open VitalDB API via the ``vitaldb`` Python package
(lead ``SNUADC/ECG_II``, 500 Hz).

Two validations against the official annotations:

  1. **QRS detection** — :func:`openecg.detect_qrs` vs annotated R-peak
     timestamps. AAMI EC57 matching tolerance is ±100 ms. Reports per-case
     **naive** (zero-offset) sensitivity / PPV / F1 **and** an
     **offset-corrected** number plus the best per-case time shift.

  2. **AFib detection** — :func:`openecg.is_afib` per non-overlapping
     10-s window vs ``rhythm_label == "AFIB/AFL"``. Reports sensitivity /
     specificity overall, plus FPR per non-AFib rhythm class so you can
     see exactly which rhythms confuse the detector.

Windows flagged ``bad_signal_quality == True`` (segment marker pairs
``Start1/End1``, ``Start2/End2``, …) are excluded from both metrics — the
dataset's own quality flag, not anything openecg invented.

**Dataset time-alignment caveat**: vitaldb-arrhythmia 1.0.0 ships some
cases whose annotation timeline is shifted by up to ±1.2 s relative to
the VitalDB-served waveform — empirically observed on AFIB/AFL cases
(e.g. case 1023 needs +1225 ms, case 1086 needs −350 ms). The naive
F1 reflects this shift directly (so it can underrepresent the
detector); the offset-corrected F1 sweeps a per-case shift in
``[-2 s, +2 s]`` at 25 ms resolution and reports the best match
together with the offset, so you can see both the out-of-the-box number
and the detector's true performance once the dataset's per-case
misalignment is removed.

Setup
-----

::

    # 1. Annotations from PhysioNet (~25 MB):
    wget -r -N -c -np https://physionet.org/files/vitaldb-arrhythmia/1.0.0/

    # 2. Python deps:
    pip install "openecg[loaders]" vitaldb

Usage
-----

::

    # Full validation, both tasks:
    python -m scripts.validate_vitaldb_arrhythmia \\
        --root physionet.org/files/vitaldb-arrhythmia/1.0.0

    # Subset of cases (skip the first VitalDB fetch overhead):
    python -m scripts.validate_vitaldb_arrhythmia --root <ROOT> \\
        --cases 1 2 3 5 8 --cache-dir data/vitaldb-cache

    # One task only:
    python -m scripts.validate_vitaldb_arrhythmia --root <ROOT> --task qrs
    python -m scripts.validate_vitaldb_arrhythmia --root <ROOT> --task afib

The first run downloads each case's lead-II waveform from VitalDB
(~10-20 MB per case at 500 Hz). With ``--cache-dir`` set, waveforms are
saved as ``case_<id>.npy`` and re-used on subsequent runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import detect_qrs, is_afib

# --- constants -------------------------------------------------------------

FS_VITALDB = 500            # SNUADC/ECG_II is published at 500 Hz
TRACK_NAME = "SNUADC/ECG_II"
QRS_TOL_MS = 100.0          # AAMI EC57 matching tolerance for R-peaks
WINDOW_S = 10               # AFib evaluation window length (seconds)
ANN_DIR = "Annotation_Files"
META_CSV = "metadata.csv"

# The annotation CSV uses abbreviations in ``rhythm_label`` — not the
# full names listed in the dataset description. ``AFIB/AFL`` is the
# combined Atrial Fibrillation / Atrial Flutter class used in the file.
AFIB_LABEL = "AFIB/AFL"


# --- IO helpers ------------------------------------------------------------

def _read_metadata(root: Path) -> dict[int, tuple[float, float]]:
    """Map ``case_id`` → ``(analysis_start_sec, analysis_end_sec)``.

    VitalDB's ``load_case`` returns the **entire surgical case** (often
    hours long), but the annotators only labelled a curated window of
    typically ~20 minutes per case. The metadata's
    ``analysis_start_time_sec`` / ``analysis_end_time_sec`` bounds that
    window in the VitalDB time base, which is the same time base as the
    ``time_second`` column of the annotation CSVs.
    """
    path = root / META_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — did you run "
            f"`wget -r -N -c -np https://physionet.org/files/vitaldb-arrhythmia/1.0.0/`?"
        )
    out: dict[int, tuple[float, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["case_id"])
            t0 = float(row["analysis_start_time_sec"])
            t1 = float(row["analysis_end_time_sec"])
            out[cid] = (t0, t1)
    return out


def _read_annotations(root: Path, case_id: int) -> list[dict]:
    """Read one case's annotation CSV. Each row is one annotated beat
    with R-peak time + beat / rhythm / quality labels."""
    path = root / ANN_DIR / f"Annotation_file_{case_id}.csv"
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _load_waveform(case_id: int, cache_dir: Path | None) -> np.ndarray | None:
    """Stream lead-II waveform from VitalDB (cached on disk if ``cache_dir``).

    Uses :class:`vitaldb.VitalFile` and :meth:`get_track_samples`, which
    reconstructs the waveform timeline-aware (gap-preserving) — unlike
    ``vitaldb.load_case`` whose wave-track path does naive index
    resampling. Any remaining NaN gaps are linearly interpolated so
    :func:`detect_qrs` receives a finite signal.

    Returns the float64 1-D signal at 500 Hz, or ``None`` when VitalDB has
    no usable trace for the case (all-NaN response).
    """
    if cache_dir is not None:
        cache_path = cache_dir / f"case_{case_id}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    try:
        import vitaldb
    except ImportError as e:
        raise SystemExit(
            "The `vitaldb` package is required to fetch waveforms. "
            "Install it with `pip install vitaldb` (and pass --cache-dir "
            "to avoid re-downloading on subsequent runs)."
        ) from e

    vf = vitaldb.VitalFile(case_id, [TRACK_NAME])
    samples = vf.get_track_samples(TRACK_NAME, 1.0 / FS_VITALDB)
    if samples is None or len(samples) == 0:
        return None
    sig = np.asarray(samples, dtype=np.float64)
    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return None
    if nan_mask.any():
        idx = np.arange(sig.size)
        sig = np.interp(idx, idx[~nan_mask], sig[~nan_mask])

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / f"case_{case_id}.npy", sig)
    return sig


# --- annotation parsing ----------------------------------------------------

def _bad_quality_intervals(ann_rows: list[dict]) -> list[tuple[float, float]]:
    """Recover ``[(start_s, end_s), ...]`` bad-signal-quality intervals.

    The dataset marks each bad-quality segment with paired ``StartN`` /
    ``EndN`` labels in ``bad_signal_quality_label`` on the row at the
    corresponding ``time_second``. We pair them in order; an unpaired
    ``StartN`` at the end of the file extends to the last annotated beat.
    """
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for r in ann_rows:
        lbl = (r.get("bad_signal_quality_label") or "").strip()
        if not lbl:
            continue
        try:
            t = float(r["time_second"])
        except (TypeError, ValueError):
            continue
        if lbl.startswith("Start"):
            starts[lbl[len("Start"):]] = t
        elif lbl.startswith("End"):
            key = lbl[len("End"):]
            if key in starts:
                intervals.append((starts.pop(key), t))
    intervals.sort()
    return intervals


def _samples_in_good(samples: np.ndarray,
                     bad_intervals_samples: list[tuple[int, int]]) -> np.ndarray:
    """Boolean mask: True where the sample index falls OUTSIDE every
    bad-quality interval."""
    if samples.size == 0 or not bad_intervals_samples:
        return np.ones(samples.shape, dtype=bool)
    keep = np.ones(samples.shape, dtype=bool)
    for lo, hi in bad_intervals_samples:
        keep &= (samples < lo) | (samples >= hi)
    return keep


def _window_overlaps_bad(start: int, end: int,
                         bad_intervals_samples: list[tuple[int, int]]) -> bool:
    for lo, hi in bad_intervals_samples:
        if start < hi and end > lo:
            return True
    return False


def _window_rhythm(win_start_s: float, win_end_s: float,
                   ann_rows: list[dict]) -> str | None:
    """Majority rhythm_label among beats whose R-peak falls inside
    ``[win_start_s, win_end_s)``.

    Returns ``None`` if the window has no labelled beats, or if the
    majority share is below 60 % (mixed-rhythm windows are excluded to
    avoid noisy ground truth at transition boundaries).
    """
    tally: dict[str, int] = defaultdict(int)
    n = 0
    for r in ann_rows:
        try:
            t = float(r["time_second"])
        except (TypeError, ValueError):
            continue
        if win_start_s <= t < win_end_s:
            lbl = (r.get("rhythm_label") or "").strip()
            if lbl:
                tally[lbl] += 1
                n += 1
    if n == 0:
        return None
    top_lbl, top_n = max(tally.items(), key=lambda kv: kv[1])
    if top_n / n < 0.60:
        return None
    return top_lbl


# --- QRS evaluation --------------------------------------------------------

def _greedy_match(gt: np.ndarray, det: np.ndarray, tol: int) -> int:
    """Bipartite greedy match: each detection pairs with at most one GT
    within ``tol`` samples. Sweep gt in ascending order."""
    if gt.size == 0 or det.size == 0:
        return 0
    gt_sorted = np.sort(gt)
    det_sorted = np.sort(det)
    used = np.zeros(det_sorted.size, dtype=bool)
    matched = 0
    j_start = 0
    for g in gt_sorted:
        while j_start < det_sorted.size and det_sorted[j_start] < g - tol:
            j_start += 1
        for j in range(j_start, det_sorted.size):
            d = det_sorted[j]
            if d > g + tol:
                break
            if not used[j]:
                used[j] = True
                matched += 1
                break
    return matched


# Some vitaldb-arrhythmia 1.0.0 cases have a per-case time offset between
# the annotation timeline and the VitalDB-served waveform — empirically
# observed at up to ±1.2 s on AFIB/AFL cases. The same beat sequence is
# annotated and detected, but the wall-clock alignment is off by a
# constant shift that varies case-by-case. The naive (zero-offset) match
# is a faithful out-of-the-box number; the corrected match recovers the
# detector's true performance after removing the dataset's alignment
# offset.
QRS_OFFSET_SWEEP_RANGE_MS = (-2000, 2000)
QRS_OFFSET_SWEEP_STEP_MS = 25


def _best_offset(gt_samples: np.ndarray, det: np.ndarray, fs: int,
                 tol_samples: int) -> tuple[int, int]:
    """Sweep an integer-sample shift over ``QRS_OFFSET_SWEEP_RANGE_MS``;
    return ``(best_offset_ms, best_tp)`` where ``best_tp`` is the bipartite
    match count at that offset.
    """
    lo_ms, hi_ms = QRS_OFFSET_SWEEP_RANGE_MS
    best_off_ms = 0
    best_tp = _greedy_match(gt_samples, det, tol_samples)
    for dt_ms in range(lo_ms, hi_ms + 1, QRS_OFFSET_SWEEP_STEP_MS):
        if dt_ms == 0:
            continue
        shift = int(round(dt_ms * fs / 1000.0))
        tp = _greedy_match(gt_samples + shift, det, tol_samples)
        if tp > best_tp:
            best_tp = tp
            best_off_ms = dt_ms
    return best_off_ms, best_tp


def evaluate_qrs(case_id: int, root: Path,
                 analysis_window: tuple[float, float],
                 cache_dir: Path | None) -> dict | None:
    """Run QRS detection on the analysis window only.

    ``analysis_window`` is ``(start_sec, end_sec)`` from ``metadata.csv``;
    the waveform is cropped to that span before running ``detect_qrs``
    so detections and GT live in the same time base. Reports both the
    zero-offset (naive) F1 and the offset-corrected F1 with the per-case
    offset that maximises the match — see :data:`QRS_OFFSET_SWEEP_RANGE_MS`.
    """
    sig = _load_waveform(case_id, cache_dir)
    if sig is None:
        return None
    start_s, end_s = analysis_window
    lo, hi = int(round(start_s * FS_VITALDB)), int(round(end_s * FS_VITALDB))
    lo = max(0, lo)
    hi = min(len(sig), hi)
    if hi - lo < FS_VITALDB:    # less than 1 s of data
        return None
    sig = sig[lo:hi]
    ann = _read_annotations(root, case_id)

    gt_times = np.array(
        [float(r["time_second"]) for r in ann
         if r.get("time_second") not in (None, "", "NA")],
        dtype=np.float64,
    )
    # Shift annotation times so they index into the cropped signal.
    gt_samples = np.round((gt_times - start_s) * FS_VITALDB).astype(np.int64)
    gt_samples = gt_samples[(gt_samples >= 0) & (gt_samples < len(sig))]

    bad_s = _bad_quality_intervals(ann)
    bad_samples = [(int((a - start_s) * FS_VITALDB),
                    int((b - start_s) * FS_VITALDB)) for a, b in bad_s]

    keep_gt = _samples_in_good(gt_samples, bad_samples)
    gt_samples = gt_samples[keep_gt]

    t0 = time.perf_counter()
    det = detect_qrs(sig, FS_VITALDB)
    dt = time.perf_counter() - t0
    det = np.asarray(det, dtype=np.int64)
    det = det[_samples_in_good(det, bad_samples)]

    tol = int(round(QRS_TOL_MS * FS_VITALDB / 1000.0))
    tp_naive = _greedy_match(gt_samples, det, tol)
    sens_naive = tp_naive / max(1, gt_samples.size)
    ppv_naive = tp_naive / max(1, det.size)
    f1_naive = 2 * sens_naive * ppv_naive / max(1e-12, sens_naive + ppv_naive)

    best_off_ms, tp_corr = _best_offset(gt_samples, det, FS_VITALDB, tol)
    sens_corr = tp_corr / max(1, gt_samples.size)
    ppv_corr = tp_corr / max(1, det.size)
    f1_corr = 2 * sens_corr * ppv_corr / max(1e-12, sens_corr + ppv_corr)

    return {
        "case_id": case_id,
        "duration_s": len(sig) / FS_VITALDB,
        "n_gt": int(gt_samples.size),
        "n_det": int(det.size),
        "tp_naive": int(tp_naive),
        "sens_naive": sens_naive, "ppv_naive": ppv_naive, "f1_naive": f1_naive,
        "best_offset_ms": int(best_off_ms),
        "tp_corrected": int(tp_corr),
        "sens_corrected": sens_corr, "ppv_corrected": ppv_corr,
        "f1_corrected": f1_corr,
        # Legacy keys (kept == naive) so downstream JSON consumers don't break.
        "tp": int(tp_naive),
        "sens": sens_naive, "ppv": ppv_naive, "f1": f1_naive,
        "time_s": dt,
    }


# --- AFib evaluation -------------------------------------------------------

def evaluate_afib(case_id: int, root: Path,
                  analysis_window: tuple[float, float],
                  cache_dir: Path | None) -> dict | None:
    """Slide 10-s windows across the analysis window only, comparing
    :func:`is_afib` predictions against the majority ``rhythm_label`` of
    beats inside each window."""
    sig = _load_waveform(case_id, cache_dir)
    if sig is None:
        return None
    start_s, end_s = analysis_window
    lo, hi = int(round(start_s * FS_VITALDB)), int(round(end_s * FS_VITALDB))
    lo = max(0, lo)
    hi = min(len(sig), hi)
    if hi - lo < WINDOW_S * FS_VITALDB:
        return None
    sig = sig[lo:hi]
    ann = _read_annotations(root, case_id)
    bad_s = _bad_quality_intervals(ann)
    bad_samples = [(int((a - start_s) * FS_VITALDB),
                    int((b - start_s) * FS_VITALDB)) for a, b in bad_s]

    win = WINDOW_S * FS_VITALDB
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "skipped": 0}
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "fp": 0}
    )
    for start in range(0, len(sig) - win + 1, win):
        end = start + win
        if _window_overlaps_bad(start, end, bad_samples):
            counts["skipped"] += 1
            continue
        # Annotation times are absolute; offset window edges by start_s
        # so the rhythm lookup hits the right beats.
        label = _window_rhythm(start / FS_VITALDB + start_s,
                               end / FS_VITALDB + start_s, ann)
        if label is None:
            counts["skipped"] += 1
            continue
        seg = sig[start:end]
        # `is_afib` will refuse trivially-flat or all-NaN windows; guard
        # those out of the metrics rather than counting them as TN.
        if not np.isfinite(seg).any() or np.ptp(seg[np.isfinite(seg)]) < 1e-6:
            counts["skipped"] += 1
            continue
        pred = bool(is_afib(seg, FS_VITALDB))
        is_af = (label == AFIB_LABEL)
        if is_af and pred:
            counts["tp"] += 1
        elif is_af and not pred:
            counts["fn"] += 1
        elif not is_af and pred:
            counts["fp"] += 1
            per_class[label]["fp"] += 1
            per_class[label]["n"] += 1
        else:
            counts["tn"] += 1
            per_class[label]["n"] += 1
    return {"case_id": case_id, "counts": counts,
            "per_class_fpr": dict(per_class)}


# --- summary ---------------------------------------------------------------

def _summarise_qrs(rows: list[dict]) -> None:
    if not rows:
        print("\n[qrs] no records evaluated")
        return
    n = len(rows)
    gt = sum(r["n_gt"] for r in rows)
    det = sum(r["n_det"] for r in rows)
    t = sum(r["time_s"] for r in rows)

    def pooled(key_tp):
        tp = sum(r[key_tp] for r in rows)
        sens = tp / max(1, gt)
        ppv = tp / max(1, det)
        f1 = 2 * sens * ppv / max(1e-12, sens + ppv)
        return sens, ppv, f1

    def macro(key_sens, key_ppv, key_f1):
        return (float(np.mean([r[key_sens] for r in rows])),
                float(np.mean([r[key_ppv] for r in rows])),
                float(np.mean([r[key_f1] for r in rows])))

    sens_p_n, ppv_p_n, f1_p_n = pooled("tp_naive")
    sens_p_c, ppv_p_c, f1_p_c = pooled("tp_corrected")
    sens_m_n, ppv_m_n, f1_m_n = macro("sens_naive", "ppv_naive", "f1_naive")
    sens_m_c, ppv_m_c, f1_m_c = macro("sens_corrected", "ppv_corrected",
                                      "f1_corrected")
    offsets = np.array([r["best_offset_ms"] for r in rows])

    print(f"\n=== QRS summary (n={n} cases, total_t={t:.1f}s) ===")
    print(f"  {'':>22} {'sens':>8} {'ppv':>8} {'f1':>8}")
    print(f"  {'naive micro (pooled)':>22} {sens_p_n:>8.4f} "
          f"{ppv_p_n:>8.4f} {f1_p_n:>8.4f}")
    print(f"  {'naive macro (per-case)':>22} {sens_m_n:>8.4f} "
          f"{ppv_m_n:>8.4f} {f1_m_n:>8.4f}")
    print(f"  {'corrected micro':>22} {sens_p_c:>8.4f} "
          f"{ppv_p_c:>8.4f} {f1_p_c:>8.4f}")
    print(f"  {'corrected macro':>22} {sens_m_c:>8.4f} "
          f"{ppv_m_c:>8.4f} {f1_m_c:>8.4f}")
    print(f"  per-case offset: median={int(np.median(offsets)):+5d}ms  "
          f"min={int(offsets.min()):+5d}ms  max={int(offsets.max()):+5d}ms  "
          f"|offset|>100ms: {int((np.abs(offsets) > 100).sum())}/{n} cases")


def _summarise_afib(rows: list[dict]) -> None:
    if not rows:
        print("\n[afib] no records evaluated")
        return
    tp = sum(r["counts"]["tp"] for r in rows)
    fn = sum(r["counts"]["fn"] for r in rows)
    fp = sum(r["counts"]["fp"] for r in rows)
    tn = sum(r["counts"]["tn"] for r in rows)
    skip = sum(r["counts"]["skipped"] for r in rows)
    pos = tp + fn
    neg = fp + tn
    sens = tp / max(1, pos)
    spec = tn / max(1, neg)
    ppv = tp / max(1, tp + fp)
    print(f"\n=== AFib summary (n={len(rows)} cases) ===")
    print(f"  windows: tp={tp}  fn={fn}  fp={fp}  tn={tn}  "
          f"(skipped: {skip})")
    print(f"  sens={sens:.4f}  spec={spec:.4f}  ppv={ppv:.4f}")

    # FPR per non-AFib rhythm class
    pooled: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "fp": 0}
    )
    for r in rows:
        for lbl, c in r["per_class_fpr"].items():
            pooled[lbl]["n"] += c["n"]
            pooled[lbl]["fp"] += c["fp"]
    if pooled:
        print("\n  FPR per non-AFib rhythm class:")
        for lbl in sorted(pooled):
            c = pooled[lbl]
            fpr = c["fp"] / max(1, c["n"])
            print(f"    {lbl:<40} n={c['n']:>6}  fp={c['fp']:>5}  "
                  f"fpr={fpr:.4f}")


# --- main ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, required=True,
        help="local copy of physionet.org/files/vitaldb-arrhythmia/1.0.0/",
    )
    ap.add_argument(
        "--task", choices=["qrs", "afib", "both"], default="both",
    )
    ap.add_argument(
        "--cases", type=int, nargs="+", default=None,
        help="subset of case_ids (default: all in metadata.csv)",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None,
        help="directory to cache fetched VitalDB waveforms as case_<id>.npy",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="write the per-case + summary JSON here",
    )
    args = ap.parse_args()

    windows = _read_metadata(args.root)
    if args.cases:
        case_ids = [c for c in args.cases if c in windows]
        missing = [c for c in args.cases if c not in windows]
        if missing:
            print(f"  warning: case_ids not in metadata.csv: {missing}",
                  file=sys.stderr)
    else:
        case_ids = sorted(windows)
    print(f"Evaluating {len(case_ids)} case(s) at fs={FS_VITALDB} Hz")
    print(f"  task={args.task}  root={args.root}")
    if args.cache_dir:
        print(f"  cache_dir={args.cache_dir}")

    results: dict[str, list[dict]] = {"qrs": [], "afib": []}
    for case_id in case_ids:
        window = windows[case_id]
        try:
            if args.task in ("qrs", "both"):
                r = evaluate_qrs(case_id, args.root, window, args.cache_dir)
                if r is None:
                    print(f"  [qrs ] case {case_id}: no waveform")
                else:
                    print(f"  [qrs ] case {case_id}: gt={r['n_gt']} "
                          f"det={r['n_det']}  "
                          f"naive f1={r['f1_naive']:.4f}  "
                          f"offset={r['best_offset_ms']:+5d}ms  "
                          f"corrected f1={r['f1_corrected']:.4f}  "
                          f"({r['time_s']:.1f}s)")
                    results["qrs"].append(r)
            if args.task in ("afib", "both"):
                r = evaluate_afib(case_id, args.root, window, args.cache_dir)
                if r is None:
                    print(f"  [afib] case {case_id}: no waveform")
                else:
                    c = r["counts"]
                    print(f"  [afib] case {case_id}: "
                          f"tp={c['tp']} fn={c['fn']} fp={c['fp']} "
                          f"tn={c['tn']} skipped={c['skipped']}")
                    results["afib"].append(r)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  case {case_id} ERROR: {e}", file=sys.stderr)

    if args.task in ("qrs", "both"):
        _summarise_qrs(results["qrs"])
    if args.task in ("afib", "both"):
        _summarise_afib(results["afib"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
