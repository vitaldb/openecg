"""Validate ``openecg.detect_qrs`` on the MIT-BIH Arrhythmia DB
(the AAMI EC57 reference cohort) against ``pyvital.detect_qrs``.

Both detectors are Pan-Tompkins variants. We evaluate per-record on
lead 0 (typically MLII) with the standard 100 ms (= ±36 samples @ 360 Hz)
matching tolerance:

    sens = TP / (TP + FN) = TP / |GT|
    PPV  = TP / (TP + FP) = TP / |DET|
    F1   = 2·sens·PPV / (sens + PPV)

We report per-record numbers and macro-aggregate across all 48 records
plus the 44-record AAMI subset (excludes 102, 104, 107, 217 — paced).

Usage:
    python scripts/validate_qrs_mitdb.py            # all 48 records
    python scripts/validate_qrs_mitdb.py --aami     # 44-record AAMI set
    python scripts/validate_qrs_mitdb.py --records 100 101 105
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import detect_qrs as openecg_detect_qrs
from openecg import mitdb

TOL_MS = 100.0


def _greedy_match(gt: np.ndarray, det: np.ndarray, tol: int) -> int:
    """Greedy bipartite match: each detection matches at most one GT
    within ``tol`` samples; sweep gt sorted ascending."""
    if gt.size == 0 or det.size == 0:
        return 0
    gt_sorted = np.sort(gt)
    det_sorted = np.sort(det)
    used = np.zeros(det_sorted.size, dtype=bool)
    matched = 0
    j_start = 0
    for g in gt_sorted:
        # Skip detections too far below g - tol; advance j_start.
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


def _pyvital_detect_qrs(signal: np.ndarray, fs: int) -> np.ndarray:
    import pyvital as pv
    return np.asarray(pv.detect_qrs(np.asarray(signal, dtype=np.float64), fs),
                      dtype=np.int64)


def _evaluate(name: str, fn, record_ids: list[int]) -> dict:
    """Run detector ``fn`` on each record's lead 0 and return per-record
    + aggregate metrics."""
    rows = []
    total_tp = total_gt = total_det = 0
    total_t = 0.0
    print(f"\n=== {name} ===")
    print(f"{'rec':>4} {'gt':>5} {'det':>5} {'tp':>5} {'sens':>6} "
          f"{'ppv':>6} {'f1':>6} {'time_s':>7}")
    for rid in record_ids:
        rec = mitdb.load_record(rid)
        sig = rec["signal"][:, 0].astype(np.float64)
        fs = rec["fs"]
        gt = mitdb.load_qrs_peaks(rid)
        tol = int(round(TOL_MS * fs / 1000.0))
        t0 = time.perf_counter()
        try:
            det = fn(sig, fs)
        except Exception as e:
            print(f"{rid:>4}  EXCEPTION: {e}")
            continue
        dt = time.perf_counter() - t0
        det = np.asarray(det, dtype=np.int64)
        tp = _greedy_match(gt, det, tol)
        sens = tp / max(1, gt.size)
        ppv = tp / max(1, det.size)
        f1 = 2 * sens * ppv / max(1e-12, sens + ppv)
        rows.append({
            "rid": rid, "gt": int(gt.size), "det": int(det.size),
            "tp": tp, "sens": sens, "ppv": ppv, "f1": f1, "time_s": dt,
        })
        total_tp += tp
        total_gt += gt.size
        total_det += det.size
        total_t += dt
        print(f"{rid:>4} {gt.size:>5} {det.size:>5} {tp:>5} "
              f"{sens:>6.4f} {ppv:>6.4f} {f1:>6.4f} {dt:>7.2f}")
    sens_g = total_tp / max(1, total_gt)
    ppv_g = total_tp / max(1, total_det)
    f1_g = 2 * sens_g * ppv_g / max(1e-12, sens_g + ppv_g)
    macro_sens = float(np.mean([r["sens"] for r in rows])) if rows else 0.0
    macro_ppv = float(np.mean([r["ppv"] for r in rows])) if rows else 0.0
    macro_f1 = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
    print(f"\n  micro (pooled TP/GT/DET):  sens={sens_g:.4f}  "
          f"ppv={ppv_g:.4f}  f1={f1_g:.4f}  total_t={total_t:.1f}s")
    print(f"  macro (mean over records): sens={macro_sens:.4f}  "
          f"ppv={macro_ppv:.4f}  f1={macro_f1:.4f}")
    return {
        "rows": rows,
        "micro": {"sens": sens_g, "ppv": ppv_g, "f1": f1_g, "total_t": total_t},
        "macro": {"sens": macro_sens, "ppv": macro_ppv, "f1": macro_f1},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aami", action="store_true",
                    help="evaluate the 44-record AAMI EC57 subset")
    ap.add_argument("--records", type=int, nargs="+", default=None,
                    help="evaluate only these record IDs")
    ap.add_argument("--skip-pyvital", action="store_true",
                    help="skip pyvital comparison (e.g., when not installed)")
    args = ap.parse_args()

    if args.records:
        record_ids = sorted(args.records)
    elif args.aami:
        record_ids = mitdb.aami_record_ids()
    else:
        record_ids = mitdb.all_record_ids()
    print(f"evaluating on {len(record_ids)} records: {record_ids}")
    print(f"tolerance: {TOL_MS:.0f} ms (= {int(TOL_MS * 360 / 1000)} samples @360Hz)")

    res_oe = _evaluate("openecg.detect_qrs", openecg_detect_qrs, record_ids)

    res_pv = None
    if not args.skip_pyvital:
        try:
            import pyvital  # noqa: F401
            res_pv = _evaluate("pyvital.detect_qrs", _pyvital_detect_qrs,
                               record_ids)
        except ImportError as e:
            print(f"\n[skipping pyvital] {e}")

    if res_pv is not None:
        print("\n=== summary ===")
        print(f"{'detector':<22} {'micro_f1':>9} {'macro_f1':>9} "
              f"{'sens_micro':>11} {'ppv_micro':>10}")
        for label, res in [("openecg.detect_qrs", res_oe),
                           ("pyvital.detect_qrs", res_pv)]:
            print(f"{label:<22} {res['micro']['f1']:>9.4f} "
                  f"{res['macro']['f1']:>9.4f} "
                  f"{res['micro']['sens']:>11.4f} "
                  f"{res['micro']['ppv']:>10.4f}")
        # Per-record delta — flag any record where openecg under-performs.
        print("\n=== per-record F1 delta (openecg − pyvital) ===")
        loss = 0
        for ro, rp in zip(res_oe["rows"], res_pv["rows"]):
            d = ro["f1"] - rp["f1"]
            flag = ""
            if d < -0.005:
                flag = "  <-- openecg WORSE"
                loss += 1
            elif d > 0.005:
                flag = "  ++ openecg better"
            print(f"  rid={ro['rid']:>3}  oe={ro['f1']:.4f}  "
                  f"pv={rp['f1']:.4f}  Δ={d:+.4f}{flag}")
        n = len(res_oe["rows"])
        print(f"\nopenecg worse on {loss}/{n} records "
              f"(diff threshold 0.005 F1).")


if __name__ == "__main__":
    main()
