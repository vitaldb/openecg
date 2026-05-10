"""v5 — wide-QRS-related RR masking, then re-evaluate single-metric F1.

Per-beat width is used to identify PVC/wide-QRS beats. RR intervals
touching a wide beat are *deleted* from the sequence; each feature is
recomputed on the residual. AFib+PVC stays chaotic (only PVC-related
RRs are removed; the rest are still random). Sinus+PVC collapses to a
near-regular skeleton (PVC-related RRs removed → leftover are sinus).

This is more precise than a window-level wide-QRS veto: it surgically
removes PVC contamination instead of throwing away the whole window.

Procedure
---------
1. For each window, define wide-beat mask using a threshold:
     wide_i  iff  width_i > max(120, 1.25 × median(widths))
2. RR_i (between beat i and i+1) is deleted if wide_i or wide_{i+1}.
3. Each feature is computed on the residual RR sequence.
4. For each feature × T × mode {full, masked} we report:
     AUROC, best F1 (over threshold sweep), corresponding sens/spec/precision.
5. Composite built from masked-feature pool, compared to v4.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scripts.afib_deadband_sweep import FEATURES, T_GRID, build_rr_cache

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)


WIDE_ABS_MS = 120.0
WIDE_REL_FACTOR = 1.25
MIN_BEATS_AFTER_MASK = 4


def mask_wide_related_rr(rr_ms: np.ndarray, widths_ms: np.ndarray) -> np.ndarray:
    """Delete RR intervals that touch a wide-QRS beat.

    rr_ms[i] is between beat i and beat i+1.
    widths_ms[i] is the width of beat i.
    A beat is 'wide' if width ≥ max(120 ms, 1.25 × median width).
    """
    if len(widths_ms) == 0 or len(rr_ms) == 0:
        return rr_ms.astype(np.float64).copy()
    rr = rr_ms.astype(np.float64)
    w = widths_ms.astype(np.float64)
    med_w = float(np.median(w))
    wide_thr = max(WIDE_ABS_MS, WIDE_REL_FACTOR * med_w)
    wide = w >= wide_thr
    if not wide.any():
        return rr  # nothing to mask
    n_rr = len(rr)
    keep = np.ones(n_rr, dtype=bool)
    for i in range(n_rr):
        # RR_i links beat i to beat i+1
        if i < len(wide) and wide[i]:
            keep[i] = False
        if (i + 1) < len(wide) and wide[i + 1]:
            keep[i] = False
    return rr[keep]


def best_threshold_f1(scores: np.ndarray, y: np.ndarray):
    """Find threshold maximising F1. Try both directions."""
    best_f1 = 0.0
    best = dict(sign="≥", thr=0.0, f1=0.0, sens=0.0, spec=0.0, prec=0.0,
                tp_mask=np.zeros(len(y), bool), fp_mask=np.zeros(len(y), bool))
    s = np.nan_to_num(scores.astype(float), nan=float(np.nanmedian(scores))
                      if np.isfinite(np.nanmedian(scores)) else 0.0)
    for sign in ("≥", "≤"):
        s_use = s if sign == "≥" else -s
        thrs = np.unique(s_use)
        if len(thrs) > 100:
            thrs = np.quantile(s_use, np.linspace(0, 1, 200))
        for t in thrs:
            pred = s_use >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if prec + rec < 1e-9:
                continue
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                spec = 1 - fp / max(1, (y == 0).sum())
                raw_thr = float(t if sign == "≥" else -t)
                best = dict(sign=sign, thr=raw_thr, f1=f1,
                            sens=float(rec), spec=float(spec), prec=float(prec),
                            tp_mask=pred & (y == 1), fp_mask=pred & (y == 0))
    return best


def auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    s = np.nan_to_num(scores.astype(float), nan=float(np.nanmedian(scores))
                      if np.isfinite(np.nanmedian(scores)) else 0.0)
    auc_hi = roc_auc_score(y, s)
    return max(auc_hi, 1 - auc_hi)


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_full_arrays = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    w_arrays      = [np.asarray(w, dtype=np.float64) for w in df["widths_ms"]]
    y = df["y"].values
    cls = df["class"].values

    # Build masked-RR arrays
    rr_masked_arrays = []
    n_masked_too_short = 0
    for rr, w in zip(rr_full_arrays, w_arrays):
        rr_m = mask_wide_related_rr(rr, w)
        if len(rr_m) < MIN_BEATS_AFTER_MASK:
            rr_m = rr  # fall back to full (don't lose the window)
            n_masked_too_short += 1
        rr_masked_arrays.append(rr_m)
    print(f"## Setup")
    print(f"   n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())}")
    print(f"   wide-beat criterion: width ≥ max({WIDE_ABS_MS}, "
          f"{WIDE_REL_FACTOR}×median)")
    print(f"   {n_masked_too_short} windows fell back to full (too few "
          f"RRs after masking)")

    # Diagnose: how many RRs masked per class?
    n_removed = []
    for rr, rr_m in zip(rr_full_arrays, rr_masked_arrays):
        n_removed.append(len(rr) - len(rr_m))
    n_removed = np.array(n_removed)
    print(f"\n   RRs deleted by class:")
    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        m = cls == c
        if m.any():
            arr = n_removed[m]
            print(f"     {c:>5}: median={np.median(arr):.1f}  max={arr.max()}  "
                  f"any-masked windows={int((arr>0).sum())}/{int(m.sum())}")

    # Per-metric comparison: full vs masked
    print("\n## Single-metric F1 (AFib=1) — full vs masked RR sequence")
    print(f"{'metric':<16}{'T':>5}  {'AUROC_full':>11}{'AUROC_msk':>11}  "
          f"{'F1_full':>9}{'F1_msk':>9}  {'gain':>7}")
    print("-" * 88)
    rows = []
    for name, fn in FEATURES.items():
        # Pick best T per (mode) by AUROC
        best_full = dict(auc=0.0, T=0, f1=0.0)
        best_msk  = dict(auc=0.0, T=0, f1=0.0)
        for T in T_GRID:
            sc_f = np.array([fn(rr, T) for rr in rr_full_arrays])
            sc_m = np.array([fn(rr, T) for rr in rr_masked_arrays])
            a_f, a_m = auroc(sc_f, y), auroc(sc_m, y)
            f1_f = best_threshold_f1(sc_f, y)["f1"]
            f1_m = best_threshold_f1(sc_m, y)["f1"]
            if a_f > best_full["auc"]:
                best_full = dict(auc=a_f, T=T, f1=f1_f, scores=sc_f)
            if a_m > best_msk["auc"]:
                best_msk  = dict(auc=a_m, T=T, f1=f1_m, scores=sc_m)
        # F1 details at each mode's best T
        f1f = best_threshold_f1(best_full["scores"], y)
        f1m = best_threshold_f1(best_msk ["scores"], y)
        gain = f1m["f1"] - f1f["f1"]
        print(f"{name:<16}{best_full['T']:>3}/{best_msk['T']:<2}  "
              f"{best_full['auc']:>11.3f}{best_msk['auc']:>11.3f}  "
              f"{f1f['f1']:>9.3f}{f1m['f1']:>9.3f}  {gain:>+7.3f}")
        rows.append(dict(metric=name, T_full=best_full["T"], T_msk=best_msk["T"],
                         auc_full=best_full["auc"], auc_msk=best_msk["auc"],
                         f1_full=f1f["f1"], f1_msk=f1m["f1"], gain=gain))

    print("\n## Detail at best (T, mode) per metric")
    print(f"{'metric':<14}{'mode':>7} {'T':>4}{'AUROC':>8}{'F1':>7}{'sens':>7}"
          f"{'spec':>7}{'prec':>7}  rule")
    for r in rows:
        for mode in ("full", "msk"):
            T_key = "T_full" if mode == "full" else "T_msk"
            arrs = rr_full_arrays if mode == "full" else rr_masked_arrays
            sc = np.array([FEATURES[r["metric"]](rr, r[T_key]) for rr in arrs])
            best = best_threshold_f1(sc, y)
            print(f"  {r['metric']:<12}{mode:>7} {r[T_key]:>4}{auroc(sc, y):>8.3f}"
                  f"{best['f1']:>7.3f}{best['sens']:>7.2f}"
                  f"{best['spec']:>7.2f}{best['prec']:>7.2f}  "
                  f"AFib if {r['metric']} {best['sign']} {best['thr']:.4g}")

    # Composite with masked features (high-spec OR-union)
    print("\n## Composite — same recipe as v4, but each rule uses masked RR")
    pool = []
    for name, fn in FEATURES.items():
        for T in T_GRID:
            sc = np.array([fn(rr, T) for rr in rr_masked_arrays])
            from scripts.afib_qrs_width_v3 import enumerate_rules
            pool.extend(enumerate_rules(sc, y, name, T, 0.95))
    pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
    print(f"   masked-feature rule pool: {len(pool)}")

    from scripts.afib_qrs_width_v4 import greedy_union
    veto = np.zeros(len(y), dtype=bool)
    fp_budget = int(np.floor(0.05 * int((y == 0).sum())))
    chosen, tp, fp = greedy_union(pool, y, veto, fp_budget)
    sens = (tp & (y == 1)).sum() / max(1, int((y == 1).sum()))
    spec = 1 - (fp & (y == 0)).sum() / max(1, int((y == 0).sum()))
    pred = tp | fp
    print(f"\n   masked-only composite: {len(chosen)} rules  "
          f"sens={sens:.3f}  spec={spec:.3f}")
    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        m = cls == c
        if m.any():
            print(f"     {c:>5}: {pred[m].mean()*100:5.1f}%  "
                  f"({int(pred[m].sum())}/{m.sum()})")

    print("\n   chosen rules (each on masked RR):")
    for i, r in enumerate(chosen, 1):
        db = f"  [T={r['T']}]" if r["T"] != 0 else ""
        print(f"     R{i}:  {r['name']} {r['sign']} {r['thr']:.4g}{db}")


if __name__ == "__main__":
    main()
