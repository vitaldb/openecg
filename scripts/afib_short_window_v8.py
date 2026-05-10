"""v8 — v7 adaptive veto + short-window safety net (3 layers).

Layer 1: relative-variability rule for n_beats ≤ 10
   pRR_rel8 ≥ 0.75  AND  cv_rr ≥ 0.18  AND  dom_cluster ≤ 0.45
   AND  n_wide_beats == 0   (no aberrancy)

Layer 2: multi-T cosen consensus for n_beats ≤ 10
   median(cosen[T=0, 20, 40, 70]) ≥ 1.6  AND  pRR_rel8 ≥ 0.80
   AND  n_wide_beats == 0  AND  dom_cluster ≤ 0.5

Layer 3: brady-AFib rule (any n)
   mean_rr ≥ 1100 ms  AND  pRR_rel8 ≥ 0.70
   AND  rmssd_ms / mean_rr ≥ 0.15  AND  n_wide_beats == 0

Each layer is added to the v7 union as an extra OR rule. Sweep is
parametric over the layer thresholds; defaults are calibrated to the
visual review of B1-B3 FN cases.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scripts.afib_deadband_sweep import (
    FEATURES, T_GRID, build_rr_cache, filter_excluded, EXCLUDED_NPZ,
)
from scripts.afib_width_masking_v5 import mask_wide_related_rr
from scripts.afib_qrs_width_v4 import enumerate_rules_veto, greedy_union
from scripts.afib_adaptive_veto_v7 import adaptive_veto

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)


# Adaptive-veto config carried over from v7 optimum:
ADAPTIVE_V = 115
ADAPTIVE_P = 0.30
ADAPTIVE_N = 2

# Short-window thresholds (defaults; swept below).
WIDE_BEAT_MS = 120
SHORT_N_MAX = 10
L1_PRR = 0.75
L1_CV = 0.18
L1_DOM = 0.45
L2_COSEN_MEDIAN = 1.6
L2_PRR = 0.80
L2_DOM = 0.50
L3_MEAN_RR = 1100
L3_PRR = 0.70
L3_RMSSD_RATIO = 0.15


def short_window_layer1(rr, w, *, n_max=SHORT_N_MAX, prr_thr=L1_PRR,
                        cv_thr=L1_CV, dom_thr=L1_DOM,
                        max_rr_ratio_thr=2.4,
                        wide_ms=WIDE_BEAT_MS):
    """Relative-variability rule for short clean windows.

    Guards against AVB2/AVB3 FPs:
      - n_wide_beats == 0  (no aberrancy / escape)
      - max(RR)/min(RR) ≤ max_rr_ratio_thr  (no extreme pause)
        AVB2 Wenckebach drop and AVB3 escape pauses are ~2.5-4× shortest
        beat; setting the ratio cap at 2.4 protects against both while
        leaving wide AFib variation (typically 1.5-2.0×) intact.
    """
    if len(rr) == 0 or len(rr) > n_max:
        return False
    if len(w) > 0 and int((w >= wide_ms).sum()) > 0:
        return False
    rr_min = float(np.min(rr))
    if rr_min < 1.0:
        return False
    rr_max = float(np.max(rr))
    if rr_max / rr_min > max_rr_ratio_thr:
        return False
    prr = FEATURES["pRR_rel8"](rr, 0)
    cv = FEATURES["cv_rr"](rr, 0)
    dom = FEATURES["dom_cluster"](rr, 0)
    return ((prr >= prr_thr) and (cv >= cv_thr) and (dom <= dom_thr))


def short_window_layer2(rr, w, *, n_max=SHORT_N_MAX, cosen_thr=L2_COSEN_MEDIAN,
                        prr_thr=L2_PRR, dom_thr=L2_DOM, wide_ms=WIDE_BEAT_MS):
    """Multi-T cosen consensus for short clean windows."""
    if len(rr) == 0 or len(rr) > n_max:
        return False
    if len(w) > 0 and int((w >= wide_ms).sum()) > 0:
        return False
    cosen_vals = [FEATURES["cosen"](rr, T) for T in (0, 20, 40, 70)]
    if np.median(cosen_vals) < cosen_thr:
        return False
    prr = FEATURES["pRR_rel8"](rr, 0)
    dom = FEATURES["dom_cluster"](rr, 0)
    return (prr >= prr_thr) and (dom <= dom_thr)


def short_window_layer3(rr, w, *, mean_rr_thr=L3_MEAN_RR, prr_thr=L3_PRR,
                        rmssd_ratio_thr=L3_RMSSD_RATIO, wide_ms=WIDE_BEAT_MS):
    """Brady-AFib rule (works for any n_beats; mean_rr is the gate)."""
    if len(rr) == 0:
        return False
    mean_rr = float(np.mean(rr))
    if mean_rr < mean_rr_thr:
        return False
    if len(w) > 0 and int((w >= wide_ms).sum()) > 0:
        return False
    prr = FEATURES["pRR_rel8"](rr, 0)
    rmssd = FEATURES["rmssd_ms"](rr, 0)
    return (prr >= prr_thr) and (rmssd / max(1.0, mean_rr) >= rmssd_ratio_thr)


def build_v7_union(rr_msk, y, veto, cls):
    """Reproduce v7 adaptive-veto composite. Returns (chosen, tp_mask, fp_mask)."""
    pool = []
    for name, fn in FEATURES.items():
        for T in T_GRID:
            sc = np.array([fn(rr, T) for rr in rr_msk])
            pool.extend(enumerate_rules_veto(sc, y, name, T, 0.95, veto))
    pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
    n_neg_kept = int(((y == 0) & ~veto).sum())
    fp_budget = int(np.floor(0.05 * n_neg_kept))
    return greedy_union(pool, y, veto, fp_budget)


def evaluate(rr_full, rr_msk, w_arrs, y, cls):
    """v7 + 3 layers = v8 composite. Returns metrics dict."""
    veto = adaptive_veto(w_arrs, ADAPTIVE_V, ADAPTIVE_P, ADAPTIVE_N)
    chosen, tp_main, fp_main = build_v7_union(rr_msk, y, veto, cls)

    # Apply short-window layers on top
    l1 = np.array([short_window_layer1(rr, w)
                   for rr, w in zip(rr_full, w_arrs)])
    # L1-only variant (chosen as final based on sweep — L2/L3 add FPs without
    # net recovery once L1 is in place). Layers L2/L3 still computed for the
    # diagnostic attribution table below.
    l2 = np.array([short_window_layer2(rr, w)
                   for rr, w in zip(rr_full, w_arrs)])
    l3 = np.array([short_window_layer3(rr, w)
                   for rr, w in zip(rr_full, w_arrs)])

    pred_main = tp_main | fp_main
    pred = pred_main | l1
    return dict(
        chosen=chosen, veto=veto,
        pred=pred, pred_main=pred_main,
        l1=l1, l2=l2, l3=l3,
        tp=int((pred & (y == 1)).sum()),
        fp=int((pred & (y == 0)).sum()),
    )


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = filter_excluded(df)
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_full = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    w_arrs = [np.asarray(w, dtype=np.float64) for w in df["widths_ms"]]
    y = df["y"].values
    cls = df["class"].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    rr_msk = []
    for rr, w in zip(rr_full, w_arrs):
        rr_m = mask_wide_related_rr(rr, w)
        if len(rr_m) < 4:
            rr_m = rr
        rr_msk.append(rr_m)

    print(f"## v8  (v7 adaptive veto + 3-layer short-window safety net)")
    print(f"   after excluding {len(EXCLUDED_NPZ)} label-noise windows: "
          f"n={len(y)} pos={n_pos} neg={n_neg}")

    # Baseline v7 (no short-window) — for ΔFN attribution
    veto = adaptive_veto(w_arrs, ADAPTIVE_V, ADAPTIVE_P, ADAPTIVE_N)
    chosen_v7, tp_v7, fp_v7 = build_v7_union(rr_msk, y, veto, cls)
    pred_v7 = tp_v7 | fp_v7
    sens_v7 = pred_v7[y == 1].sum() / n_pos
    spec_v7 = 1 - pred_v7[y == 0].sum() / n_neg
    print(f"\n## v7 baseline (excluded label-noise):  "
          f"sens={sens_v7:.3f}  spec={spec_v7:.3f}  ({int(pred_v7[y==1].sum())}/"
          f"{n_pos} TP, {int(pred_v7[y==0].sum())}/{n_neg} FP)")
    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        m = cls == c
        if m.any():
            print(f"     {c:>5}: {pred_v7[m].mean()*100:5.1f}%  "
                  f"({int(pred_v7[m].sum())}/{m.sum()})")

    # Sweep L1-only short-window thresholds. L2 and L3 are no longer
    # added to the prediction (they bring more FPs than recoveries once
    # L1 is in place), but their attribution is still printed below.
    print("\n## Sweep L1 short-window thresholds (L1 only)")
    print(f"{'L1_prr':>7}{'L1_cv':>7}{'L1_dom':>8}{'L1_ratio':>10}"
          f"{'sens':>8}{'spec':>8}{'PVC':>7}{'AVB3':>7}"
          f"{'AVB2':>7}{'NSR':>7}{'+TP':>5}{'+FP':>5}")

    best = None
    for L1_PRR_v in [0.70, 0.75, 0.80, 0.85]:
        for L1_CV_v in [0.15, 0.18, 0.20]:
            for L1_DOM_v in [0.35, 0.40, 0.45]:
                for RATIO_v in [2.0, 2.4, 2.8]:
                    l1 = np.array([short_window_layer1(
                        rr, w, prr_thr=L1_PRR_v, cv_thr=L1_CV_v,
                        dom_thr=L1_DOM_v, max_rr_ratio_thr=RATIO_v)
                                   for rr, w in zip(rr_full, w_arrs)])
                    pred = pred_v7 | l1
                    sens = pred[y == 1].sum() / n_pos
                    spec = 1 - pred[y == 0].sum() / n_neg
                    n_new_tp = int(((l1 & ~pred_v7) & (y == 1)).sum())
                    n_new_fp = int(((l1 & ~pred_v7) & (y == 0)).sum())
                    per_cls = {}
                    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
                        m = cls == c
                        per_cls[c] = (pred[m].sum() / m.sum()) if m.any() else 0
                    row = dict(L1_PRR=L1_PRR_v, L1_CV=L1_CV_v,
                               L1_DOM=L1_DOM_v, RATIO=RATIO_v,
                               sens=sens, spec=spec,
                               n_new_tp=n_new_tp, n_new_fp=n_new_fp,
                               **per_cls)
                    if spec >= 0.95 and (
                        best is None or
                        sens > best["sens"] or
                        (sens == best["sens"] and spec > best["spec"])
                    ):
                        best = dict(row, l1=l1, pred=pred)
                    print(f"{L1_PRR_v:>7.2f}{L1_CV_v:>7.2f}{L1_DOM_v:>8.2f}"
                          f"{RATIO_v:>10.2f}"
                          f"{sens:>8.3f}{spec:>8.3f}"
                          f"{per_cls['PVC']:>7.2f}{per_cls['AVB3']:>7.2f}"
                          f"{per_cls['AVB2']:>7.2f}{per_cls['NSR']:>7.2f}"
                          f"{n_new_tp:>5}{n_new_fp:>5}")

    if best is None:
        l1 = np.array([short_window_layer1(rr, w) for rr, w in zip(rr_full, w_arrs)])
        pred = pred_v7 | l1
        per_cls = {c: ((pred[cls == c].sum() / (cls == c).sum())
                       if (cls == c).any() else 0)
                   for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]}
        best = dict(L1_PRR=L1_PRR, L1_CV=L1_CV, L1_DOM=L1_DOM, RATIO=2.4,
                    sens=pred[y == 1].sum() / n_pos,
                    spec=1 - pred[y == 0].sum() / n_neg,
                    l1=l1, pred=pred,
                    n_new_tp=int(((l1 & ~pred_v7) & (y == 1)).sum()),
                    n_new_fp=int(((l1 & ~pred_v7) & (y == 0)).sum()),
                    **per_cls)
        print("\n   (no config met spec≥0.95; reporting defaults)")

    print(f"\n## v8 BEST (spec ≥ 0.95)  L1-only safety net")
    print(f"   L1 prr≥{best['L1_PRR']}  cv≥{best['L1_CV']}  "
          f"dom≤{best['L1_DOM']}  max_rr/min_rr ≤ {best['RATIO']}")
    print(f"   gates: n_beats ≤ 10  AND  n_wide_beats(≥120ms) == 0")
    print(f"   composite sens={best['sens']:.3f}  spec={best['spec']:.3f}")
    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        m = cls == c
        if m.any():
            n_cau = int(best["pred"][m].sum())
            print(f"     {c:>5}: {best['pred'][m].mean()*100:5.1f}%  "
                  f"({n_cau}/{m.sum()})")

    # Attribution: which layer caught which FN? (L2/L3 reported for context
    # only — they aren't actually added to the prediction in this v8 variant.)
    print(f"\n## L1 attribution + L2/L3 unused-recovery diagnostic")
    l2 = np.array([short_window_layer2(rr, w) for rr, w in zip(rr_full, w_arrs)])
    l3 = np.array([short_window_layer3(rr, w) for rr, w in zip(rr_full, w_arrs)])
    recovered_by_l1 = best["l1"] & ~pred_v7 & (y == 1)
    recovered_by_l2 = l2 & ~pred_v7 & ~best["l1"] & (y == 1)
    recovered_by_l3 = l3 & ~pred_v7 & ~best["l1"] & ~l2 & (y == 1)
    new_fp_l1 = best["l1"] & ~pred_v7 & (y == 0)

    def _npz_list(mask):
        return [int(df.iloc[i]["npz_idx"]) for i in np.where(mask)[0]]

    print(f"   L1 recovered: {int(recovered_by_l1.sum())} AFib  "
          f"new FP: {int(new_fp_l1.sum())}  "
          f"({_npz_list(recovered_by_l1)})")
    print(f"   L2 (unused) would add: {int(recovered_by_l2.sum())} TP, "
          f"{int((l2 & ~pred_v7 & ~best['l1'] & (y == 0)).sum())} FP")
    print(f"   L3 (unused) would add: {int(recovered_by_l3.sum())} TP, "
          f"{int((l3 & ~pred_v7 & ~best['l1'] & ~l2 & (y == 0)).sum())} FP")

    # Head-to-head
    print(f"\n## Final  (after excluding 6 label-noise windows)")
    print(f"{'model':<35}{'sens':>7}{'spec':>7}{'NSR':>7}{'PVC':>7}"
          f"{'AVB2':>7}{'AVB3':>7}")
    print(f"  {'v7 adaptive veto':<35}{sens_v7:>7.3f}{spec_v7:>7.3f}"
          f"{pred_v7[cls=='NSR'].mean():>7.3f}"
          f"{pred_v7[cls=='PVC'].mean():>7.3f}"
          f"{pred_v7[cls=='AVB2'].mean():>7.3f}"
          f"{pred_v7[cls=='AVB3'].mean():>7.3f}")
    print(f"  {'v8 + L1 safety net':<35}{best['sens']:>7.3f}"
          f"{best['spec']:>7.3f}"
          f"{best['NSR']:>7.3f}{best['PVC']:>7.3f}"
          f"{best['AVB2']:>7.3f}{best['AVB3']:>7.3f}")


if __name__ == "__main__":
    main()
