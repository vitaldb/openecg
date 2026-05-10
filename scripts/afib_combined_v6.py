"""v6 — masking + width-veto combined, with sweep.

Each candidate rule uses features computed on PVC-masked RR (from v5).
On top, a global wide-QRS veto (from v4) gates the OR-union. The veto
is sweep over multiple thresholds. Compare v3, v4, v5, v6 head-to-head.
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)


GLOBAL_TARGET_SPEC = 0.95
SPEC_FLOORS = [0.95, 0.90]
VETO_THRESHOLDS = [None, 110, 115, 120, 125, 130, 140]


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = filter_excluded(df)
    print(f"   excluded {len(EXCLUDED_NPZ)} label-noise-suspect windows "
          f"(npz: {sorted(EXCLUDED_NPZ)})")
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_full = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    w_arrs  = [np.asarray(w, dtype=np.float64) for w in df["widths_ms"]]
    y = df["y"].values
    cls = df["class"].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    max_w = np.array([float(np.max(w)) if len(w) else 0.0 for w in w_arrs])

    # Pre-mask RRs once
    rr_msk = []
    for rr, w in zip(rr_full, w_arrs):
        rr_m = mask_wide_related_rr(rr, w)
        if len(rr_m) < 4:
            rr_m = rr
        rr_msk.append(rr_m)

    print(f"## v6 — masking + width veto")
    print(f"   n={len(y)} pos={n_pos} neg={n_neg}  "
          f"global spec target ≥ {GLOBAL_TARGET_SPEC}")

    print("\n## Precomputing masked-feature scores…")
    cache_scores = {}
    for name, fn in FEATURES.items():
        for T in T_GRID:
            cache_scores[(name, T)] = np.array([fn(rr, T) for rr in rr_msk])

    print(f"\n## Sweep — spec_floor × veto V")
    print(f"{'sp_fl':<7}{'V':>5}{'sens':>8}{'spec':>8}"
          f"{'PVC':>7}{'AVB3':>7}{'NSR':>7}{'AVB2':>7}{'k':>4}{'caught':>8}")
    print("-" * 75)

    results = []
    for spec_floor in SPEC_FLOORS:
        for V in VETO_THRESHOLDS:
            veto = max_w >= V if V is not None else np.zeros(len(max_w), bool)
            pool = []
            for (name, T), sc in cache_scores.items():
                pool.extend(enumerate_rules_veto(
                    sc, y, name, T, spec_floor, veto))
            pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
            n_neg_kept = int(((y == 0) & ~veto).sum())
            fp_budget = int(np.floor((1 - GLOBAL_TARGET_SPEC) * n_neg_kept))
            chosen, tp, fp = greedy_union(pool, y, veto, fp_budget)
            pred = tp | fp
            sens = pred[y == 1].sum() / max(1, n_pos)
            spec = 1 - pred[y == 0].sum() / max(1, n_neg)
            per_cls = {}
            for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
                m = cls == c
                per_cls[c] = (pred[m].sum() / max(1, m.sum())) if m.any() else 0
            print(f"{spec_floor:<7.2f}{str(V):>5}{sens:>8.3f}{spec:>8.3f}"
                  f"{per_cls['PVC']:>7.2f}{per_cls['AVB3']:>7.2f}"
                  f"{per_cls['NSR']:>7.2f}{per_cls['AVB2']:>7.2f}"
                  f"{len(chosen):>4}{int(pred[y==1].sum()):>5}/{n_pos}")
            results.append(dict(
                spec_floor=spec_floor, V=V, sens=sens, spec=spec,
                k=len(chosen), chosen=chosen,
                **{f"fpr_{c}": v for c, v in per_cls.items() if c != "AFib"},
            ))

    valid = [r for r in results if r["spec"] >= GLOBAL_TARGET_SPEC]
    valid.sort(key=lambda r: -r["sens"])
    if valid:
        best = valid[0]
        print(f"\n## OPTIMAL v6 (max sens with spec ≥ {GLOBAL_TARGET_SPEC})")
        print(f"   spec_floor={best['spec_floor']}  veto V={best['V']}  "
              f"k={best['k']} rules")
        print(f"   sens={best['sens']:.3f}  spec={best['spec']:.3f}")
        for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
            if c == "AFib":
                print(f"     AFib: {best['sens']*100:5.1f}%")
            else:
                print(f"     {c:>5}: {best.get(f'fpr_{c}', 0)*100:5.1f}%")
        print("\n   Final ruleset (each on PVC-masked RR):")
        for i, r in enumerate(best["chosen"], 1):
            db = f"  [T={r['T']}]" if r["T"] != 0 else ""
            print(f"     R{i}:  {r['name']} {r['sign']} {r['thr']:.4g}{db}")
        if best["V"] is not None:
            print(f"\n   AND  max(qrs_width) < {best['V']} ms")

    print("\n## Pareto frontier (sens vs PVC FPR)")
    print(f"  {'sens':>6}  {'spec':>6}  {'PVC':>5}  {'AVB3':>5}  config")
    sorted_by_sens = sorted(results, key=lambda r: -r["sens"])
    for r in sorted_by_sens[:12]:
        if r["spec"] >= GLOBAL_TARGET_SPEC:
            mark = "*"
        else:
            mark = " "
        print(f"  {r['sens']:>6.3f}  {r['spec']:>6.3f}  {r['fpr_PVC']:>5.2f}  "
              f"{r['fpr_AVB3']:>5.2f}  "
              f"spec_floor={r['spec_floor']:.2f}  V={str(r['V']):>4}  "
              f"k={r['k']}  {mark}")

    print("\n## Head-to-head: previous best vs v6 best")
    prev = [
        ("v3 high-spec (RR only, no mask, no veto)", 0.637, 0.954,
         0.000, 0.020, 0.000, 0.114),
        ("v4 RR + width veto V=120 (no mask)",        0.688, 0.968,
         0.000, 0.020, 0.091, 0.063),
        ("v5 masked features (no veto)",              0.713, 0.954,
         0.000, 0.041, 0.000, 0.101),
    ]
    print(f"  {'model':<46}{'sens':>7}{'spec':>7}{'NSR':>7}"
          f"{'PVC':>7}{'AVB2':>7}{'AVB3':>7}")
    for m, s, sp, nsr, pvc, avb2, avb3 in prev:
        print(f"  {m:<46}{s:>7.3f}{sp:>7.3f}{nsr:>7.3f}"
              f"{pvc:>7.3f}{avb2:>7.3f}{avb3:>7.3f}")
    if valid:
        b = valid[0]
        print(f"  {'v6 masking + veto V=' + str(b['V']):<46}{b['sens']:>7.3f}"
              f"{b['spec']:>7.3f}{b['fpr_NSR']:>7.3f}{b['fpr_PVC']:>7.3f}"
              f"{b['fpr_AVB2']:>7.3f}{b['fpr_AVB3']:>7.3f}")


if __name__ == "__main__":
    main()
