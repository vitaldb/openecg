"""v7 — adaptive width veto.

Instead of vetoing the whole window when any single beat is wide
(``max_w >= V`` in v6), require that *enough* beats are wide. Concretely:

  veto  iff  wide_fraction ≥ P_min  AND  wide_count ≥ N_min

This keeps "AFib + 1-2 aberrant beats" predictable while still excluding
bigeminy / ventricular-escape windows where wide beats dominate.

Sweep (V_per_beat, P_min, N_min) to find the Pareto-optimum under
global spec ≥ 95%.
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
V_PER_BEAT_GRID = [115, 120, 125, 130]
P_MIN_GRID = [0.20, 0.25, 0.30, 0.40, 0.50]
N_MIN_GRID = [2, 3]


def adaptive_veto(widths_list, V_per_beat, P_min, N_min):
    """Return boolean veto mask. A window is vetoed iff
    (#beats with width >= V_per_beat) >= max(N_min, P_min × n_beats)."""
    out = np.zeros(len(widths_list), dtype=bool)
    for i, w in enumerate(widths_list):
        if len(w) == 0:
            continue
        n_wide = int(np.sum(w >= V_per_beat))
        threshold = max(N_min, int(np.ceil(P_min * len(w))))
        out[i] = n_wide >= threshold
    return out


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = filter_excluded(df)
    print(f"   excluded {len(EXCLUDED_NPZ)} label-noise-suspect windows")
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_full = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    w_arrs = [np.asarray(w, dtype=np.float64) for w in df["widths_ms"]]
    y = df["y"].values
    cls = df["class"].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    # Pre-mask RRs once
    rr_msk = []
    for rr, w in zip(rr_full, w_arrs):
        rr_m = mask_wide_related_rr(rr, w)
        if len(rr_m) < 4:
            rr_m = rr
        rr_msk.append(rr_m)

    print(f"## v7 adaptive veto — n={len(y)} pos={n_pos} neg={n_neg}")

    print("\n## Precomputing masked-feature scores…")
    cache_scores = {}
    for name, fn in FEATURES.items():
        for T in T_GRID:
            cache_scores[(name, T)] = np.array([fn(rr, T) for rr in rr_msk])

    # Reference: v6 (max_w >= 125)
    max_w = np.array([float(np.max(w)) if len(w) else 0.0 for w in w_arrs])
    veto_v6 = max_w >= 125

    print(f"\n## Sweep — (V_per_beat, P_min, N_min)")
    print(f"{'V':>4}{'Pmin':>7}{'Nmin':>5}{'#veto':>7}"
          f"{'sens':>8}{'spec':>8}"
          f"{'PVC':>7}{'AVB3':>7}{'NSR':>7}{'AVB2':>7}{'k':>4}")
    print("-" * 72)
    results = []
    for V in V_PER_BEAT_GRID:
        for P in P_MIN_GRID:
            for N in N_MIN_GRID:
                veto = adaptive_veto(w_arrs, V, P, N)
                pool = []
                for (name, T), sc in cache_scores.items():
                    pool.extend(enumerate_rules_veto(sc, y, name, T, 0.95, veto))
                pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
                n_neg_kept = int(((y == 0) & ~veto).sum())
                fp_budget = int(np.floor(0.05 * n_neg_kept))
                chosen, tp, fp = greedy_union(pool, y, veto, fp_budget)
                pred = tp | fp
                sens = pred[y == 1].sum() / max(1, n_pos)
                spec = 1 - pred[y == 0].sum() / max(1, n_neg)
                per_cls = {}
                for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
                    m = cls == c
                    per_cls[c] = (pred[m].sum() / max(1, m.sum())) if m.any() else 0
                n_veto = int(veto.sum())
                print(f"{V:>4}{P:>7.2f}{N:>5}{n_veto:>7}"
                      f"{sens:>8.3f}{spec:>8.3f}"
                      f"{per_cls['PVC']:>7.2f}{per_cls['AVB3']:>7.2f}"
                      f"{per_cls['NSR']:>7.2f}{per_cls['AVB2']:>7.2f}"
                      f"{len(chosen):>4}")
                results.append(dict(V=V, P=P, N=N, sens=sens, spec=spec,
                                    chosen=chosen, n_veto=n_veto,
                                    **{f"fpr_{c}": v for c, v in per_cls.items()
                                       if c != "AFib"}))

    # Pareto best by sens with global spec >= 0.95
    valid = [r for r in results if r["spec"] >= GLOBAL_TARGET_SPEC]
    valid.sort(key=lambda r: -r["sens"])

    print(f"\n## Top 10 by sens (spec ≥ {GLOBAL_TARGET_SPEC})")
    for r in valid[:10]:
        print(f"   V={r['V']} P={r['P']:.2f} N={r['N']}  "
              f"sens={r['sens']:.3f}  spec={r['spec']:.3f}  "
              f"PVC={r['fpr_PVC']:.2f}  AVB3={r['fpr_AVB3']:.2f}  "
              f"k={len(r['chosen'])}  veto={r['n_veto']}")

    if valid:
        best = valid[0]
        print(f"\n## OPTIMAL v7  (adaptive veto)")
        print(f"   wide-beat threshold V_per_beat = {best['V']} ms")
        print(f"   veto rule: wide_count ≥ max({best['N']}, "
              f"{best['P']:.0%} × n_beats)")
        print(f"   sens = {best['sens']:.3f}   spec = {best['spec']:.3f}")
        print(f"   per-class:")
        for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
            if c == "AFib":
                print(f"     AFib: {best['sens']*100:5.1f}%")
            else:
                print(f"     {c:>5}: {best.get(f'fpr_{c}', 0)*100:5.1f}%")
        print(f"\n   {len(best['chosen'])} chosen rules:")
        for i, r in enumerate(best["chosen"], 1):
            db = f"  [T={r['T']}]" if r["T"] != 0 else ""
            print(f"     R{i}:  {r['name']} {r['sign']} {r['thr']:.4g}{db}")

    # Diff vs v6: which FN got recovered, which gained as FP?
    print("\n## FN recovery analysis (v6 max_w≥125 → v7 best)")
    pool_v6 = []
    for (name, T), sc in cache_scores.items():
        pool_v6.extend(enumerate_rules_veto(sc, y, name, T, 0.95, veto_v6))
    pool_v6.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
    n_neg_kept_v6 = int(((y == 0) & ~veto_v6).sum())
    fp_budget_v6 = int(np.floor(0.05 * n_neg_kept_v6))
    chosen_v6, tp_v6, fp_v6 = greedy_union(pool_v6, y, veto_v6, fp_budget_v6)
    pred_v6 = tp_v6 | fp_v6
    sens_v6 = pred_v6[y == 1].sum() / max(1, n_pos)
    spec_v6 = 1 - pred_v6[y == 0].sum() / max(1, n_neg)
    print(f"   v6: sens={sens_v6:.3f} spec={spec_v6:.3f}")

    if valid:
        best = valid[0]
        veto_best = adaptive_veto(w_arrs, best["V"], best["P"], best["N"])
        pool_best = []
        for (name, T), sc in cache_scores.items():
            pool_best.extend(enumerate_rules_veto(sc, y, name, T, 0.95, veto_best))
        pool_best.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
        n_neg_kept_b = int(((y == 0) & ~veto_best).sum())
        fp_budget_b = int(np.floor(0.05 * n_neg_kept_b))
        chosen_b, tp_b, fp_b = greedy_union(pool_best, y, veto_best, fp_budget_b)
        pred_b = tp_b | fp_b

        recovered = (~pred_v6 & pred_b) & (y == 1)
        lost = (pred_v6 & ~pred_b) & (y == 1)
        new_fp = (~pred_v6 & pred_b) & (y == 0)
        cleared_fp = (pred_v6 & ~pred_b) & (y == 0)
        print(f"   AFib FN recovered : {int(recovered.sum())} cases "
              f"(npz: {[int(df.iloc[i]['npz_idx']) for i in np.where(recovered)[0]]})")
        print(f"   AFib lost         : {int(lost.sum())}")
        print(f"   New FP            : {int(new_fp.sum())} "
              f"(classes: {dict(zip(*np.unique(cls[new_fp], return_counts=True)))})")
        print(f"   FP cleared        : {int(cleared_fp.sum())}")


if __name__ == "__main__":
    main()
