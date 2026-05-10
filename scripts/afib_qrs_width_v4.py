"""v4 — composite RR rules + global wide-QRS veto, veto-aware enumeration.

For each candidate rule and each V, compute the rule's TP/FP *after* the
width veto is applied. Filter rules by post-veto spec ≥ spec_floor.
Greedy under post-veto FP budget. Outer sweep over V and spec_floor to
find the Pareto-optimum.
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


GLOBAL_TARGET_SPEC = 0.95
SPEC_FLOORS = [0.95, 0.90, 0.85, 0.80, 0.75]
VETO_THRESHOLDS = [None, 95, 100, 105, 110, 115, 120, 125, 130]


def enumerate_rules_veto(scores, y, name, T, spec_floor, veto_mask):
    """Like enumerate_rules but TP/FP computed only on non-vetoed windows."""
    out = []
    keep = ~veto_mask
    n_pos = int(((y == 1) & keep).sum())
    n_neg = int(((y == 0) & keep).sum())
    if n_pos == 0 or n_neg == 0:
        return out
    for sign in ("≥", "≤"):
        s_use = scores if sign == "≥" else -scores
        for t in np.unique(s_use):
            pred_raw = s_use >= t
            pred = pred_raw & keep
            tp = pred & (y == 1)
            fp = pred & (y == 0)
            sens = tp.sum() / n_pos
            spec = 1 - fp.sum() / n_neg
            if spec < spec_floor or tp.sum() == 0:
                continue
            raw_thr = float(t if sign == "≥" else -t)
            out.append(dict(
                name=name, T=int(T), sign=sign, thr=raw_thr,
                tp_mask=tp, fp_mask=fp,
                sens=float(sens), spec=float(spec),
                n_tp=int(tp.sum()), n_fp=int(fp.sum()),
            ))
    return out


def greedy_union(rule_pool, y, veto_mask, fp_budget):
    """Greedy with veto applied. Budget is on post-veto FPs."""
    n_pos = ((y == 1) & ~veto_mask).sum()
    chosen = []
    used = set()
    union_tp = np.zeros(len(y), dtype=bool)
    union_fp = np.zeros(len(y), dtype=bool)
    while True:
        best = None
        best_score = 0.0
        best_meta = None
        for r in rule_pool:
            key = (r["name"], r["T"])
            if key in used:
                continue
            new_tp = int((r["tp_mask"] & ~union_tp).sum())
            new_fp = int((r["fp_mask"] & ~union_fp).sum())
            if new_tp == 0:
                continue
            if (union_fp | r["fp_mask"]).sum() > fp_budget:
                continue
            score = new_tp / (1.0 + new_fp)
            if score > best_score:
                best_score = score
                best = r
                best_meta = (new_tp, new_fp)
        if best is None:
            break
        chosen.append(best)
        used.add((best["name"], best["T"]))
        union_tp |= best["tp_mask"]
        union_fp |= best["fp_mask"]
    return chosen, union_tp, union_fp


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_arrays = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    w_arrays  = [np.asarray(w, dtype=np.float64) for w in df["widths_ms"]]
    y = df["y"].values
    cls = df["class"].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    max_w = np.array([float(np.max(w)) if len(w) else 0.0 for w in w_arrays])

    print(f"## Setup n={len(y)} pos={n_pos} neg={n_neg}  "
          f"global target spec ≥ {GLOBAL_TARGET_SPEC}")

    # Precompute scores
    print("\n## Precomputing feature × T scores…")
    cache_scores = {}
    for name, fn in FEATURES.items():
        for T in T_GRID:
            cache_scores[(name, T)] = np.array([fn(rr, T) for rr in rr_arrays])

    print(f"\n## Sweep — per-rule spec floor × wide-QRS veto V (veto-aware)")
    print(f"{'spec_fl':<8}{'V':>5}{'sens':>8}{'spec':>8}"
          f"{'PVC':>7}{'AVB3':>7}{'NSR':>7}{'AVB2':>7}{'k':>4}{'caught':>8}")
    print("-" * 75)

    results = []
    for spec_floor in SPEC_FLOORS:
        for V in VETO_THRESHOLDS:
            veto = max_w >= V if V is not None else np.zeros(len(max_w), dtype=bool)
            # Build veto-aware pool
            pool = []
            for (name, T), scores in cache_scores.items():
                pool.extend(enumerate_rules_veto(
                    scores, y, name, T, spec_floor, veto))
            pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))

            # FP budget on post-veto negatives
            n_neg_kept = int(((y == 0) & ~veto).sum())
            fp_budget = int(np.floor((1 - GLOBAL_TARGET_SPEC) * n_neg_kept))

            chosen, tp, fp = greedy_union(pool, y, veto, fp_budget)
            # Global metrics (denominator = original n_pos / n_neg, not veto-filtered)
            pred = tp | fp
            sens = pred[y == 1].sum() / max(1, n_pos)
            spec = 1 - pred[y == 0].sum() / max(1, n_neg)
            per_cls = {}
            for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
                m = cls == c
                per_cls[c] = (pred[m].sum() / max(1, m.sum())) if m.any() else 0.0
            caught = int(pred[y == 1].sum())
            print(f"{spec_floor:<8.2f}{str(V):>5}{sens:>8.3f}{spec:>8.3f}"
                  f"{per_cls['PVC']:>7.2f}{per_cls['AVB3']:>7.2f}"
                  f"{per_cls['NSR']:>7.2f}{per_cls['AVB2']:>7.2f}"
                  f"{len(chosen):>4}{caught:>5}/{n_pos}")
            results.append({
                "spec_floor": spec_floor, "V": V,
                "sens": sens, "spec": spec, "k": len(chosen),
                "chosen": chosen,
                **{f"fpr_{c}": v for c, v in per_cls.items() if c != "AFib"},
            })

    # Pick best by sens with global spec ≥ 0.95
    valid = [r for r in results if r["spec"] >= GLOBAL_TARGET_SPEC]
    valid.sort(key=lambda r: -r["sens"])
    if valid:
        best = valid[0]
        print(f"\n## OPTIMAL  (highest sens with global spec ≥ {GLOBAL_TARGET_SPEC})")
        print(f"   per-rule spec floor = {best['spec_floor']}")
        print(f"   wide-QRS veto V = {best['V']} ms")
        print(f"   k = {best['k']} RR rules")
        print(f"   composite sens = {best['sens']:.3f}   spec = {best['spec']:.3f}")
        print("   per-class:")
        for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
            if c == "AFib":
                print(f"     AFib: {best['sens']*100:5.1f}%")
            else:
                print(f"     {c:>5}: {best.get(f'fpr_{c}',0)*100:5.1f}%")
        print("\n   RR rules:")
        for i, r in enumerate(best["chosen"], 1):
            db = f"  [T={r['T']}]" if r["T"] != 0 else ""
            print(f"     R{i}:  {r['name']} {r['sign']} {r['thr']:.4g}{db}")
        if best["V"] is not None:
            print(f"\n   Width veto:  AND max(qrs_width) < {best['V']} ms")

    print(f"\n## Top 8 configurations by sens (global spec ≥ {GLOBAL_TARGET_SPEC})")
    for r in valid[:8]:
        print(f"   spec_floor={r['spec_floor']:.2f}  V={str(r['V']):>4}  "
              f"sens={r['sens']:.3f}  spec={r['spec']:.3f}  k={r['k']:>2}  "
              f"PVC={r['fpr_PVC']:.2f}  AVB3={r['fpr_AVB3']:.2f}")


if __name__ == "__main__":
    main()
