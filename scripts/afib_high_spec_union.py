"""High-spec greedy-union composite for AFib detection (v2).

Each candidate rule (feature, T, threshold, direction) must hit spec ≥ 95%.
Greedy union under global FP budget; objective per round is to maximize
new_TP / (1 + new_FP) so the picked rule complements the existing pool.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scripts.afib_deadband_sweep import (
    FEATURES, T_GRID, build_rr_cache,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")

os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)

PER_RULE_SPEC_FLOOR = 0.95
GLOBAL_FP_BUDGET = 0.05    # composite spec ≥ 0.95


def enumerate_rules(scores: np.ndarray, y: np.ndarray, name: str, T: int,
                    spec_floor: float):
    """Return every (sign, threshold) with spec ≥ spec_floor and at least 1 TP."""
    out = []
    for sign in ("≥", "≤"):
        s_use = scores if sign == "≥" else -scores
        # Try thresholds at every unique score value, plus midpoints to capture
        # tie-resolutions cleanly.
        candidate_thrs = np.unique(s_use)
        for t in candidate_thrs:
            pred = s_use >= t
            tp_mask = pred & (y == 1)
            fp_mask = pred & (y == 0)
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            sens = tp_mask.sum() / max(1, n_pos)
            spec = 1 - fp_mask.sum() / max(1, n_neg)
            if spec < spec_floor or tp_mask.sum() == 0:
                continue
            raw_thr = float(t if sign == "≥" else -t)
            out.append(dict(
                name=name, T=int(T), sign=sign, thr=raw_thr,
                tp_mask=tp_mask, fp_mask=fp_mask,
                sens=float(sens), spec=float(spec),
                n_tp=int(tp_mask.sum()), n_fp=int(fp_mask.sum()),
            ))
    return out


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_arrays = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    y = df["y"].values
    cls = df["class"].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    fp_budget = int(np.floor(GLOBAL_FP_BUDGET * n_neg))
    print(f"## Setup: n={len(y)} pos={n_pos} neg={n_neg} "
          f"per-rule spec ≥ {PER_RULE_SPEC_FLOOR}  "
          f"global FP budget = {fp_budget} (≈ spec ≥ {1 - GLOBAL_FP_BUDGET})")

    print("\n## Enumerating candidate rules…")
    pool: list[dict] = []
    for name, fn in FEATURES.items():
        for T in T_GRID:
            scores = np.array([fn(rr, T) for rr in rr_arrays])
            cands = enumerate_rules(scores, y, name, T, PER_RULE_SPEC_FLOOR)
            pool.extend(cands)
    print(f"   candidate rules: {len(pool)}")
    if not pool:
        print("   no rules passed spec floor!")
        return

    # Sort pool by sens descending for deterministic tie-break inside greedy.
    pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
    top5 = pool[:5]
    print("\n## Top-5 rules by individual TP count")
    for r in top5:
        print(f"   {r['name']:<16}  T={r['T']:>2}  {r['name']} {r['sign']} "
              f"{r['thr']:.4g}  TP={r['n_tp']}  FP={r['n_fp']}  "
              f"sens={r['sens']:.3f}  spec={r['spec']:.3f}")

    # ----- Greedy union with complementarity objective ------------------------
    print(f"\n## Greedy OR-union (objective: new_TP / (1 + new_FP), "
          f"budget ≤ {fp_budget} FP)")
    chosen: list[dict] = []
    used_names: set[tuple[str, int]] = set()  # dedup by (name, T) so each feature × T appears at most once
    union_tp = np.zeros(len(y), dtype=bool)
    union_fp = np.zeros(len(y), dtype=bool)

    while True:
        best_score = 0.0
        best_rule = None
        for r in pool:
            if (r["name"], r["T"]) in used_names:
                continue
            new_tp = int((r["tp_mask"] & ~union_tp).sum())
            new_fp = int((r["fp_mask"] & ~union_fp).sum())
            if new_tp == 0:
                continue
            # composite budget check
            if (union_fp | r["fp_mask"]).sum() > fp_budget:
                continue
            score = new_tp / (1.0 + new_fp)
            if score > best_score:
                best_score = score
                best_rule = r
                best_rule_new_tp = new_tp
                best_rule_new_fp = new_fp
        if best_rule is None:
            break
        chosen.append(best_rule)
        used_names.add((best_rule["name"], best_rule["T"]))
        union_tp |= best_rule["tp_mask"]
        union_fp |= best_rule["fp_mask"]
        sens = union_tp.sum() / n_pos
        spec = 1 - union_fp.sum() / n_neg
        print(f"   + [{best_rule['name']:<16}] T={best_rule['T']:>2} "
              f"{best_rule['sign']} {best_rule['thr']:.4g}  "
              f"+{best_rule_new_tp} TP / +{best_rule_new_fp} FP  "
              f"→ composite sens={sens:.3f} spec={spec:.3f} "
              f"FP={int(union_fp.sum())}/{fp_budget}")

    print(f"\n## FINAL COMPOSITE  ({len(chosen)} rules)\n")
    print(f"AFib  ⟺  ANY of:")
    for i, r in enumerate(chosen, 1):
        print(f"  R{i}: {r['name']} {r['sign']} {r['thr']:.4g}     "
              f"[deadband T={r['T']}ms]")

    sens = union_tp.sum() / n_pos
    spec = 1 - union_fp.sum() / n_neg
    print(f"\n   composite sens = {sens:.3f}   spec = {spec:.3f}  "
          f"({int(union_tp.sum())}/{n_pos} TP, {int(union_fp.sum())}/{n_neg} FP)")

    pred = union_tp | union_fp
    print("\n   per-class predicted-as-AFib rate:")
    for c in ["AFib", "NSR", "PVC", "AVB2", "AVB3"]:
        mask = cls == c
        if mask.any():
            rate = pred[mask].mean() * 100
            n_caught = int(pred[mask].sum())
            print(f"     {c:>5}: {rate:5.1f}%  ({n_caught}/{mask.sum()})")

    # Attribution
    print("\n   AFib coverage attribution:")
    union = np.zeros_like(y, dtype=bool)
    for i, r in enumerate(chosen, 1):
        unique_to = int((r["tp_mask"] & ~union).sum())
        union |= r["tp_mask"]
        print(f"     R{i}: catches {r['n_tp']:>2} TP total "
              f"({unique_to:>2} unique given earlier rules)")

    fn_idx = np.where((y == 1) & ~union_tp)[0]
    print(f"\n   {len(fn_idx)} AFib still missed:")
    for i in fn_idx[:20]:
        rr = rr_arrays[i]
        npz = int(df.iloc[i]["npz_idx"])
        print(f"     npz={npz:>6}  n={len(rr):>2}  meanRR={rr.mean():.0f}  "
              f"RR=[" + " ".join(f"{int(round(v))}" for v in rr) + "]")
    if len(fn_idx) > 20:
        print(f"     ... and {len(fn_idx) - 20} more")

    fp_idx = np.where((y == 0) & union_fp)[0]
    print(f"\n   {len(fp_idx)} non-AFib falsely flagged:")
    for i in fp_idx:
        rr = rr_arrays[i]
        npz = int(df.iloc[i]["npz_idx"])
        c = df.iloc[i]["class"]
        print(f"     [{c}] npz={npz:>6}  n={len(rr):>2}  meanRR={rr.mean():.0f}  "
              f"RR=[" + " ".join(f"{int(round(v))}" for v in rr) + "]")


if __name__ == "__main__":
    main()
