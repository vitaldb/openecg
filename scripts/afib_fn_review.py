"""Visualize AFib false negatives from the v6 composite (masking + V=125).

For each missed AFib window:
  - Plot lead-II ECG with R-peaks and per-beat QRS width annotations
  - Show RR / dRR series + the masked RR
  - Print cosen, sarkar_fill, dom_cluster, rmssd, pRR_rel8 values + thresholds
Grouped into a multi-page PNG so we can scan patterns quickly.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openecg.qrs import detect_qrs
from openecg.lydus import load_signal, FS_NATIVE
from scripts.afib_deadband_sweep import FEATURES, build_rr_cache
from scripts.afib_width_masking_v5 import mask_wide_related_rr
from scripts.afib_qrs_width_v4 import enumerate_rules_veto, greedy_union

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)


VETO_V_MS = 125     # v6 best operating point
LEAD_II = 1


def build_v6_composite(rr_full, w_arrays, y, max_w):
    """Reproduce the v6 composite (masking + veto V=125) and return rule list,
    plus the per-window prediction mask."""
    rr_msk = []
    for rr, w in zip(rr_full, w_arrays):
        rr_m = mask_wide_related_rr(rr, w)
        if len(rr_m) < 4:
            rr_m = rr
        rr_msk.append(rr_m)

    veto = max_w >= VETO_V_MS
    pool = []
    for name, fn in FEATURES.items():
        for T in (0, 5, 10, 15, 20, 30, 40, 50, 70):
            sc = np.array([fn(rr, T) for rr in rr_msk])
            pool.extend(enumerate_rules_veto(sc, y, name, T, 0.95, veto))
    pool.sort(key=lambda r: (-r["n_tp"], r["n_fp"]))
    n_neg_kept = int(((y == 0) & ~veto).sum())
    fp_budget = int(np.floor(0.05 * n_neg_kept))
    chosen, tp, fp = greedy_union(pool, y, veto, fp_budget)
    return chosen, tp, fp, rr_msk


def main():
    cache_path = Path("logs/afib_rr_cache.parquet")
    df = build_rr_cache(cache_path)
    df = df[df["rr_ms"].apply(lambda x: len(x) >= 4)].reset_index(drop=True)
    rr_full = [np.asarray(r, dtype=np.float64) for r in df["rr_ms"]]
    w_arrs = [np.asarray(w, dtype=np.float64) for w in df["widths_ms"]]
    y = df["y"].values
    max_w = np.array([float(np.max(w)) if len(w) else 0.0 for w in w_arrs])

    chosen, tp, fp, rr_msk = build_v6_composite(rr_full, w_arrs, y, max_w)
    print(f"## v6 composite (V={VETO_V_MS}): {len(chosen)} rules")
    for i, r in enumerate(chosen, 1):
        db = f"  [T={r['T']}]" if r["T"] != 0 else ""
        print(f"   R{i}: {r['name']} {r['sign']} {r['thr']:.4g}{db}")
    n_pos = int((y == 1).sum())
    print(f"   sens = {tp[y==1].sum()}/{n_pos}  spec = "
          f"{1 - fp[y==0].sum() / max(1, (y==0).sum()):.3f}")

    # AFib FN: y=1 and not predicted
    pred = tp | fp
    fn_mask = (y == 1) & ~pred
    fn_idx = np.where(fn_mask)[0]
    print(f"\n## AFib FN: {len(fn_idx)} cases  (missed by v6)")

    # Categorize FNs by why they were missed
    # Reason 1: vetoed (max_w >= V) — they're AFib with wide-QRS aberrancy
    # Reason 2: masking made residual too short / regular
    # Reason 3: borderline cosen (1.5-2.0) — RR signature subtle
    print(f"\n   FN broken down:")
    categories = {"vetoed_by_width": [], "borderline_chaos": [], "regular_looking": []}
    for i in fn_idx:
        rr = rr_full[i]
        rr_m = rr_msk[i]
        # was the case vetoed?
        if max_w[i] >= VETO_V_MS:
            categories["vetoed_by_width"].append(i)
        else:
            # compute cosen on masked RR
            cosen_v = FEATURES["cosen"](rr_m, 30)
            cv = np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else 0
            if cv < 0.10 and cosen_v < 1.5:
                categories["regular_looking"].append(i)
            else:
                categories["borderline_chaos"].append(i)
    for k, idxs in categories.items():
        print(f"     {k:<20} n={len(idxs)}")
        for i in idxs:
            print(f"        npz={int(df.iloc[i]['npz_idx']):>6}  "
                  f"n_beats={len(rr_full[i]):>2}  "
                  f"max_w={max_w[i]:.0f}ms  "
                  f"meanRR={np.mean(rr_full[i]):.0f}")

    # Build per-FN visualization page (6 plots per page)
    fn_npz = [int(df.iloc[i]["npz_idx"]) for i in fn_idx]
    out_dir = Path("logs/afib_fn_viz")
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n## Plotting FN cases → {out_dir}/")
    per_page = 6
    page = 0
    cosen_thrs = sorted(set(r["thr"] for r in chosen if r["name"] == "cosen"))

    for start in range(0, len(fn_idx), per_page):
        page += 1
        batch = fn_idx[start:start + per_page]
        fig, axes = plt.subplots(len(batch), 1, figsize=(14, 2.6 * len(batch)),
                                 squeeze=False)
        for row, i in enumerate(batch):
            ax = axes[row, 0]
            npz = int(df.iloc[i]["npz_idx"])
            sig = load_signal(npz, lead_idx=LEAD_II, fs_target=FS_NATIVE)
            t = np.arange(len(sig)) / FS_NATIVE
            ax.plot(t, sig, color="#222", linewidth=0.7)
            peaks, widths = detect_qrs(sig, FS_NATIVE, return_widths=True)
            # mark R peaks
            ax.scatter(peaks / FS_NATIVE, sig[peaks], color="red", s=20, zorder=5)
            # annotate widths
            for p, w in zip(peaks, widths):
                color = "darkred" if w >= VETO_V_MS else "blue"
                ax.text(p / FS_NATIVE, sig[p] * 1.05 + 0.5,
                        f"{int(w)}ms", color=color, fontsize=7, ha="center")
            # rule-relevant features
            rr = rr_full[i]
            rr_m = rr_msk[i]
            cosen_v = FEATURES["cosen"](rr_m, 30)
            sarkar_v = FEATURES["sarkar_fill"](rr_m, 15)
            dom_v = FEATURES["dom_cluster"](rr_m, 70)
            cv = np.std(rr) / np.mean(rr) if len(rr) > 0 else 0
            cls_label = df.iloc[i]["class"]
            ax.set_title(
                f"npz={npz}  [{cls_label}]  "
                f"n_beats={len(peaks)}  meanRR={np.mean(rr):.0f}ms  CV={cv:.2f}  "
                f"max_w={max_w[i]:.0f}ms  "
                f"cosen[msk]={cosen_v:.2f}(<{min(cosen_thrs):.2f})  "
                f"sarkar={sarkar_v:.3f}  dom_cluster={dom_v:.2f}",
                fontsize=9, loc="left",
            )
            ax.set_xlim(0, 10)
            ax.set_ylabel("lead II")
            ax.grid(alpha=0.3)
            # RR overlay (small)
            if len(rr) > 0:
                rr_txt = "RR=[" + " ".join(f"{int(round(v))}" for v in rr) + "]"
                ax.text(0.01, 0.95, rr_txt, transform=ax.transAxes,
                        fontsize=7, va="top", family="monospace",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        out = out_dir / f"fn_page_{page:02d}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"   saved {out}  ({len(batch)} cases)")

    # Summary stats: feature distribution of FN vs catchable AFib
    print(f"\n## Distribution comparison: FN vs caught AFib")
    caught_idx = np.where(tp & (y == 1))[0]
    print(f"{'feature':<16}{'FN median':>12}{'caught median':>16}{'gap':>8}")
    for feat in ("cosen", "sarkar_fill", "dom_cluster", "pRR_rel8", "rmssd_ms"):
        T = {"cosen": 30, "sarkar_fill": 15, "dom_cluster": 70,
             "pRR_rel8": 0, "rmssd_ms": 0}[feat]
        sc = np.array([FEATURES[feat](rr_msk[i], T) for i in range(len(rr_msk))])
        fn_med = np.median(sc[fn_idx])
        ct_med = np.median(sc[caught_idx])
        print(f"  {feat:<16}{fn_med:>12.3f}{ct_med:>16.3f}{ct_med - fn_med:>+8.3f}")


if __name__ == "__main__":
    main()
