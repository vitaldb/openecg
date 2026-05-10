"""Render B4 (label-noise suspects) and B5 (ambiguous) AFib FN cases.

These are AFib-labelled windows in Lydus that look like NSR + APC or
NSR + occasional outlier, not classical AFib. Save high-resolution
panels with lead I + lead II + annotated RR / widths, intended for
clinical chart review.

If the user confirms these aren't AFib, Lydus's next version should
re-label them. We print the npz_idx and rid so they can be found.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openecg.qrs import detect_qrs
from openecg.lydus import load_signal, FS_NATIVE
from scripts.afib_deadband_sweep import FEATURES

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault(
    "OPENECG_LYDUS_DIR", "G:/Shared drives/datasets/ecg/lydus_ecg"
)
LYDUS_DIR = os.environ["OPENECG_LYDUS_DIR"]

# B4: label-noise suspects (RR pattern looks like NSR + APC)
B4_NPZ = [101420, 36019, 156875, 126526]
# B5: visually-ambiguous cluster patterns
B5_NPZ = [58269, 123285]


def fetch_lydus_metadata(npz_indices):
    con = duckdb.connect(str(Path(LYDUS_DIR) / "lydus_ecg.duckdb"), read_only=True)
    placeholders = ",".join(str(i) for i in npz_indices)
    rows = con.execute(
        f"SELECT npz_idx, rid, rhythm, premature_beat, bbb, avb, vrate, arate, "
        f"pri, qrsd, conclusion, age, sex FROM records WHERE npz_idx IN ({placeholders})"
    ).fetchall()
    cols = [d[0] for d in con.description]
    return {int(r[0]): dict(zip(cols, r)) for r in rows}


def render_one(npz_idx: int, meta: dict, out_path: Path, category: str):
    # Load lead I, II, V1 if available
    lead_names = ["I (idx 0)", "II (idx 1)", "V1 (idx 3)"]
    lead_idxs = [0, 1, 3]
    sigs = [load_signal(npz_idx, lead_idx=li, fs_target=FS_NATIVE) for li in lead_idxs]

    # QRS detect on lead II (our reference)
    peaks, widths = detect_qrs(sigs[1], FS_NATIVE, return_widths=True)
    rr_ms = np.diff(peaks) * (1000.0 / FS_NATIVE)

    # Compute features
    feat_vals = {
        "cosen[T=30]": FEATURES["cosen"](rr_ms, 30),
        "cosen[T=40]": FEATURES["cosen"](rr_ms, 40),
        "sarkar_fill[T=15]": FEATURES["sarkar_fill"](rr_ms, 15),
        "dom_cluster[T=70]": FEATURES["dom_cluster"](rr_ms, 70),
        "pRR_rel8": FEATURES["pRR_rel8"](rr_ms, 0),
        "rmssd": FEATURES["rmssd_ms"](rr_ms, 0),
        "cv_rr": FEATURES["cv_rr"](rr_ms, 0),
    }

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 1.5, 1.5], hspace=0.4)

    # Header
    title = (f"npz={npz_idx}   rid={meta.get('rid', '?')}   "
             f"category={category}\n"
             f"Lydus label: rhythm={meta.get('rhythm')}  "
             f"premature_beat={meta.get('premature_beat')}  "
             f"avb={meta.get('avb')}  bbb={meta.get('bbb')}\n"
             f"age={meta.get('age')}  sex={meta.get('sex')}  "
             f"vrate={meta.get('vrate')}  arate={meta.get('arate')}  "
             f"qrsd={meta.get('qrsd')}  PR={meta.get('pri')}")
    fig.suptitle(title, fontsize=10, y=0.985, ha="left", x=0.05)

    t = np.arange(len(sigs[0])) / FS_NATIVE
    for ax_idx, (lead_name, sig) in enumerate(zip(lead_names, sigs)):
        ax = fig.add_subplot(gs[ax_idx, 0])
        ax.plot(t, sig, color="#222", linewidth=0.7)
        ax.scatter(peaks / FS_NATIVE, sig[peaks], color="red", s=30, zorder=5)
        for p, w in zip(peaks, widths):
            color = "darkred" if w >= 125 else "blue"
            ax.text(p / FS_NATIVE, sig[p] + (sig.max() - sig.min()) * 0.07,
                    f"{int(w)}ms", color=color, fontsize=8, ha="center")
        ax.set_xlim(0, 10)
        ax.set_ylabel(f"lead {lead_name}")
        ax.grid(alpha=0.3)
        ax.set_xticks(np.arange(0, 10.5, 0.5), minor=True)
        ax.grid(which="minor", alpha=0.15)
    ax.set_xlabel("time (s)")

    # RR / features text panel
    ax_txt = fig.add_subplot(gs[3, 0])
    ax_txt.axis("off")
    rr_str = "RR(ms) = [" + " ".join(f"{int(round(v))}" for v in rr_ms) + "]"
    dr_str = "dRR(ms)= [" + " ".join(f"{int(round(v))}" for v in np.diff(rr_ms)) + "]"
    w_str  = "QRSw(ms)=[" + " ".join(f"{int(round(v))}" for v in widths) + "]"
    ax_txt.text(0.0, 0.95, rr_str, fontsize=9, family="monospace", va="top",
                transform=ax_txt.transAxes)
    ax_txt.text(0.0, 0.55, dr_str, fontsize=9, family="monospace", va="top",
                transform=ax_txt.transAxes)
    ax_txt.text(0.0, 0.15, w_str, fontsize=9, family="monospace", va="top",
                transform=ax_txt.transAxes)

    # Feature values
    ax_feat = fig.add_subplot(gs[4, 0])
    ax_feat.axis("off")
    lines = [f"n_beats={len(peaks)}", f"meanRR={np.mean(rr_ms):.0f}ms",
             f"max_w={float(np.max(widths)):.0f}ms",
             f"n_wide(≥120)={int((widths >= 120).sum())}/{len(widths)}"]
    for k, v in feat_vals.items():
        lines.append(f"{k}={v:.3f}")
    ax_feat.text(0.0, 0.9, "  ".join(lines), fontsize=9, family="monospace",
                 va="top", transform=ax_feat.transAxes)

    # Conclusion text if any
    conclusion = meta.get("conclusion") or ""
    if conclusion:
        ax_feat.text(0.0, 0.3,
                     f"Lydus conclusion: {conclusion[:200]}",
                     fontsize=9, color="#444", va="top",
                     transform=ax_feat.transAxes,
                     wrap=True)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path("logs/afib_label_review")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for npz in B4_NPZ:
        targets.append((npz, "B4-label-noise-suspect"))
    for npz in B5_NPZ:
        targets.append((npz, "B5-ambiguous"))

    metas = fetch_lydus_metadata([t[0] for t in targets])
    print(f"## Rendering {len(targets)} label-review cases → {out_dir}/")
    for npz, cat in targets:
        meta = metas.get(npz, {})
        out_path = out_dir / f"{cat}_npz{npz}_rid{meta.get('rid', 'x')}.png"
        render_one(npz, meta, out_path, cat)
        print(f"  {out_path.name}  rhythm={meta.get('rhythm')}  "
              f"pb={meta.get('premature_beat')}  conclusion={(meta.get('conclusion') or '')[:60]}")

    # Print summary table
    print("\n## Summary (for clinical chart review)")
    print(f"  {'npz':>7}  {'rid':>6}  {'category':<25}  {'rhythm':<10}  "
          f"{'pb':<14}  conclusion (truncated)")
    for npz, cat in targets:
        m = metas.get(npz, {})
        concl = (m.get("conclusion") or "")[:60]
        print(f"  {npz:>7}  {m.get('rid', 0):>6}  {cat:<25}  "
              f"{str(m.get('rhythm')):<10}  {str(m.get('premature_beat')):<14}  {concl}")


if __name__ == "__main__":
    main()
