# scripts/diagnose_v3.py
"""Visual error analysis: where does v3 fail vs v1.0 on LUDB val?

For each LUDB val sequence (subset for speed) we run both v1.0 and v3 inference
and rank by per-sequence QRS-F1 drop (v3 worse than v1.0). For the worst
offenders we print an ASCII frame-by-frame diff and save a matplotlib panel
to ``out/diag_v3_<rid>_<lead>.png``.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecgcode import eval as ee, ludb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.infer import load_model, predict_frames

V1_CKPT = "data/checkpoints/stage2_v1.pt"
V3_CKPT = "data/checkpoints/stage2_v3.pt"
OUT_DIR = Path("out")

CHAR = {0: "_", 1: "p", 2: "r", 3: "t"}  # super: 0 OTHER, 1 P, 2 QRS, 3 T
CMAP = {0: "lightgray", 1: "red", 2: "blue", 3: "green"}


def render(frames):
    return "".join(CHAR[int(c)] for c in frames)


def confusion(pred, true):
    """Return 4x4 confusion matrix (rows=true, cols=pred)."""
    cm = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...", flush=True)
    m1 = load_model(V1_CKPT, device=device)  # default dims (d_model=64, n_layers=4)
    m3 = load_model(V3_CKPT, device=device, d_model=128, n_layers=8)

    print("Loading LUDB val...", flush=True)
    val_ids = ludb.load_split()["val"]
    # Use a subset of records (each = 12 leads = 12 sequences) for speed.
    ds = LUDBFrameDataset(val_ids[:15])
    print(f"  {len(ds)} sequences cached", flush=True)

    # Find sequences where v3 is much worse than v1.0
    diffs = []
    for idx in range(len(ds)):
        rid, lead = ds.items[idx]
        sig, lead_idx, true_frames = ds.cache[(rid, lead)]
        # LUDBFrameDataset stores supercategory frames already (via gt_to_super_frames).
        true_super = true_frames.astype(np.uint8)
        # FrameClassifier expects exactly 2500 samples
        if len(sig) < 2500:
            continue
        sig = sig[:2500]
        true_super = true_super[:500]
        p1 = predict_frames(m1, sig, lead_idx, device=device)
        p3 = predict_frames(m3, sig, lead_idx, device=device)
        n = min(len(true_super), len(p1), len(p3))
        f1_v1 = ee.frame_f1(p1[:n], true_super[:n])
        f1_v3 = ee.frame_f1(p3[:n], true_super[:n])
        # Sort key: macro drop across P+QRS+T (focus on physiological classes).
        macro_v1 = (f1_v1[ee.SUPER_P]["f1"] + f1_v1[ee.SUPER_QRS]["f1"] + f1_v1[ee.SUPER_T]["f1"]) / 3
        macro_v3 = (f1_v3[ee.SUPER_P]["f1"] + f1_v3[ee.SUPER_QRS]["f1"] + f1_v3[ee.SUPER_T]["f1"]) / 3
        drop_macro = macro_v1 - macro_v3
        drop_qrs = f1_v1[ee.SUPER_QRS]["f1"] - f1_v3[ee.SUPER_QRS]["f1"]
        drop_p = f1_v1[ee.SUPER_P]["f1"] - f1_v3[ee.SUPER_P]["f1"]
        drop_t = f1_v1[ee.SUPER_T]["f1"] - f1_v3[ee.SUPER_T]["f1"]
        diffs.append({
            "drop_macro": drop_macro,
            "drop_qrs": drop_qrs, "drop_p": drop_p, "drop_t": drop_t,
            "rid": rid, "lead": lead,
            "p1": p1[:n], "p3": p3[:n], "true": true_super[:n],
            "sig": sig, "f1_v1": f1_v1, "f1_v3": f1_v3,
        })

    # Aggregate stats
    avg_drop_p = np.mean([d["drop_p"] for d in diffs])
    avg_drop_qrs = np.mean([d["drop_qrs"] for d in diffs])
    avg_drop_t = np.mean([d["drop_t"] for d in diffs])
    print(f"\nAverage F1 drop (v1.0 - v3) across {len(diffs)} sequences:", flush=True)
    print(f"  P drop:   {avg_drop_p:+.3f}", flush=True)
    print(f"  QRS drop: {avg_drop_qrs:+.3f}", flush=True)
    print(f"  T drop:   {avg_drop_t:+.3f}", flush=True)

    diffs.sort(key=lambda d: d["drop_macro"], reverse=True)
    print("\n=== Top 5 sequences where v3 underperforms v1.0 (by macro PQRS T F1 drop) ===\n", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    cm_v3_total = np.zeros((4, 4), dtype=np.int64)
    cm_v1_total = np.zeros((4, 4), dtype=np.int64)
    for d in diffs[:5]:
        rid, lead = d["rid"], d["lead"]
        true_super = d["true"]; p1 = d["p1"]; p3 = d["p3"]; sig = d["sig"]
        f1_v1 = d["f1_v1"]; f1_v3 = d["f1_v3"]
        n = len(true_super)
        print(
            f"\n--- Record {rid}, lead {lead} "
            f"(macro drop={d['drop_macro']:+.3f} | "
            f"P={d['drop_p']:+.3f} QRS={d['drop_qrs']:+.3f} T={d['drop_t']:+.3f}) ---",
            flush=True,
        )
        # ASCII rendering: print in 100-frame (=2s) chunks for readability
        for c0 in range(0, n, 100):
            c1 = min(c0 + 100, n)
            print(f"  frames {c0:3d}-{c1-1:3d}", flush=True)
            print(f"    GT  : {render(true_super[c0:c1])}", flush=True)
            print(f"    v1.0: {render(p1[c0:c1])}", flush=True)
            print(f"    v3  : {render(p3[c0:c1])}", flush=True)
        print(
            f"  v1.0 F1: P={f1_v1[1]['f1']:.3f} QRS={f1_v1[2]['f1']:.3f} T={f1_v1[3]['f1']:.3f}",
            flush=True,
        )
        print(
            f"  v3   F1: P={f1_v3[1]['f1']:.3f} QRS={f1_v3[2]['f1']:.3f} T={f1_v3[3]['f1']:.3f}",
            flush=True,
        )
        cm_v1 = confusion(p1, true_super)
        cm_v3 = confusion(p3, true_super)
        print(f"  Confusion v1 (rows=true,cols=pred):\n{cm_v1}", flush=True)
        print(f"  Confusion v3 (rows=true,cols=pred):\n{cm_v3}", flush=True)

        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
        t_sig = np.arange(len(sig)) / 250.0
        axes[0].plot(t_sig, sig, color="black", linewidth=0.6)
        axes[0].set_ylabel("ECG (z-norm)")
        axes[0].set_title(
            f"Record {rid} lead {lead} | macro PQRS T F1 drop = {d['drop_macro']:+.3f} | "
            f"v1.0 F1 P/QRS/T={f1_v1[1]['f1']:.2f}/{f1_v1[2]['f1']:.2f}/{f1_v1[3]['f1']:.2f}  "
            f"v3 F1 P/QRS/T={f1_v3[1]['f1']:.2f}/{f1_v3[2]['f1']:.2f}/{f1_v3[3]['f1']:.2f}"
        )

        for ax, frames, label in [
            (axes[1], true_super, "GT"),
            (axes[2], p1, "v1.0"),
            (axes[3], p3, "v3"),
        ]:
            for f_idx, c in enumerate(frames):
                color = CMAP[int(c)]
                ax.axvspan(f_idx * 0.02, (f_idx + 1) * 0.02, color=color, alpha=0.7)
            ax.set_ylabel(label)
            ax.set_yticks([])
            ax.set_ylim(0, 1)

        axes[3].set_xlabel("Time (s)")
        plt.tight_layout()
        out_path = OUT_DIR / f"diag_v3_{rid}_{lead}.png"
        plt.savefig(out_path, dpi=80)
        plt.close(fig)
        print(f"  -> saved {out_path}", flush=True)

    # Aggregate confusion across the whole sample
    for d in diffs:
        cm_v1_total += confusion(d["p1"], d["true"])
        cm_v3_total += confusion(d["p3"], d["true"])
    print("\n=== Aggregate confusion (rows=true, cols=pred) ===", flush=True)
    print(f"Order: 0=other 1=P 2=QRS 3=T", flush=True)
    print(f"v1.0:\n{cm_v1_total}", flush=True)
    print(f"v3:\n{cm_v3_total}", flush=True)

    # Per-lead aggregate macro-F1 drop (v1.0 - v3) to confirm lead-specific failure.
    print("\n=== Per-lead macro F1 drop (v1.0 minus v3), averaged across records ===", flush=True)
    by_lead = {}
    for d in diffs:
        by_lead.setdefault(d["lead"], []).append(d["drop_macro"])
    for lead in ludb.LEADS_12:
        drops = by_lead.get(lead, [])
        if not drops:
            continue
        print(
            f"  {lead:5s}: mean drop = {np.mean(drops):+.3f}  "
            f"(max = {np.max(drops):+.3f}, n={len(drops)})",
            flush=True,
        )

    # Per-class diff: what does v3 do extra that v1.0 doesn't?
    print("\nPer-true-class breakdown of v3 vs v1.0 (predictions on each true class):", flush=True)
    for true_cls in range(4):
        v1_row = cm_v1_total[true_cls]
        v3_row = cm_v3_total[true_cls]
        total = v1_row.sum()
        if total == 0:
            continue
        print(
            f"  True={ee.SUPER_NAMES[true_cls]} (n={total}):\n"
            f"    v1.0 pred dist: other={v1_row[0]/total:.2%} P={v1_row[1]/total:.2%} "
            f"QRS={v1_row[2]/total:.2%} T={v1_row[3]/total:.2%}\n"
            f"    v3   pred dist: other={v3_row[0]/total:.2%} P={v3_row[1]/total:.2%} "
            f"QRS={v3_row[2]/total:.2%} T={v3_row[3]/total:.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
