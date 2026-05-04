"""ISP label sanity check: are ISP annotations using same convention as LUDB?

Why this matters: G (combined model) trained on 4836 ISP sequences and still
got F1=0.15-0.35 on ISP test. F (LUDB only) got near-zero on ISP test. Either:
  (a) ISP labels are subtly broken (different convention, sample-index offset, etc.)
  (b) ISP signals have characteristics LUDB-trained model can't generalize to
  (c) Both

Steps:
  1. Distribution stats: per-class fraction in LUDB val vs ISP test (should be similar
     if same task; large differences suggest convention mismatch)
  2. Pick 3 ISP test records, render GT vs F prediction side-by-side
  3. Check boundary convention: do P_on/P_off in ISP look like P-wave start/end
     when overlaid on signal?
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ecgcode import eval as ee
from ecgcode import isp, ludb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.multi_dataset import CombinedFrameDataset
from ecgcode.stage2.train import load_checkpoint

OUT_DIR = Path("out") / "diag_isp"
F_CKPT = Path("data/checkpoints/stage2_v4_ludb_only.pt")
CMAP = {0: "lightgray", 1: "red", 2: "blue", 3: "green"}
NAMES = {0: "other", 1: "P", 2: "QRS", 3: "T"}


def class_distribution(ds, name):
    """Return per-class frame fraction across the entire dataset."""
    counts = np.zeros(4, dtype=np.int64)
    for idx in range(len(ds)):
        _, _, labels = ds[idx]
        for c in range(4):
            counts[c] += int((labels.numpy() == c).sum())
    total = counts.sum()
    fracs = counts / total
    print(f"\n{name} (n={len(ds)}, total_frames={total}):", flush=True)
    for c in range(4):
        print(f"  {NAMES[c]:6s}: {fracs[c]*100:6.2f}%  ({counts[c]:>9,d})", flush=True)
    return {NAMES[c]: float(fracs[c]) for c in range(4)}


def predict(model, sig, lead_id, device):
    model.train(False)
    with torch.no_grad():
        x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
        lid = torch.tensor([lead_id], dtype=torch.long, device=device)
        logits = model(x, lid)
        return logits.argmax(dim=-1).cpu().numpy().squeeze(0).astype(np.uint8)


def plot_record(rid, lead, sig, gt_frames, pred_frames, out_path, fs=250):
    """Render signal + GT + prediction stripes for visual sanity check."""
    n_samples = len(sig)
    n_frames = len(gt_frames)
    t_sig = np.arange(n_samples) / fs
    fig, axes = plt.subplots(3, 1, figsize=(16, 5),
                             gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
    axes[0].plot(t_sig, sig, color="black", linewidth=0.6)
    axes[0].set_ylabel("ECG (z-norm)")
    axes[0].set_title(f"ISP test rid={rid} lead={lead}  |  GT vs F (LUDB-only) prediction")
    frame_dt = 1.0 / 50.0  # 50Hz frames = 20ms
    for ax, frames, label in [(axes[1], gt_frames, "GT"), (axes[2], pred_frames, "F pred")]:
        for f_idx, c in enumerate(frames):
            ax.axvspan(f_idx * frame_dt, (f_idx + 1) * frame_dt, color=CMAP[int(c)], alpha=0.7)
        ax.set_ylabel(label)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("\nLoading datasets...", flush=True)
    ludb_val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    isp_test_ds = CombinedFrameDataset(["isp_test"])
    ludb_train_ds = LUDBFrameDataset(ludb.load_split()["train"][:30])  # subset for stats

    # Step 1: class distributions
    print("\n=== Per-class frame distribution ===", flush=True)
    dist_ludb_train = class_distribution(ludb_train_ds, "LUDB train (30 records)")
    dist_ludb_val = class_distribution(ludb_val_ds, "LUDB val")
    dist_isp = class_distribution(isp_test_ds, "ISP test")

    # Step 2: load F model and visualize
    print("\nLoading F (LUDB-only no lead_emb) model...", flush=True)
    model = FrameClassifier(d_model=64, n_layers=4, use_lead_emb=False)
    load_checkpoint(F_CKPT, model)
    model = model.to(device)

    # Step 3: pick 5 ISP test records (diverse) and render
    print("\nRendering 5 ISP test sequences (lead II) for visual inspection...", flush=True)
    rendered = 0
    for idx in range(len(isp_test_ds)):
        src = isp_test_ds.items[idx]
        if src[0] != "isp":
            continue
        if src[2] != "ii":  # use lead II for inspection
            continue
        sig, lead_id, labels = isp_test_ds.cache[idx]
        gt = labels.astype(np.uint8)
        pred = predict(model, sig, int(lead_id), device)
        n = min(len(gt), len(pred))
        gt = gt[:n]; pred = pred[:n]
        f1 = ee.frame_f1(pred, gt)
        rid = src[1]
        out_path = OUT_DIR / f"isp_{rid}_lead_ii.png"
        plot_record(rid, "ii", sig[:n*5], gt, pred, out_path)
        print(f"  rid={rid} lead=ii  pred F1 P/QRS/T = "
              f"{f1[ee.SUPER_P]['f1']:.3f} / {f1[ee.SUPER_QRS]['f1']:.3f} / "
              f"{f1[ee.SUPER_T]['f1']:.3f}  -> {out_path}", flush=True)
        rendered += 1
        if rendered >= 5:
            break

    summary = {
        "class_distribution": {
            "ludb_train": dist_ludb_train,
            "ludb_val": dist_ludb_val,
            "isp_test": dist_isp,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_DIR}/summary.json and {rendered} PNGs", flush=True)


if __name__ == "__main__":
    main()
