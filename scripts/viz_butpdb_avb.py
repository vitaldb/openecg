"""Visualize v12_reg predictions on BUT PDB AV-block records (rids 1, 3, 13, 22).

BUT PDB records are 2 min × 360 Hz × 2-lead. We decimate to 250 Hz, slide
non-overlapping 10s windows, run inference, and plot 3 representative windows
per (record, lead) showing:
  - signal
  - GT P peaks (vertical red ticks; BUT PDB has no on/off)
  - GT QRS peaks (vertical blue ticks)
  - v12_reg P/QRS/T bands

Output: out/viz_butpdb_avb/<rid>_<lead_idx>_<window>.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scipy_signal
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import butpdb
from openecg.stage2.infer import (
    apply_reg_to_boundaries, extract_boundaries,
    post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.train import load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS

CKPT = REPO / "data" / "checkpoints" / "stage2_v12_reg.pt"
OUT_DIR = REPO / "out" / "viz_butpdb_avb"
WINDOW_SAMPLES = 2500          # 10s @ 250Hz
FS = 250
FRAME_MS = 20

CMAP = {1: ("red", "P"), 2: ("blue", "QRS"), 3: ("green", "T")}
PATHO_LABEL = {1: "BII (2°AVB)", 3: "BIII (3°AVB, paced)",
               13: "BII (2°AVB)", 22: "BI (1°AVB)"}


def predict_one(model, sig, lead_idx, device):
    sig = sig.astype(np.float32)
    sig_n = ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)
    frames, reg = predict_frames_with_reg(model, sig_n, lead_idx, device=device)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    return pp


def frames_to_bands(frames, n_samples, frame_ms=FRAME_MS):
    spf = int(round(frame_ms * FS / 1000))
    bands = []
    cur_cls = None
    cur_lo = 0
    for fi, c in enumerate(frames):
        c = int(c)
        if c != cur_cls:
            if cur_cls in CMAP:
                bands.append((cur_lo, fi * spf, cur_cls))
            cur_cls = c
            cur_lo = fi * spf
    if cur_cls in CMAP:
        bands.append((cur_lo, n_samples, cur_cls))
    return bands


def plot_panel(sig, gt_p_peaks, gt_qrs_peaks, bands, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    t = np.arange(len(sig)) / FS
    ax.plot(t, sig, color="k", lw=0.6)
    # GT P peaks (top region)
    y_gt = sig.max() + 0.5
    for s in gt_p_peaks:
        ax.axvline(s / FS, ymin=0.85, ymax=0.97, color="red", lw=1.2)
    for s in gt_qrs_peaks:
        ax.axvline(s / FS, ymin=0.85, ymax=0.97, color="blue", lw=0.8, alpha=0.6)
    # Model bands (bottom strip)
    y_lo = sig.min() - 0.4
    y_hi = sig.min() - 0.1
    for s, e, c in bands:
        color = CMAP[c][0]
        ax.axvspan(s / FS, e / FS, ymin=0.02, ymax=0.10, color=color, alpha=0.6)
    # legend
    ax.plot([], [], "r|", ms=12, label="GT P peak")
    ax.plot([], [], "b|", ms=12, label="GT QRS peak", alpha=0.6)
    for cls in (1, 2, 3):
        ax.plot([], [], "s", color=CMAP[cls][0], label=f"v12_reg {CMAP[cls][1]}")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal")
    ax.set_xlim(0, len(sig) / FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model = FrameClassifierViTReg(**KWARGS, n_reg=6)
    load_checkpoint(str(CKPT), model)
    model = model.to(device).train(False)

    summary = []
    for rid in butpdb.AVB_RECORDS:
        rec = butpdb.load_record(rid)
        fs_native = rec["fs"]
        sig_full = rec["signal"]            # (n, 2)
        p_peaks_native = butpdb.load_pwave_peaks(rid)
        qrs = butpdb.load_qrs(rid)
        qrs_peaks_native = qrs["sample"]

        # Decimate each lead to 250 Hz; resample annotation positions
        if fs_native % FS == 0:
            factor = fs_native // FS
            sig_250 = np.stack(
                [scipy_signal.decimate(sig_full[:, c], factor, zero_phase=True)
                 for c in range(sig_full.shape[1])], axis=-1,
            )
        else:
            n_new = int(round(sig_full.shape[0] * FS / fs_native))
            sig_250 = np.stack(
                [scipy_signal.resample(sig_full[:, c], n_new)
                 for c in range(sig_full.shape[1])], axis=-1,
            )
        scale = sig_250.shape[0] / sig_full.shape[0]
        p_peaks_250 = (p_peaks_native * scale).astype(np.int64)
        qrs_peaks_250 = (qrs_peaks_native * scale).astype(np.int64)

        n_total = sig_250.shape[0]
        n_windows = n_total // WINDOW_SAMPLES
        # Pick 3 representative windows: first, middle, last
        chosen_windows = sorted({0, n_windows // 2, max(0, n_windows - 1)})

        for lead_idx in (0, 1):
            # Map to LUDB lead-id for the model. BUT PDB sources are MIT-BIH:
            # lead 0 typically MLII (LUDB id 1 = "ii"), lead 1 typically V1 / V5
            # — we use a generic guess: 0 -> ii (1), 1 -> v5 (10).
            model_lead_id = 1 if lead_idx == 0 else 10
            for w in chosen_windows:
                lo = w * WINDOW_SAMPLES
                hi = lo + WINDOW_SAMPLES
                sig_win = sig_250[lo:hi, lead_idx]
                if len(sig_win) < WINDOW_SAMPLES:
                    continue
                p_in = [int(s - lo) for s in p_peaks_250 if lo <= s < hi]
                q_in = [int(s - lo) for s in qrs_peaks_250 if lo <= s < hi]
                pp = predict_one(model, sig_win, model_lead_id, device)
                bands = frames_to_bands(pp, len(sig_win))
                pred_p = sum(1 for _, _, c in bands if c == 1)
                pred_qrs = sum(1 for _, _, c in bands if c == 2)
                title = (f"BUT PDB rid={rid:02d} ({PATHO_LABEL[rid]}) "
                         f"lead={lead_idx}  win={w} ({lo/FS:.0f}-{hi/FS:.0f}s)  "
                         f"GT P={len(p_in)}/QRS={len(q_in)}  pred P={pred_p}")
                out_path = OUT_DIR / f"{rid:02d}_lead{lead_idx}_win{w:02d}.png"
                plot_panel(sig_win, p_in, q_in, bands, title, out_path)
                summary.append((rid, lead_idx, w, len(p_in), len(q_in), pred_p))
                print(f"  {out_path.name}  GT P={len(p_in):>2d} QRS={len(q_in):>2d}"
                      f"  pred P={pred_p:>2d} QRS={pred_qrs:>2d}", flush=True)

    print(f"\nWrote {len(summary)} panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
