"""Side-by-side viz of v12_reg (old) vs v12_reg+synth (new) on the AV-block
records that changed the most after the synth retrain.

Targets (from compare_avb_synth_*.md):
  * BUT PDB rid=3 lead 0  (BIII paced)  — biggest regression: F1 0.54 -> 0.10
  * LUDB     rid=74 lead i (3°AVB)       — biggest gain:        1 -> 9 predicted P
  * LUDB     rid=34 lead v5 (3°AVB)      — regression:          4 -> 1 predicted P

For BUT PDB we slice the 2-min record into the same non-overlapping 10-s
windows as scripts/viz_butpdb_avb.py and pick window 0.

Output: out/viz_avb_compare/<tag>.png  (3 panels)
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

from openecg import butpdb, ludb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import (
    extract_boundaries, post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.train import load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS

CKPT_OLD = REPO / "data" / "checkpoints" / "stage2_v12_reg.pt"
CKPT_NEW = REPO / "data" / "checkpoints" / "stage2_v12_reg_with_synth.pt"
OUT_DIR = REPO / "out" / "viz_avb_compare"
WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20

CMAP = {1: "red", 2: "blue", 3: "green"}


def _load(path, device):
    m = FrameClassifierViTReg(**KWARGS, n_reg=6)
    load_checkpoint(str(path), m)
    return m.to(device).train(False)


def _run(model, sig, lead_id, device):
    sig_n = ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)
    frames, _ = predict_frames_with_reg(model, sig_n, lead_id, device=device)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    bands = []
    cur = None; lo = 0
    spf = int(round(FRAME_MS * FS / 1000))
    for fi, c in enumerate(pp):
        c = int(c)
        if c != cur:
            if cur in CMAP:
                bands.append((lo, fi * spf, cur))
            cur = c; lo = fi * spf
    if cur in CMAP:
        bands.append((lo, len(sig), cur))
    return pp, bands, bds


def plot_panel(sig, gt_p_peaks, gt_qrs_peaks, bands_old, bands_new,
               bds_old, bds_new, title, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    t = np.arange(len(sig)) / FS
    for ax, bands, bds, label in (
        (axes[0], bands_old, bds_old, "old: v12_reg"),
        (axes[1], bands_new, bds_new, "new: v12_reg + synth"),
    ):
        ax.plot(t, sig, color="k", lw=0.6)
        for s, e, c in bands:
            ax.axvspan(s / FS, e / FS, ymin=0.02, ymax=0.10,
                       color=CMAP[c], alpha=0.6)
        # GT P peaks (red ticks at top)
        for s in gt_p_peaks:
            ax.axvline(s / FS, ymin=0.85, ymax=0.97, color="red", lw=1.4)
        for s in gt_qrs_peaks:
            ax.axvline(s / FS, ymin=0.85, ymax=0.97, color="blue", lw=0.8, alpha=0.5)
        n_p_pred = len(bds.get("p_on", []))
        ax.set_ylabel(f"{label}\nP_pred={n_p_pred}")
        ax.set_xlim(0, len(sig) / FS)
    axes[1].set_xlabel("time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def case_butpdb(rid, lead_idx, model_old, model_new, device, tag, n_window_idx=0):
    rec = butpdb.load_record(rid)
    fs_native = rec["fs"]
    sig_full = rec["signal"]
    p_peaks_native = butpdb.load_pwave_peaks(rid)
    qrs = butpdb.load_qrs(rid)["sample"]
    if fs_native % FS == 0:
        factor = fs_native // FS
        sig_250 = np.stack(
            [scipy_signal.decimate(sig_full[:, c], factor, zero_phase=True)
             for c in range(sig_full.shape[1])], axis=-1)
    else:
        n_new = int(round(sig_full.shape[0] * FS / fs_native))
        sig_250 = np.stack(
            [scipy_signal.resample(sig_full[:, c], n_new)
             for c in range(sig_full.shape[1])], axis=-1)
    scale = sig_250.shape[0] / sig_full.shape[0]
    p_250 = (p_peaks_native * scale).astype(np.int64)
    q_250 = (qrs * scale).astype(np.int64)
    lo = n_window_idx * WINDOW_SAMPLES; hi = lo + WINDOW_SAMPLES
    sig = sig_250[lo:hi, lead_idx].astype(np.float32)
    p_in = [int(s - lo) for s in p_250 if lo <= s < hi]
    q_in = [int(s - lo) for s in q_250 if lo <= s < hi]
    model_lead = 1 if lead_idx == 0 else 10
    _, b_old, bds_old = _run(model_old, sig, model_lead, device)
    _, b_new, bds_new = _run(model_new, sig, model_lead, device)
    title = (f"BUT PDB rid={rid:02d} lead={lead_idx} (win {n_window_idx})  "
             f"GT P={len(p_in)}/QRS={len(q_in)}  "
             f"P_pred old={len(bds_old.get('p_on',[]))} -> "
             f"new={len(bds_new.get('p_on',[]))}")
    plot_panel(sig, p_in, q_in, b_old, b_new, bds_old, bds_new,
               title, OUT_DIR / f"butpdb_{tag}.png")


def case_ludb(rid, lead, model_old, model_new, device, tag):
    ds = LUDBFrameDataset([rid])
    sig_250, lead_idx, _ = ds.cache[(rid, lead)]
    sig = sig_250[:WINDOW_SAMPLES].astype(np.float32)
    gt500 = ludb.load_annotations(rid, lead)
    gt250 = {k: [int(s // 2) for s in v] for k, v in gt500.items()}
    _, b_old, bds_old = _run(model_old, sig, lead_idx, device)
    _, b_new, bds_new = _run(model_new, sig, lead_idx, device)
    p_peaks = gt250.get("p_peak", [])
    q_peaks = gt250.get("qrs_peak", [])
    title = (f"LUDB 3°AVB rid={rid} lead={lead}  GT P={len(p_peaks)}/QRS={len(q_peaks)}  "
             f"P_pred old={len(bds_old.get('p_on',[]))} -> "
             f"new={len(bds_new.get('p_on',[]))}")
    plot_panel(sig, p_peaks, q_peaks, b_old, b_new, bds_old, bds_new,
               title, OUT_DIR / f"ludb_{tag}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old = _load(CKPT_OLD, device)
    new = _load(CKPT_NEW, device)
    case_butpdb(3, 0, old, new, device, "rid03_lead0_BIII")
    case_ludb(74, "i", old, new, device, "rid74_i_GAIN")
    case_ludb(34, "v5", old, new, device, "rid34_v5_REGRESSION")
    print(f"Wrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
