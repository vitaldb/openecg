"""Side-by-side visual comparison: ground truth vs v56c vs NeuroKit2 vs WTdelineator.

Renders one figure per dataset showing the same ECG strip annotated by
each detector, so a human can see *why* the macro-F1 numbers in
``docs/benchmarks/v56c_vs_baselines.md`` differ.

Output: ``docs/figures/v56c_vs_baselines_<dataset>.png``.

Layout per figure (4 rows × 1 column):
  1. Ground truth (cardiologist annotations) — coloured spans on the ECG.
  2. openecg v56c — same ECG, model's predicted P/QRS/T spans.
  3. NeuroKit2 DWT.
  4. WTdelineator.

Each row also shows boundary onset/offset as vertical ticks at the top
so on/off precision is visible even when wave durations are short.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.dsp import rank_normalize
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import (
    apply_reg_to_boundaries, extract_boundaries,
    post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import load_model_from_ckpt
from openecg.stage2.multi_dataset import _decimate_to_250

# Reuse the baseline wrappers from the benchmark script.
from scripts.benchmark_v56c import (
    nk_delineate, wt_delineate, v56c_predict,
    FRAME_MS, FS, WINDOW_SAMPLES, BOUNDARY_KEYS,
)


# --- Colours: clinical convention (P=red, QRS=blue, T=green) ----------------

WAVE_COLOURS = {"P": "#d62728", "QRS": "#1f77b4", "T": "#2ca02c"}

CKPT = REPO / "data" / "checkpoints" / "stage2_v45k_noaux_L8_d128_1ch_v56c.pt"
OUT_DIR = REPO / "docs" / "figures"


# --- Boundary plotting helpers ----------------------------------------------

def _spans_from_boundaries(bounds: dict) -> list[tuple[int, int, str]]:
    """Pair on/off into (start, end, wave) by matching successive
    ``*_on`` and ``*_off`` lists. Returns a sorted list of spans."""
    spans = []
    for wave, on_key, off_key in [
        ("P", "p_on", "p_off"),
        ("QRS", "qrs_on", "qrs_off"),
        ("T", "t_on", "t_off"),
    ]:
        ons = sorted(int(x) for x in bounds.get(on_key, []))
        offs = sorted(int(x) for x in bounds.get(off_key, []))
        # Greedy match: pair each onset with the next offset after it.
        j = 0
        for on in ons:
            while j < len(offs) and offs[j] < on:
                j += 1
            if j < len(offs):
                spans.append((on, offs[j], wave))
                j += 1
    spans.sort()
    return spans


def _plot_strip(ax, sig, bounds, fs, title):
    """Plot one ECG row with shaded P/QRS/T spans + onset/offset tick marks."""
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, color="black", linewidth=0.8)
    ax.set_xlim(t[0], t[-1])
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.15 * yrange)
    for start, end, wave in _spans_from_boundaries(bounds):
        if end <= start or end <= 0 or start >= len(sig):
            continue
        c = WAVE_COLOURS[wave]
        ax.axvspan(start / fs, end / fs, color=c, alpha=0.18, linewidth=0)
        # Onset/offset tick marks (top stripe).
        tick_y = ymax + 0.05 * yrange
        ax.plot([start / fs], [tick_y], marker="|", color=c,
                markersize=8, markeredgewidth=1.5)
        ax.plot([end / fs], [tick_y], marker="|", color=c,
                markersize=8, markeredgewidth=1.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


# --- Per-dataset record selection -------------------------------------------

def _load_ludb_strip(rid: int, lead: str):
    """Return (sig_250hz [N], gt_dict, lead_idx) for a LUDB record-lead."""
    ds = LUDBFrameDataset([rid])
    sig_250, lead_idx, _ = ds.cache[(rid, lead)]
    sig_250 = sig_250[:WINDOW_SAMPLES].astype(np.float32)
    gt_500 = ludb.load_annotations(rid, lead)
    # LUDB annotations are at 500 Hz native; halve them to match the 250 Hz strip.
    gt = {}
    for k in BOUNDARY_KEYS:
        gt[k] = [int(s) // 2 for s in gt_500.get(k, [])]
    return sig_250, gt, lead_idx


def _load_isp_strip(rid: int, lead: str):
    record = isp.load_record(rid, split="test")
    ann = isp.load_annotations_as_super(int(rid), split="test")
    sig_1000 = record[lead]
    sig_250 = _decimate_to_250(sig_1000, 1000)
    sig_250 = sig_250[:WINDOW_SAMPLES].astype(np.float32)
    # ISP annotations are at 1000 Hz native; downsample to 250 Hz.
    gt = {}
    for k in BOUNDARY_KEYS:
        gt[k] = [int(s) // 4 for s in ann.get(k, [])
                  if 0 <= int(s) // 4 < WINDOW_SAMPLES]
    lead_idx = isp.LEADS_12.index(lead)
    return sig_250, gt, lead_idx


def _load_qtdb_strip(rid: str):
    record = qtdb.load_record(rid)
    ann = qtdb.load_q1c(rid)
    win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
    if win is None:
        return None
    start, end = win
    if end > 225000:
        end = 225000
        start = end - WINDOW_SAMPLES
    first_lead = list(record.keys())[0]
    sig = record[first_lead][start:end].astype(np.float32)
    gt = {}
    for k in BOUNDARY_KEYS:
        gt[k] = [int(s - start) for s in ann.get(k, [])
                  if start <= int(s) < end]
    return sig, gt, 1  # MLII is roughly lead II ≈ idx 1


# --- Figure builder ---------------------------------------------------------

def _crop_for_clarity(sig, bounds_list, gt, n_sec=4.0, fs=FS):
    """Crop to ``n_sec`` seconds around the densest ground-truth segment so
    the figure is readable. Returns the same data sliced + boundary lists
    shifted so the cropped window starts at sample 0."""
    n = int(round(n_sec * fs))
    if len(sig) <= n:
        return sig, bounds_list, gt, 0
    # Histogram density of GT boundary samples to find the peak region.
    all_gt = []
    for k in BOUNDARY_KEYS:
        all_gt.extend(gt.get(k, []))
    if not all_gt:
        return sig[:n], bounds_list, gt, 0
    bins = np.bincount(np.array(all_gt) // fs, minlength=len(sig) // fs + 1)
    center_sec = int(np.argmax(np.convolve(bins, np.ones(int(n_sec)),
                                            mode="same")))
    start = max(0, int(center_sec * fs) - n // 2)
    start = min(start, len(sig) - n)
    cropped = sig[start:start + n]

    def _shift(bd):
        return {
            k: [s - start for s in v
                if start <= s < start + n] for k, v in bd.items()
        }
    return cropped, [_shift(b) for b in bounds_list], _shift(gt), start


def render_dataset(name, sig, gt, lead_idx, model, device, out_path):
    """Run all 4 detectors on the same strip and save a 4-row PNG."""
    p_v56c = v56c_predict(model, sig, lead_idx, device)
    p_nk = nk_delineate(sig)
    p_wt = wt_delineate(sig)

    sig_c, (p_v56c_c, p_nk_c, p_wt_c), gt_c, _ = _crop_for_clarity(
        sig, [p_v56c, p_nk, p_wt], gt, n_sec=4.0)

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 6.5),
        gridspec_kw={"hspace": 0.45},
    )
    _plot_strip(axes[0], sig_c, gt_c, FS,
                "Ground truth (cardiologist annotation)")
    _plot_strip(axes[1], sig_c, p_v56c_c, FS, "openecg v56c (this work)")
    _plot_strip(axes[2], sig_c, p_nk_c, FS, "NeuroKit2 DWT")
    _plot_strip(axes[3], sig_c, p_wt_c, FS,
                "WTdelineator (Martínez 2004 reimpl.)")
    axes[-1].set_xlabel("time (s)")

    # Legend for wave colours.
    legend_patches = [
        mpatches.Patch(color=c, alpha=0.35, label=wave)
        for wave, c in WAVE_COLOURS.items()
    ]
    axes[0].legend(handles=legend_patches, loc="upper right",
                    fontsize=8, ncol=3, framealpha=0.9)

    fig.suptitle(name, fontsize=12, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {CKPT} on {device}", flush=True)
    model, _ = load_model_from_ckpt(str(CKPT), device=device)
    model.train(False)

    cases = [
        # Record selections aim for "median difficulty" — neither obvious
        # nor pathological — so the visual gap is fair.
        ("LUDB val record 16, lead II", "ludb",
         lambda: _load_ludb_strip(16, "ii"),
         OUT_DIR / "v56c_vs_baselines_ludb.png"),
        ("ISP test record 2, lead II", "isp",
         lambda: _load_isp_strip(2, "ii"),
         OUT_DIR / "v56c_vs_baselines_isp.png"),
        ("QTDB record sel100, MLII", "qtdb",
         lambda: _load_qtdb_strip("sel100"),
         OUT_DIR / "v56c_vs_baselines_qtdb.png"),
    ]
    for title, _ds, loader, out_path in cases:
        try:
            data = loader()
        except Exception as exc:
            print(f"!! skip {title}: {exc}")
            continue
        if data is None:
            print(f"!! skip {title}: no annotated window")
            continue
        sig, gt, lead_idx = data
        with torch.no_grad():
            render_dataset(title, sig, gt, lead_idx, model, device, out_path)


if __name__ == "__main__":
    main()
