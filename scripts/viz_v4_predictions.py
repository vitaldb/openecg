"""Visualize v4 C model predictions vs GT on LUDB / ISP / QTDB samples.

Picks a handful of records per domain, runs inference, and saves a panel per
sample showing:
  - Signal trace (250Hz)
  - GT mask: colored bands for P / QRS / T
  - Predicted mask: same coloring (post-processed + p_off shift applied)
  - Boundary markers (vertical lines) for both GT and pred

Output: out/viz_v4/<domain>_<id>.png
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.infer import (
    extract_boundaries, load_model_bundle,
    post_process_frames, predict_frames,
)
from openecg.stage2.multi_dataset import _decimate_to_250, _normalize

CKPT = Path("data/checkpoints/stage2_v4_C.pt")
OUT_DIR = Path("out/viz_v4")
WINDOW_SAMPLES = 2500
FRAME_MS = 20
FS = 250

CMAP = {1: ("red", "P"), 2: ("blue", "QRS"), 3: ("green", "T")}


def frames_to_bands(frames, n_samples):
    """Convert per-frame labels to (start_sample, end_sample, class) bands."""
    spf = int(round(FRAME_MS * FS / 1000.0))  # 5
    out = []
    cur = int(frames[0])
    start = 0
    for i, c in enumerate(frames):
        c = int(c)
        if c != cur:
            if cur in CMAP:
                out.append((start * spf, i * spf, cur))
            cur = c
            start = i
    if cur in CMAP:
        out.append((start * spf, len(frames) * spf, cur))
    return [(s, min(e, n_samples), c) for s, e, c in out]


def gt_ann_to_bands(ann):
    """Convert {p_on, p_off, qrs_on, ..., t_off} (sample idx) to bands."""
    out = []
    for cls, on_key, off_key in [(1, "p_on", "p_off"), (2, "qrs_on", "qrs_off"),
                                  (3, "t_on", "t_off")]:
        for on, off in zip(ann.get(on_key, []), ann.get(off_key, [])):
            out.append((int(on), int(off), cls))
    return out


def plot_one(sig, gt_bands, pred_bands, gt_boundaries, pred_boundaries,
             title, out_path):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    t = np.arange(len(sig)) / FS
    ax.plot(t, sig, color="black", linewidth=0.8, zorder=3)

    # Y bands: top = GT, bottom = pred
    y_lo, y_hi = float(sig.min()) - 0.4, float(sig.max()) + 0.4
    band_h = (y_hi - y_lo) * 0.06
    gt_y = y_hi - band_h * 0.4
    pred_y = y_lo + band_h * 0.4

    for s, e, c in gt_bands:
        color, _ = CMAP[c]
        ax.axhspan(gt_y - band_h / 2, gt_y + band_h / 2, xmin=s / len(sig),
                   xmax=e / len(sig), facecolor=color, alpha=0.45)
    for s, e, c in pred_bands:
        color, _ = CMAP[c]
        ax.axhspan(pred_y - band_h / 2, pred_y + band_h / 2, xmin=s / len(sig),
                   xmax=e / len(sig), facecolor=color, alpha=0.45)

    # Boundary markers (vertical lines, dashed=GT solid=pred)
    for key, samples in gt_boundaries.items():
        for s in samples:
            if 0 <= s < len(sig):
                color = CMAP[1 if key.startswith("p") else 2 if key.startswith("qrs") else 3][0]
                ax.axvline(s / FS, color=color, linestyle="--", linewidth=0.6, alpha=0.5)
    for key, samples in pred_boundaries.items():
        for s in samples:
            if 0 <= s < len(sig):
                color = CMAP[1 if key.startswith("p") else 2 if key.startswith("qrs") else 3][0]
                ax.axvline(s / FS, color=color, linestyle="-", linewidth=0.6, alpha=0.7)

    # Labels
    ax.text(-0.02, gt_y, "GT", transform=ax.get_yaxis_transform(), va="center",
            ha="right", fontsize=9, color="dimgray")
    ax.text(-0.02, pred_y, "pred", transform=ax.get_yaxis_transform(), va="center",
            ha="right", fontsize=9, color="dimgray")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.45) for c, _ in CMAP.values()]
    handles += [plt.Line2D([0], [0], color="black", linestyle="--", label="GT bound"),
                plt.Line2D([0], [0], color="black", linestyle="-", label="pred bound")]
    labels = [n for _, n in CMAP.values()] + ["GT bound", "pred bound"]
    ax.legend(handles, labels, loc="upper right", fontsize=8, ncols=5,
              framealpha=0.9)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (z-norm)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, len(sig) / FS)
    ax.set_ylim(y_lo - band_h, y_hi + band_h)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def predict_full(model, sig, lead_idx, device):
    raw = predict_frames(model, sig, lead_idx, device=device)
    pp = post_process_frames(raw, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    return pp, bds


def viz_ludb(model, device, n_records=4):
    rec_ids = ludb.load_split()["val"][:n_records]
    ds = LUDBFrameDataset(rec_ids)
    leads_to_show = ["i", "ii", "v2", "v5"]
    for rid in rec_ids:
        for lead in leads_to_show:
            try:
                sig_250, lead_idx, true_frames = ds.cache[(rid, lead)]
            except KeyError:
                continue
            sig = sig_250[:WINDOW_SAMPLES]
            true_frames = true_frames[:WINDOW_SAMPLES // 5]
            if len(sig) < WINDOW_SAMPLES:
                continue
            pp, bds = predict_full(model, sig, lead_idx, device)
            gt_ann = ludb.load_annotations(rid, lead)
            # LUDB ann is at 500Hz native -> divide by 2 for 250Hz domain
            gt_ann_250 = {k: [int(v // 2) for v in vals]
                          for k, vals in gt_ann.items()}
            gt_bands = gt_ann_to_bands(gt_ann_250)
            pred_bands = frames_to_bands(pp, len(sig))
            gt_b = {k: gt_ann_250[k] for k in
                    ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}
            title = f"LUDB val rid={rid} lead={lead}"
            out_path = OUT_DIR / f"ludb_{rid:03d}_{lead}.png"
            plot_one(sig, gt_bands, pred_bands, gt_b, bds, title, out_path)
            print(f"  {out_path.name}", flush=True)


def viz_isp(model, device, n_records=4):
    rec_ids = isp.load_split()["test"][:n_records]
    leads_to_show = ["i", "ii", "v2", "v5"]
    for rid in rec_ids:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception as e:
            print(f"  ISP rid={rid} skip: {e}", flush=True)
            continue
        for lead in leads_to_show:
            if lead not in record:
                continue
            lead_idx = isp.LEADS_12.index(lead)
            sig_1000 = record[lead]
            sig_250 = _decimate_to_250(sig_1000, 1000)
            sig_n = _normalize(sig_250)
            if len(sig_n) < WINDOW_SAMPLES:
                pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                sig_n = np.concatenate([sig_n, pad])
            sig = sig_n[:WINDOW_SAMPLES]
            pp, bds = predict_full(model, sig, lead_idx, device)
            ann_250 = {k: [int(v // 4) for v in vals if int(v // 4) < WINDOW_SAMPLES]
                       for k, vals in ann.items()}
            gt_bands = gt_ann_to_bands(ann_250)
            pred_bands = frames_to_bands(pp, len(sig))
            gt_b = {k: ann_250[k] for k in
                    ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")
                    if k in ann_250}
            title = f"ISP test rid={rid} lead={lead}"
            out_path = OUT_DIR / f"isp_{rid}_{lead}.png"
            plot_one(sig, gt_bands, pred_bands, gt_b, bds, title, out_path)
            print(f"  {out_path.name}", flush=True)


def viz_qtdb(model, device, n_records=4):
    rids = qtdb.records_with_q1c()[:n_records]
    for rid in rids:
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_q1c(rid)
        except Exception as e:
            print(f"  QTDB rid={rid} skip: {e}", flush=True)
            continue
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
        if win is None:
            continue
        start, end = win
        if end > 225000:
            end = 225000
            start = end - WINDOW_SAMPLES
        for lead_name, sig_full in record.items():
            sig = sig_full[start:end]
            if len(sig) < WINDOW_SAMPLES:
                continue
            sig_n = _normalize(sig)
            from openecg.stage2.multi_dataset import QTDB_LEAD_TO_LUDB_ID
            lead_idx = QTDB_LEAD_TO_LUDB_ID.get(lead_name, 1)
            pp, bds = predict_full(model, sig_n, lead_idx, device)
            ann_local = {k: [int(s - start) for s in v
                              if start <= s < end]
                          for k, v in ann.items()}
            gt_bands = gt_ann_to_bands(ann_local)
            pred_bands = frames_to_bands(pp, len(sig))
            gt_b = {k: ann_local[k] for k in
                    ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")
                    if k in ann_local}
            title = f"QTDB rid={rid} lead={lead_name} (samples {start}-{end})"
            out_path = OUT_DIR / f"qtdb_{rid}_{lead_name}.png"
            plot_one(sig_n, gt_bands, pred_bands, gt_b, bds, title, out_path)
            print(f"  {out_path.name}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT}...", flush=True)
    bundle = load_model_bundle(CKPT, device=device)
    model = bundle["model"]
    model = model.train(False)
    print(f"Device {device}, model_config={bundle.get('model_config')}", flush=True)

    print("\n--- LUDB val ---", flush=True)
    viz_ludb(model, device, n_records=4)
    print("\n--- ISP test ---", flush=True)
    viz_isp(model, device, n_records=4)
    print("\n--- QTDB ---", flush=True)
    viz_qtdb(model, device, n_records=4)
    print(f"\nAll plots saved under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
