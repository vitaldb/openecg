"""Visual side-by-side: signal + v4 C predictions + q1c + q2c + pu annotations.

For each q2c-available QTDB record (11 records), plot the q1c-annotated time
range with all 4 annotation tracks overlaid so a human can judge agreement.

Output: out/viz_annotators/<rid>.png

Layout per panel:
  Top:    raw signal (z-norm)
  Track 1: v4 C predicted bands (P/QRS/T color)
  Track 2: q1c labeled bands (expert 1)
  Track 3: q2c labeled bands (expert 2; sparse — usually only QRS+T_off)
  Track 4: pu labeled bands (dense auto algorithm by QTDB authors)

Vertical lines mark each labeled boundary (color = wave class).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import qtdb
from openecg.stage2.infer import (
    extract_boundaries, load_model_bundle,
    post_process_frames, predict_frames,
)
from openecg.stage2.multi_dataset import _normalize

CKPT = REPO / "data" / "checkpoints" / "stage2_v4_C.pt"
OUT_DIR = REPO / "out" / "viz_annotators"
WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20

CMAP = {1: ("red", "P"), 2: ("blue", "QRS"), 3: ("green", "T")}

Q2C_RECORDS = ["sel100", "sel102", "sel103", "sel114", "sel116",
               "sel117", "sel123", "sel213", "sel221", "sel223", "sel230"]


def load_ann_as_dict(record_path, ext):
    """Returns {p_on, p_off, qrs_on, qrs_off, t_on, t_off} as sample lists."""
    try:
        ann = wfdb.rdann(str(record_path), ext)
    except Exception:
        return None
    out = {"p_on": [], "p_off": [], "qrs_on": [], "qrs_off": [],
           "t_on": [], "t_off": []}
    for i, sym in enumerate(ann.symbol):
        s = int(ann.sample[i])
        if sym == "p":
            if i > 0 and ann.symbol[i - 1] == "(":
                out["p_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["p_off"].append(int(ann.sample[i + 1]))
        elif sym == "N":
            if i > 0 and ann.symbol[i - 1] == "(":
                out["qrs_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["qrs_off"].append(int(ann.sample[i + 1]))
        elif sym == "t":
            if i > 0 and ann.symbol[i - 1] == "(":
                out["t_on"].append(int(ann.sample[i - 1]))
            if i + 1 < len(ann.symbol) and ann.symbol[i + 1] == ")":
                out["t_off"].append(int(ann.sample[i + 1]))
    return out


def ann_to_bands(ann, lo, hi):
    """Returns list of (start, end, class) bands within [lo, hi).

    Pairs sorted ons[i] with offs[i] for each wave class. Tolerates count
    mismatch by greedy nearest-after-onset matching: for each onset, take the
    first offset >= onset that hasn't been used yet.
    """
    out = []
    for cls, on_k, off_k in [(1, "p_on", "p_off"), (2, "qrs_on", "qrs_off"),
                              (3, "t_on", "t_off")]:
        ons = sorted(ann.get(on_k, []))
        offs = sorted(ann.get(off_k, []))
        if not ons or not offs:
            # If only offsets exist (e.g. q1c has t_off but no t_on), treat each
            # offset as a tiny marker so it's still visible.
            for s in offs:
                if lo <= s < hi:
                    out.append((s - lo, min(s + 5, hi) - lo, cls))
            continue
        used_off = 0
        for on in ons:
            # Skip already-used offsets
            while used_off < len(offs) and offs[used_off] < on:
                used_off += 1
            if used_off >= len(offs):
                break
            off = offs[used_off]
            used_off += 1
            # Sanity bound: typical wave duration < 250ms = 62 samples @250Hz.
            # Reject pairs longer than 500ms (probably mis-paired across beats).
            if off - on > 125:
                continue
            if off < lo or on >= hi:
                continue
            on_c = max(on, lo)
            off_c = min(off, hi)
            out.append((on_c - lo, off_c - lo, cls))
    return out


def ann_to_boundary_lines(ann, lo, hi):
    """Returns dict {class_name: [sample]}."""
    out = {}
    for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
        out[k] = [s - lo for s in ann.get(k, []) if lo <= s < hi]
    return out


def frames_to_bands(frames, n_samples):
    spf = int(round(FRAME_MS * FS / 1000.0))
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


def plot_one(rid, sig, n_samples, v4_bands, v4_bds, q1c_bands, q1c_bds,
             q2c_bands, q2c_bds, pu_bands, pu_bds, win_lo, win_hi,
             out_path):
    fig, ax = plt.subplots(figsize=(16, 5.5))
    t = np.arange(n_samples) / FS
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
    ax.plot(t, sig_norm, color="black", linewidth=0.7, zorder=4)

    y_lo = float(sig_norm.min()) - 0.5
    y_hi = float(sig_norm.max()) + 0.5
    band_h = (y_hi - y_lo) * 0.06
    tracks = [
        ("v4 C",  y_hi - band_h * 0.5, v4_bands, v4_bds),
        ("q1c",   y_hi - band_h * 1.7, q1c_bands, q1c_bds),
        ("q2c",   y_lo + band_h * 1.7, q2c_bands, q2c_bds),
        ("pu0",    y_lo + band_h * 0.5, pu_bands, pu_bds),
    ]
    for label, y, bands, bds in tracks:
        for s, e, c in bands:
            if s < 0 or e > n_samples:
                continue
            color, _ = CMAP[c]
            ax.axhspan(y - band_h / 2, y + band_h / 2,
                       xmin=s / n_samples, xmax=e / n_samples,
                       facecolor=color, alpha=0.45)
        for k, samples in bds.items():
            color = CMAP[1 if k.startswith("p") else 2 if k.startswith("qrs") else 3][0]
            for s in samples:
                if 0 <= s < n_samples:
                    ax.plot([s / FS, s / FS], [y - band_h / 2, y + band_h / 2],
                            color=color, linewidth=0.8, alpha=0.8)
        ax.text(-0.012, y, label, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=10, color="dimgray")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.45) for c, _ in CMAP.values()]
    labels = [n for _, n in CMAP.values()]
    ax.legend(handles, labels, loc="upper right", fontsize=9, ncols=3,
              framealpha=0.9, title="wave class")

    ax.set_xlabel(f"time within window (s)  |  absolute samples [{win_lo}, {win_hi})")
    ax.set_ylabel("signal (z-norm)")
    ax.set_title(f"QTDB {rid}: v4 C vs q1c (expert 1) vs q2c (expert 2) vs pu (auto)",
                 fontsize=11)
    ax.set_xlim(0, n_samples / FS)
    ax.set_ylim(y_lo - band_h, y_hi + band_h)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT}", flush=True)
    bundle = load_model_bundle(CKPT, device=device)
    model = bundle["model"].train(False)

    inner = qtdb.ensure_extracted()

    for rid in Q2C_RECORDS:
        print(f"\n--- {rid} ---", flush=True)
        record = qtdb.load_record(rid)
        first_lead = list(record.keys())[0]
        sig_full = record[first_lead].astype(np.float32)
        n = len(sig_full)

        ann_q1c = load_ann_as_dict(inner / rid, "q1c") or {}
        ann_q2c = load_ann_as_dict(inner / rid, "q2c") or {}
        ann_pu  = load_ann_as_dict(inner / rid, "pu0") or {}

        # Pick the window: use q1c span (small annotated burst) ±2s context
        all_q1c = []
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            all_q1c.extend(ann_q1c.get(k, []))
        if not all_q1c:
            print(f"  {rid}: no q1c annotations — skip", flush=True)
            continue
        q1c_min = min(all_q1c)
        q1c_max = max(all_q1c)
        # Display ±2s context if available
        margin = 2 * FS
        win_lo = max(0, q1c_min - margin)
        win_hi = min(n, q1c_max + margin)
        # Cap at 30s for readability
        if win_hi - win_lo > 30 * FS:
            win_hi = win_lo + 30 * FS
        sig_win = sig_full[win_lo:win_hi]
        n_win = len(sig_win)

        # Run v4 in sliding non-overlapping 10s windows over the display window
        v4_b = {k: [] for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}
        v4_frames_concat = []
        sig_norm_full = _normalize(sig_full)
        # Slide over display window
        for sub_start in range(win_lo, win_hi - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
            sub = sig_norm_full[sub_start:sub_start + WINDOW_SAMPLES].astype(np.float32)
            # Per-window re-normalize for parity with training
            sub = ((sub - sub.mean()) / (sub.std() + 1e-6)).astype(np.float32)
            raw = predict_frames(model, sub, lead_id=1, device=device)
            pp = post_process_frames(raw, frame_ms=FRAME_MS)
            v4_frames_concat.append(pp)
            local = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            offset_in_win = sub_start - win_lo
            for k, vs in local.items():
                for v in vs:
                    if 0 <= v < WINDOW_SAMPLES:
                        v4_b[k].append(int(v) + offset_in_win)

        v4_frames = np.concatenate(v4_frames_concat) if v4_frames_concat else np.array([], dtype=np.uint8)
        v4_bands = frames_to_bands(v4_frames, n_win) if len(v4_frames) else []

        q1c_bands = ann_to_bands(ann_q1c, win_lo, win_hi)
        q2c_bands = ann_to_bands(ann_q2c, win_lo, win_hi)
        pu_bands  = ann_to_bands(ann_pu, win_lo, win_hi)

        q1c_bds = ann_to_boundary_lines(ann_q1c, win_lo, win_hi)
        q2c_bds = ann_to_boundary_lines(ann_q2c, win_lo, win_hi)
        pu_bds  = ann_to_boundary_lines(ann_pu, win_lo, win_hi)
        v4_bds = v4_b

        out_path = OUT_DIR / f"{rid}.png"
        plot_one(rid, sig_win, n_win, v4_bands, v4_bds, q1c_bands, q1c_bds,
                 q2c_bands, q2c_bds, pu_bands, pu_bds, win_lo, win_hi,
                 out_path)
        print(f"  saved {out_path.name} (window {(win_hi-win_lo)/FS:.1f}s)", flush=True)

    print(f"\nAll plots saved under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
