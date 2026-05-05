"""Comprehensive visualization across LUDB / ISP / QTDB test sets.

Layout per panel (top to bottom):
  Wave (signal trace)
  GT     (cardiologist annotation)
  v4 C   (our model)
  NK     (NeuroKit2 DWT baseline)
  WT     (WTdelineator baseline)

For QTDB we show both q1c (sparse expert) and pu0 (dense automatic) as separate
GT references since neither alone is complete.

Output:
  out/viz_test/ludb/<rid>_<lead>.png      (4 records × 4 leads = 16 plots)
  out/viz_test/isp/<rid>_<lead>.png       (4 records × 4 leads = 16 plots)
  out/viz_test/qtdb/<rid>_<lead>.png      (6 records × first lead = 6 plots)
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.infer import (
    extract_boundaries, load_model_bundle,
    post_process_frames, predict_frames,
)
from ecgcode.stage2.multi_dataset import _decimate_to_250, _normalize

CKPT = REPO / "data" / "checkpoints" / "stage2_v4_C.pt"
OUT_BASE = REPO / "out" / "viz_test"
WINDOW_SAMPLES = 2500
FS = 250
FRAME_MS = 20
CMAP = {1: ("red", "P"), 2: ("blue", "QRS"), 3: ("green", "T")}


# ---------- Annotation parsing ----------

def parse_wfdb_ann(record_path, ext):
    try:
        ann = wfdb.rdann(str(record_path), ext)
    except Exception:
        return None
    out = {k: [] for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}
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
    """Pair sorted onsets/offsets per wave class. Reject pairs > 500ms (mis-pair)."""
    out = []
    for cls, on_k, off_k in [(1, "p_on", "p_off"), (2, "qrs_on", "qrs_off"),
                              (3, "t_on", "t_off")]:
        ons = sorted(ann.get(on_k, []))
        offs = sorted(ann.get(off_k, []))
        if not ons or not offs:
            for s in offs:
                if lo <= s < hi:
                    out.append((s - lo, min(s + 5, hi) - lo, cls))
            continue
        used_off = 0
        for on in ons:
            while used_off < len(offs) and offs[used_off] < on:
                used_off += 1
            if used_off >= len(offs):
                break
            off = offs[used_off]
            used_off += 1
            if off - on > 125:
                continue
            if off < lo or on >= hi:
                continue
            out.append((max(on, lo) - lo, min(off, hi) - lo, cls))
    return out


def ann_to_lines(ann, lo, hi):
    return {k: [s - lo for s in ann.get(k, []) if lo <= s < hi]
            for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}


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


# ---------- Baselines ----------

def nk_delineate(sig, fs=FS):
    import neurokit2 as nk
    try:
        _, info = nk.ecg_delineate(sig, sampling_rate=fs, method="dwt")
    except Exception:
        return {k: [] for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}
    return {
        "p_on":   [int(x) for x in info.get("ECG_P_Onsets",  []) if x is not None and not np.isnan(x)],
        "p_off":  [int(x) for x in info.get("ECG_P_Offsets", []) if x is not None and not np.isnan(x)],
        "qrs_on": [int(x) for x in info.get("ECG_R_Onsets",  []) if x is not None and not np.isnan(x)],
        "qrs_off":[int(x) for x in info.get("ECG_R_Offsets", []) if x is not None and not np.isnan(x)],
        "t_on":   [int(x) for x in info.get("ECG_T_Onsets",  []) if x is not None and not np.isnan(x)],
        "t_off":  [int(x) for x in info.get("ECG_T_Offsets", []) if x is not None and not np.isnan(x)],
    }


def wt_delineate(sig, fs=FS):
    import WTdelineator as wav
    try:
        P, QRS, T = wav.signalDelineation(sig.astype(np.float64), fs)
    except Exception:
        return {k: [] for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}
    out = {k: [] for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}
    for arr, on_k, off_k in [(P, "p_on", "p_off"), (QRS, "qrs_on", "qrs_off"),
                               (T, "t_on", "t_off")]:
        for v in arr[:, 0]:
            if int(v) != 0: out[on_k].append(int(v))
        for v in arr[:, -1]:
            if int(v) != 0: out[off_k].append(int(v))
    return out


def baselines_for_signal(sig):
    """Returns (nk_bds, nk_bands_via_pred, wt_bds, wt_bands_via_pred)."""
    nk = nk_delineate(sig)
    wt = wt_delineate(sig)
    return nk, ann_to_bands(nk, 0, len(sig)), wt, ann_to_bands(wt, 0, len(sig))


# ---------- v4 ----------

def v4_predict(model, sig, lead_idx, device):
    sig = sig.astype(np.float32)
    sig_n = ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)
    raw = predict_frames(model, sig_n, lead_idx, device=device)
    pp = post_process_frames(raw, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    bands = frames_to_bands(pp, len(sig))
    return bds, bands


# ---------- Plot ----------

def plot_panel(sig, tracks, title, out_path):
    """tracks = list of (label, bands_list, boundary_lines_dict)."""
    n = len(sig)
    fig, ax = plt.subplots(figsize=(16, 1.5 + 0.6 * len(tracks) + 3.0))
    t = np.arange(n) / FS
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
    ax.plot(t, sig_norm, color="black", linewidth=0.7, zorder=4)

    y_lo = float(sig_norm.min()) - 0.5
    y_hi = float(sig_norm.max()) + 0.5
    band_h = (y_hi - y_lo) * 0.05
    n_tracks = len(tracks)
    # All tracks below the wave (below y=0 area)
    track_top = y_lo - band_h * 0.8
    spacing = band_h * 1.4
    for ti, (label, bands, lines) in enumerate(tracks):
        y = track_top - ti * spacing
        for s, e, c in bands:
            if s < 0 or e > n:
                continue
            color, _ = CMAP[c]
            ax.axhspan(y - band_h / 2, y + band_h / 2,
                       xmin=s / n, xmax=e / n, facecolor=color, alpha=0.5)
        for k, samples in lines.items():
            color = CMAP[1 if k.startswith("p") else 2 if k.startswith("qrs") else 3][0]
            for s in samples:
                if 0 <= s < n:
                    ax.plot([s / FS, s / FS], [y - band_h / 2, y + band_h / 2],
                            color=color, linewidth=0.6, alpha=0.7)
        ax.text(-0.012, y, label, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=10, fontweight="bold",
                color="dimgray")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c, _ in CMAP.values()]
    labels = [n for _, n in CMAP.values()]
    ax.legend(handles, labels, loc="upper right", fontsize=9, ncols=3,
              framealpha=0.9, title="wave class")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (z-norm)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, n / FS)
    ax.set_ylim(track_top - n_tracks * spacing - band_h, y_hi + band_h)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------- Per-dataset runners ----------

def viz_ludb(model, device, n_records=4, leads=("i", "ii", "v2", "v5")):
    out_dir = OUT_BASE / "ludb"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_ids = ludb.load_split()["val"][:n_records]
    ds = LUDBFrameDataset(rec_ids)
    for rid in rec_ids:
        for lead in leads:
            try:
                sig_250, lead_idx, _ = ds.cache[(rid, lead)]
            except KeyError:
                continue
            sig = sig_250[:WINDOW_SAMPLES]
            if len(sig) < WINDOW_SAMPLES:
                continue
            v4_bds, v4_bands = v4_predict(model, sig, lead_idx, device)
            try:
                gt_500 = ludb.load_annotations(rid, lead)
            except Exception:
                continue
            gt_250 = {k: [int(s // 2) for s in v] for k, v in gt_500.items()}
            gt_bands = ann_to_bands(gt_250, 0, len(sig))
            gt_lines = ann_to_lines(gt_250, 0, len(sig))
            nk, nk_bands, wt, wt_bands = baselines_for_signal(sig)
            tracks = [
                ("GT (cardio)", gt_bands, gt_lines),
                ("v4 C",        v4_bands, v4_bds),
                ("NeuroKit",    nk_bands, nk),
                ("WTdel",       wt_bands, wt),
            ]
            title = f"LUDB val rid={rid} lead={lead}"
            plot_panel(sig, tracks, title, out_dir / f"{rid:03d}_{lead}.png")
            print(f"  {out_dir.name}/{rid:03d}_{lead}.png", flush=True)


def viz_isp(model, device, n_records=4, leads=("i", "ii", "v2", "v5")):
    out_dir = OUT_BASE / "isp"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_ids = isp.load_split()["test"][:n_records]
    for rid in rec_ids:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception:
            continue
        for lead in leads:
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
            v4_bds, v4_bands = v4_predict(model, sig, lead_idx, device)
            ann_250 = {k: [int(s // 4) for s in v if int(s // 4) < WINDOW_SAMPLES]
                        for k, v in ann.items()}
            gt_bands = ann_to_bands(ann_250, 0, len(sig))
            gt_lines = ann_to_lines(ann_250, 0, len(sig))
            nk, nk_bands, wt, wt_bands = baselines_for_signal(sig)
            tracks = [
                ("GT (2-cardio)", gt_bands, gt_lines),
                ("v4 C",          v4_bands, v4_bds),
                ("NeuroKit",      nk_bands, nk),
                ("WTdel",         wt_bands, wt),
            ]
            title = f"ISP test rid={rid} lead={lead}"
            plot_panel(sig, tracks, title, out_dir / f"{rid}_{lead}.png")
            print(f"  {out_dir.name}/{rid}_{lead}.png", flush=True)


def viz_qtdb(model, device, n_records=6):
    """Plot q1c (sparse expert) + pu0 (dense auto) + v4 + NK + WT
    over the q1c span ±2s context for visual comparison."""
    out_dir = OUT_BASE / "qtdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = qtdb.ensure_extracted()
    rids = qtdb.records_with_q1c()[:n_records]
    for rid in rids:
        record = qtdb.load_record(rid)
        first_lead = list(record.keys())[0]
        sig_full = record[first_lead].astype(np.float32)
        n = len(sig_full)
        ann_q1c = parse_wfdb_ann(inner / rid, "q1c") or {}
        ann_pu0 = parse_wfdb_ann(inner / rid, "pu0") or {}

        all_q1c = []
        for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
            all_q1c.extend(ann_q1c.get(k, []))
        if not all_q1c:
            continue
        margin = 2 * FS
        win_lo = max(0, min(all_q1c) - margin)
        win_hi = min(n, max(all_q1c) + margin)
        # cap at one 10s window for clarity
        if win_hi - win_lo > WINDOW_SAMPLES:
            mid = (win_lo + win_hi) // 2
            win_lo = max(0, mid - WINDOW_SAMPLES // 2)
            win_hi = win_lo + WINDOW_SAMPLES
        sig_win = sig_full[win_lo:win_hi]
        n_win = len(sig_win)
        # v4 needs exactly 2500 samples; pad if shorter
        if n_win < WINDOW_SAMPLES:
            pad = np.zeros(WINDOW_SAMPLES - n_win, dtype=sig_win.dtype)
            sig_for_v4 = np.concatenate([sig_win, pad])
        else:
            sig_for_v4 = sig_win[:WINDOW_SAMPLES]
        v4_bds, v4_bands = v4_predict(model, sig_for_v4, 1, device)
        # truncate v4 outputs to actual window length
        v4_bds = {k: [s for s in v if s < n_win] for k, v in v4_bds.items()}
        v4_bands = [(s, e, c) for s, e, c in v4_bands if s < n_win]

        nk, nk_bands, wt, wt_bands = baselines_for_signal(sig_win)
        q1c_bands = ann_to_bands(ann_q1c, win_lo, win_hi)
        q1c_lines = ann_to_lines(ann_q1c, win_lo, win_hi)
        pu0_bands = ann_to_bands(ann_pu0, win_lo, win_hi)
        pu0_lines = ann_to_lines(ann_pu0, win_lo, win_hi)
        tracks = [
            ("GT q1c (expert)", q1c_bands, q1c_lines),
            ("GT pu0 (auto)",   pu0_bands, pu0_lines),
            ("v4 C",            v4_bands, v4_bds),
            ("NeuroKit",        nk_bands, nk),
            ("WTdel",           wt_bands, wt),
        ]
        title = f"QTDB rid={rid} lead={first_lead} (samples [{win_lo}, {win_hi}])"
        plot_panel(sig_win, tracks, title, out_dir / f"{rid}_{first_lead}.png")
        print(f"  {out_dir.name}/{rid}_{first_lead}.png", flush=True)


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT}", flush=True)
    bundle = load_model_bundle(CKPT, device=device)
    model = bundle["model"].train(False)
    t0 = time.time()
    print("\n--- LUDB val ---", flush=True)
    viz_ludb(model, device, n_records=4)
    print("\n--- ISP test ---", flush=True)
    viz_isp(model, device, n_records=4)
    print("\n--- QTDB ---", flush=True)
    viz_qtdb(model, device, n_records=6)
    print(f"\nAll plots saved under {OUT_BASE} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
