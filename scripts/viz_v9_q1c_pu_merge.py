"""Visualize v9_q1c_pu_merge predictions across LUDB / ISP / QTDB test sets.

Layout: signal + GT + v9_merge + NK + WT (CPU inference to avoid GPU contention).
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.infer import (
    extract_boundaries, post_process_frames, predict_frames,
)
from ecgcode.stage2.model import FrameClassifierViT
from ecgcode.stage2.multi_dataset import _decimate_to_250, _normalize
from ecgcode.stage2.train import load_checkpoint
from scripts.viz_test_all_datasets import (
    ann_to_bands, ann_to_lines, baselines_for_signal, frames_to_bands,
    parse_wfdb_ann, plot_panel, CMAP, FRAME_MS, FS, WINDOW_SAMPLES,
)

CKPT = REPO / "data" / "checkpoints" / "stage2_v9_q1c_pu_merge.pt"
OUT_BASE = REPO / "out" / "viz_v9"

KWARGS = dict(
    patch_size=5, d_model=128, n_heads=4, n_layers=8, ff=256,
    dropout=0.1, use_lead_emb=False, pos_type="learnable", conv_stem=False,
)


def predict_one(model, sig, lead_idx, device):
    sig = sig.astype(np.float32)
    sig_n = ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)
    raw = predict_frames(model, sig_n, lead_idx, device=device)
    pp = post_process_frames(raw, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    bands = frames_to_bands(pp, len(sig))
    return bds, bands


def viz_ludb(model, device, n_records=4, leads=("i", "ii", "v2", "v5")):
    out_dir = OUT_BASE / "ludb"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_ids = ludb.load_split()["val"][:n_records]
    ds = LUDBFrameDataset(rec_ids)
    for rid in rec_ids:
        for lead in leads:
            try:
                sig_250, lead_idx, _ = ds.cache[(rid, lead)]
            except KeyError: continue
            sig = sig_250[:WINDOW_SAMPLES]
            if len(sig) < WINDOW_SAMPLES: continue
            bds, bands = predict_one(model, sig, lead_idx, device)
            try:
                gt_500 = ludb.load_annotations(rid, lead)
            except Exception: continue
            gt_250 = {k: [int(s // 2) for s in v] for k, v in gt_500.items()}
            gt_bands = ann_to_bands(gt_250, 0, len(sig))
            gt_lines = ann_to_lines(gt_250, 0, len(sig))
            nk, nk_bands, wt, wt_bands = baselines_for_signal(sig)
            tracks = [
                ("GT (cardio)", gt_bands, gt_lines),
                ("v9_merge",    bands, bds),
                ("NeuroKit",    nk_bands, nk),
                ("WTdel",       wt_bands, wt),
            ]
            plot_panel(sig, tracks, f"LUDB val rid={rid} lead={lead}",
                        out_dir / f"{rid:03d}_{lead}.png")
            print(f"  ludb/{rid:03d}_{lead}.png", flush=True)


def viz_isp(model, device, n_records=4, leads=("i", "ii", "v2", "v5")):
    out_dir = OUT_BASE / "isp"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_ids = isp.load_split()["test"][:n_records]
    for rid in rec_ids:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception: continue
        for lead in leads:
            if lead not in record: continue
            lead_idx = isp.LEADS_12.index(lead)
            sig_1000 = record[lead]
            sig_250 = _decimate_to_250(sig_1000, 1000)
            sig_n = _normalize(sig_250)
            if len(sig_n) < WINDOW_SAMPLES:
                pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                sig_n = np.concatenate([sig_n, pad])
            sig = sig_n[:WINDOW_SAMPLES]
            bds, bands = predict_one(model, sig, lead_idx, device)
            ann_250 = {k: [int(s // 4) for s in v if int(s // 4) < WINDOW_SAMPLES]
                        for k, v in ann.items()}
            gt_bands = ann_to_bands(ann_250, 0, len(sig))
            gt_lines = ann_to_lines(ann_250, 0, len(sig))
            nk, nk_bands, wt, wt_bands = baselines_for_signal(sig)
            tracks = [
                ("GT (2-cardio)", gt_bands, gt_lines),
                ("v9_merge",      bands, bds),
                ("NeuroKit",      nk_bands, nk),
                ("WTdel",         wt_bands, wt),
            ]
            plot_panel(sig, tracks, f"ISP test rid={rid} lead={lead}",
                        out_dir / f"{rid}_{lead}.png")
            print(f"  isp/{rid}_{lead}.png", flush=True)


def viz_qtdb(model, device, n_records=6):
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
        if not all_q1c: continue
        margin = 2 * FS
        win_lo = max(0, min(all_q1c) - margin)
        win_hi = min(n, max(all_q1c) + margin)
        if win_hi - win_lo > WINDOW_SAMPLES:
            mid = (win_lo + win_hi) // 2
            win_lo = max(0, mid - WINDOW_SAMPLES // 2)
            win_hi = win_lo + WINDOW_SAMPLES
        sig_win = sig_full[win_lo:win_hi]
        n_win = len(sig_win)
        if n_win < WINDOW_SAMPLES:
            pad = np.zeros(WINDOW_SAMPLES - n_win, dtype=sig_win.dtype)
            sig_for_v4 = np.concatenate([sig_win, pad])
        else:
            sig_for_v4 = sig_win[:WINDOW_SAMPLES]
        bds, bands = predict_one(model, sig_for_v4, 1, device)
        bds = {k: [s for s in v if s < n_win] for k, v in bds.items()}
        bands = [(s, e, c) for s, e, c in bands if s < n_win]
        nk, nk_bands, wt, wt_bands = baselines_for_signal(sig_win)
        q1c_bands = ann_to_bands(ann_q1c, win_lo, win_hi)
        q1c_lines = ann_to_lines(ann_q1c, win_lo, win_hi)
        pu0_bands = ann_to_bands(ann_pu0, win_lo, win_hi)
        pu0_lines = ann_to_lines(ann_pu0, win_lo, win_hi)
        tracks = [
            ("GT q1c (expert)", q1c_bands, q1c_lines),
            ("GT pu0 (auto)",   pu0_bands, pu0_lines),
            ("v9_merge",        bands, bds),
            ("NeuroKit",        nk_bands, nk),
            ("WTdel",           wt_bands, wt),
        ]
        plot_panel(sig_win, tracks,
                    f"QTDB rid={rid} lead={first_lead} (samples [{win_lo}, {win_hi}])",
                    out_dir / f"{rid}_{first_lead}.png")
        print(f"  qtdb/{rid}_{first_lead}.png", flush=True)


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    device = "cpu"  # avoid GPU contention with concurrent training
    print(f"Loading {CKPT} on {device}", flush=True)
    model = FrameClassifierViT(**KWARGS)
    load_checkpoint(CKPT, model)
    model = model.to(device).train(False)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
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
