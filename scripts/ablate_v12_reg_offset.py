"""Ablation: v12_reg checkpoint, eval WITHOUT apply_reg_to_boundaries.

Isolates the multi-task-training-on-backbone effect from the inference-offset
effect. Compares three numbers per dataset:

  v9_q1c_pu_merge        — baseline (cls-only model, cls-only eval)
  v12_reg classifier-only — v12_reg model, cls argmax only (skip reg offset)
  v12_reg full           — v12_reg model, cls argmax + reg offset applied

Δ_train = (v12_reg classifier-only) − (v9 baseline)   ← multi-task gain
Δ_post  = (v12_reg full) − (v12_reg classifier-only)  ← post-processing gain
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset
from openecg.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from openecg.stage2.infer import (
    extract_boundaries, post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.multi_dataset import _decimate_to_250, _normalize
from openecg.stage2.train import load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500
FRAME_MS = 20
FS = 250
EDGE_MARGIN_MS = 100
QTDB_EVAL_SEED = 42
QTDB_PU0_WINDOWS_PER_RECORD = 5


def _eval_ludb(model, device):
    """Same as v12_reg's _eval_ludb but does NOT call apply_reg_to_boundaries."""
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    margin_250 = int(round(EDGE_MARGIN_MS * FS / 1000.0))
    for idx in range(len(val_ds)):
        rid, lead = val_ds.items[idx]
        sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
        sig_250 = sig_250[:WINDOW_SAMPLES]
        if len(sig_250) < WINDOW_SAMPLES: continue
        rng_lab = ludb.labeled_range(rid, lead)
        if rng_lab is None: continue
        lo = max(0, rng_lab[0] // 2 - margin_250)
        hi = min(WINDOW_SAMPLES, rng_lab[1] // 2 + margin_250 + 1)
        frames, _reg = predict_frames_with_reg(model, sig_250, lead_idx, device=device)
        pp = post_process_frames(frames, frame_ms=FRAME_MS)
        boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
        # NOTE: NO apply_reg_to_boundaries — that's the whole point of the ablation
        for k, vs in boundaries.items():
            for s in vs:
                if lo <= s < hi:
                    bp[k].append(int(s) + cum)
        gt = ludb.load_annotations(rid, lead)
        for k, vs in gt.items():
            if k.endswith("_on") or k.endswith("_off"):
                for s in vs:
                    s250 = int(s // 2)
                    if lo <= s250 < hi:
                        bt[k].append(s250 + cum)
        cum += WINDOW_SAMPLES
    return bp, bt


def _eval_isp(model, device):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    for rid in isp.load_split()["test"]:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception: continue
        for lead_idx, lead in enumerate(isp.LEADS_12):
            sig_1000 = record[lead]
            sig_250 = _decimate_to_250(sig_1000, 1000)
            sig_n = _normalize(sig_250)
            if len(sig_n) < WINDOW_SAMPLES:
                pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                sig_n = np.concatenate([sig_n, pad])
            sig_n = sig_n[:WINDOW_SAMPLES]
            frames, _reg = predict_frames_with_reg(model, sig_n, lead_idx, device=device)
            pp = post_process_frames(frames, frame_ms=FRAME_MS)
            boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            for k, vs in boundaries.items():
                for s in vs:
                    bp[k].append(int(s) + cum)
            for k, vs in ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in vs:
                        s250 = int(s // 4)
                        if 0 <= s250 < WINDOW_SAMPLES:
                            bt[k].append(s250 + cum)
            cum += WINDOW_SAMPLES
    return bp, bt


def _eval_qtdb(model, device, n_windows=QTDB_PU0_WINDOWS_PER_RECORD):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    rng = np.random.default_rng(QTDB_EVAL_SEED)
    for rid in qtdb.records_with_q1c():
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_pu(rid, lead=0)
        except Exception: continue
        first_lead = list(record.keys())[0]
        sig_full = record[first_lead].astype(np.float32)
        n = len(sig_full)
        sig_norm = _normalize(sig_full)
        n_max = n // WINDOW_SAMPLES
        k = min(n_windows, n_max)
        chosen = rng.choice(n_max, size=k, replace=False)
        covered = []
        for w in sorted(chosen):
            start = int(w) * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            covered.append((start, end))
            sig_win = sig_norm[start:end].astype(np.float32)
            sig_w = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)
            frames, _reg = predict_frames_with_reg(model, sig_w, lead_id=1, device=device)
            pp = post_process_frames(frames, frame_ms=FRAME_MS)
            boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            for ck, vs in boundaries.items():
                for s in vs:
                    bp[ck].append(int(start + s) + cum)
        for ck in ("p_on","p_off","qrs_on","qrs_off","t_on","t_off"):
            for s in ann.get(ck, []):
                for lo, hi in covered:
                    if lo <= s < hi:
                        bt[ck].append(int(s) + cum); break
        cum += n
    return bp, bt


def _avg_f1(bp, bt):
    f1s = []
    for k in ("p_on","p_off","qrs_on","qrs_off","t_on","t_off"):
        m = signed_boundary_metrics(
            bp.get(k, []), bt.get(k, []),
            tolerance_ms=MARTINEZ_TOLERANCE_MS[k], fs=FS,
        )
        f1s.append(m["f1"])
    return float(np.mean(f1s))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    ckpt_path = CKPT_DIR / "stage2_v12_reg.pt"
    print(f"Loading {ckpt_path}", flush=True)
    model = FrameClassifierViTReg(**KWARGS, n_reg=6)
    load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)

    t0 = time.time()
    res = {
        "ludb_edge_filtered": _avg_f1(*_eval_ludb(model, device)),
        "isp_test":           _avg_f1(*_eval_isp(model, device)),
        "qtdb_pu0_random":    _avg_f1(*_eval_qtdb(model, device)),
    }
    elapsed = time.time() - t0
    print(f"\n=== v12_reg classifier-only (no offset) — {elapsed:.1f}s ===", flush=True)
    for k, v in res.items(): print(f"  {k}: {v:.3f}")

    # Side-by-side
    full = {
        "v9_q1c_pu_merge_ref": {"ludb_edge_filtered": 0.923,
                                  "isp_test": 0.943,
                                  "qtdb_pu0_random": 0.779},
        "v12_reg_full_ref":    {"ludb_edge_filtered": 0.949,
                                  "isp_test": 0.967,
                                  "qtdb_pu0_random": 0.827},
        "v12_reg_cls_only":    {**res, "elapsed_seconds": elapsed},
    }
    print("\n" + "=" * 78, flush=True)
    print(f"{'v12_reg ablation: cls-only vs full (with reg offset)':^78}", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'run':22s} | {'LUDB':>8s} {'ISP':>8s} {'QTDB pu0':>10s}", flush=True)
    print("  " + "-" * 60, flush=True)
    for k, r in full.items():
        print(f"  {k:22s} | {r['ludb_edge_filtered']:>8.3f} "
              f"{r['isp_test']:>8.3f} {r['qtdb_pu0_random']:>10.3f}", flush=True)

    print("\n  Δ_train (cls-only − v9):", flush=True)
    for m in ("ludb_edge_filtered", "isp_test", "qtdb_pu0_random"):
        d = res[m] - full["v9_q1c_pu_merge_ref"][m]
        print(f"    {m:25s} {d:+.3f}", flush=True)
    print("\n  Δ_post  (full − cls-only):", flush=True)
    for m in ("ludb_edge_filtered", "isp_test", "qtdb_pu0_random"):
        d = full["v12_reg_full_ref"][m] - res[m]
        print(f"    {m:25s} {d:+.3f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"ablate_v12_reg_offset_{ts}.json"
    out_path.write_text(json.dumps(full, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
