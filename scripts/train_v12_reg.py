# scripts/train_v12_reg.py
"""v12_reg - same training data + ViT as v9, plus a parallel boundary
regression head. Sweeps lambda in {0.05, 0.1, 0.5}; the best LUDB val avg F1 wins.

See docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.2.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from openecg.stage2.infer import (
    apply_reg_to_boundaries, extract_boundaries,
    post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset, _decimate_to_250, _normalize,
)
from openecg.stage2.reg_targets import RegLabelDataset
from openecg.stage2.train import TrainConfig, fit_reg, load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS, _ConcatWithCounts

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500
FRAME_MS = 20
FS = 250
SEED = 42
EDGE_MARGIN_MS = 100
QTDB_EVAL_SEED = 42
QTDB_PU0_WINDOWS_PER_RECORD = 5
LAMBDAS = (0.05, 0.1, 0.5)


def _eval_ludb(model, device):
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
        frames, reg = predict_frames_with_reg(model, sig_250, lead_idx, device=device)
        pp = post_process_frames(frames, frame_ms=FRAME_MS)
        boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
        boundaries = apply_reg_to_boundaries(boundaries, reg, max_window=WINDOW_SAMPLES)
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
            frames, reg = predict_frames_with_reg(model, sig_n, lead_idx, device=device)
            pp = post_process_frames(frames, frame_ms=FRAME_MS)
            boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            boundaries = apply_reg_to_boundaries(boundaries, reg, max_window=WINDOW_SAMPLES)
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
            frames, reg = predict_frames_with_reg(model, sig_w, lead_id=1, device=device)
            pp = post_process_frames(frames, frame_ms=FRAME_MS)
            boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            boundaries = apply_reg_to_boundaries(boundaries, reg, max_window=WINDOW_SAMPLES)
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


def _eval_all(model, device):
    return {
        "ludb_edge_filtered": _avg_f1(*_eval_ludb(model, device)),
        "isp_test":           _avg_f1(*_eval_isp(model, device)),
        "qtdb_pu0_random":    _avg_f1(*_eval_qtdb(model, device)),
    }


def _build_train_loader():
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=SEED,
                                       q1c_pu_merge=True)
    reg_ludb = RegLabelDataset(ludb_train)
    reg_isp = RegLabelDataset(isp_train)
    reg_qtdb = RegLabelDataset(qtdb_merged)
    return _ConcatWithCounts([reg_ludb, reg_isp, reg_qtdb])


def main():
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_ds = _build_train_loader()
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                 mask_unlabeled_edges=True,
                                 edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)

    sweep_results = {}
    for lam in LAMBDAS:
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                    num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(ludb_val, batch_size=64, shuffle=False,
                                  num_workers=0, pin_memory=True)
        model = FrameClassifierViTReg(**KWARGS, n_reg=6)
        n_params = sum(p.numel() for p in model.parameters())
        ckpt_path = CKPT_DIR / f"stage2_v12_reg_lam{lam}.pt"
        print(f"\n=== TRAIN v12_reg lambda={lam} ({n_params:,} params) ===",
              flush=True)
        t0 = time.time()
        best = fit_reg(model, train_loader, val_loader, weights, cfg, device=device,
                         ckpt_path=ckpt_path, lambda_reg=lam)
        elapsed = time.time() - t0
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model)
        model = model.to(device).train(False)
        res = _eval_all(model, device)
        sweep_results[str(lam)] = {"params": n_params, "train_seconds": elapsed,
                                     "best_metrics": best, **res}
        print(f"  lambda={lam} eval: {res}", flush=True)

    best_lam = max(sweep_results, key=lambda l: sweep_results[l]["ludb_edge_filtered"])
    print(f"\n=== BEST lambda = {best_lam} ===", flush=True)
    best_ckpt = CKPT_DIR / f"stage2_v12_reg_lam{best_lam}.pt"
    canon_ckpt = CKPT_DIR / "stage2_v12_reg.pt"
    if canon_ckpt.exists(): canon_ckpt.unlink()
    canon_ckpt.write_bytes(best_ckpt.read_bytes())

    full = {
        "v9_q1c_pu_merge_ref": {"params": 1126660,
                                  "ludb_edge_filtered": 0.923,
                                  "isp_test": 0.943,
                                  "qtdb_pu0_random": 0.779},
        "v12_reg_sweep": sweep_results,
        "v12_reg_best": {"lambda": float(best_lam), **sweep_results[best_lam]},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v12_reg_{ts}.json"
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
