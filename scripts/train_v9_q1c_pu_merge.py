"""v9_q1c_pu_merge — single experiment focused on the T-wave fix.

Train v7_pure_q1c-equivalent model but with QTDB q1c labels enriched by
filling missing wave-onsets from pu0 (matched per-beat by QRS_on proximity).

Architecture: ViT, d=128, L=8, patch=5, learnable pos, no_lead_emb (= v7).
Train data: LUDB train (edge-masked) + ISP train + QTDB q1c+pu0 merged.

Reference: v7_pure_q1c (eval cached).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from ecgcode.stage2.infer import (
    extract_boundaries, post_process_frames, predict_frames,
)
from ecgcode.stage2.model import FrameClassifierViT
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset, _decimate_to_250, _normalize,
)
from ecgcode.stage2.train import TrainConfig, fit, load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES = 2500
FRAME_MS = 20
FS = 250
SEED = 42
EDGE_MARGIN_MS = 100
QTDB_EVAL_SEED = 42
QTDB_PU0_WINDOWS_PER_RECORD = 5

KWARGS = dict(
    patch_size=5, d_model=128, n_heads=4, n_layers=8, ff=256,
    dropout=0.1, use_lead_emb=False, pos_type="learnable", conv_stem=False,
)


class _ConcatWithCounts(ConcatDataset):
    def label_counts(self):
        from ecgcode.stage2.multi_dataset import N_CLASSES
        total = np.zeros(N_CLASSES, dtype=np.int64)
        for d in self.datasets:
            if hasattr(d, "label_counts"):
                lc = d.label_counts()
                if lc is not None:
                    total += lc[:N_CLASSES]
        return np.maximum(total, 1)


def eval_model_ludb(model, device):
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    margin_250 = int(round(EDGE_MARGIN_MS * FS / 1000.0))
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES]
            if len(sig_250) < WINDOW_SAMPLES: continue
            rng_lab = ludb.labeled_range(rid, lead)
            if rng_lab is None: continue
            lo = max(0, rng_lab[0] // 2 - margin_250)
            hi = min(WINDOW_SAMPLES, rng_lab[1] // 2 + margin_250 + 1)
            raw = predict_frames(model, sig_250, lead_idx, device=device)
            pp = post_process_frames(raw, frame_ms=FRAME_MS)
            for k, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
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


def eval_model_isp(model, device):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    with torch.no_grad():
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
                raw = predict_frames(model, sig_n, lead_idx, device=device)
                pp = post_process_frames(raw, frame_ms=FRAME_MS)
                for k, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
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


def eval_model_qtdb_pu0(model, device, n_windows=QTDB_PU0_WINDOWS_PER_RECORD):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    rng = np.random.default_rng(QTDB_EVAL_SEED)
    with torch.no_grad():
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
                raw = predict_frames(model, sig_w, lead_id=1, device=device)
                pp = post_process_frames(raw, frame_ms=FRAME_MS)
                for ck, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
                    for s in vs:
                        bp[ck].append(int(start + s) + cum)
            for ck in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
                for s in ann.get(ck, []):
                    for lo, hi in covered:
                        if lo <= s < hi:
                            bt[ck].append(int(s) + cum)
                            break
            cum += n
    return bp, bt


def avg_f1(bp, bt):
    f1s = []
    for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []),
                                      tolerance_ms=MARTINEZ_TOLERANCE_MS[k], fs=FS)
        f1s.append(m["f1"])
    return float(np.mean(f1s))


def eval_all(model, device):
    return {
        "ludb_edge_filtered": avg_f1(*eval_model_ludb(model, device)),
        "isp_test":           avg_f1(*eval_model_isp(model, device)),
        "qtdb_pu0_random":    avg_f1(*eval_model_qtdb_pu0(model, device)),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)

    print("Building train dataset (q1c+pu0 merge)...", flush=True)
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=SEED,
                                       q1c_pu_merge=True)
    print(f"  LUDB train: {len(ludb_train)}, ISP train: {len(isp_train)}, "
          f"QTDB q1c+pu0: {len(qtdb_merged)}", flush=True)
    train_ds = _ConcatWithCounts([ludb_train, isp_train, qtdb_merged])

    name = "v9_q1c_pu_merge"
    print(f"\n=== TRAIN {name} ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = FrameClassifierViT(**KWARGS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  config={KWARGS}", flush=True)
    print(f"  params={n_params:,}, train n={len(train_ds)}", flush=True)
    t0 = time.time()
    ckpt_path = CKPT_DIR / f"stage2_{name}.pt"
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=ckpt_path, use_focal=False)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)
    res = eval_all(model, device)
    print(f"\n=== {name} eval ===", flush=True)
    for k, v in res.items(): print(f"  {k}: {v:.3f}")

    # Summary
    full = {
        "v7_pure_q1c_ref": {"params": 1126660,
                              "ludb_edge_filtered": 0.923,
                              "isp_test": 0.941,
                              "qtdb_pu0_random": 0.747},
        name: {"params": n_params, "train_seconds": elapsed,
                "config": KWARGS, **res},
    }
    print("\n" + "=" * 78, flush=True)
    print(f"{'q1c+pu0 MERGE EXPERIMENT — avg Martinez F1':^78}", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'model':18s} {'params':>10s} | {'LUDB':>8s} {'ISP':>8s} {'QTDB pu0':>10s}", flush=True)
    for k, r in full.items():
        print(f"  {k:18s} {r['params']:>10,d} | "
              f"{r['ludb_edge_filtered']:>8.3f} {r['isp_test']:>8.3f} "
              f"{r['qtdb_pu0_random']:>10.3f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v9_q1c_pu_merge_{ts}.json"
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
