"""v6 architecture ablation — test conv-stem hybrid vs pure ViT.

All variants share:
  - Size l: d_model=128, n_layers=8, ff=256
  - patch_size=5 (20ms)
  - learnable positional encoding
  - use_lead_emb=False (no lead distinction, treat all leads identically)
  - Same train data: LUDB train (edge-masked) + ISP train + QTDB pu0/pu1 sliding
  - Same training config (80 epochs, lr=1e-3, batch=64)

Variants:
  v6_pure       : pure ViT (5-sample linear patch embedding)
  v6_convstem   : Conv1d(1->16, k=7) + Conv1d(16->32, k=5) + 5-sample patch
                  (linear projection of 32 * 5 = 160 features → d_model)

Reference: v4 C ckpt for comparison.
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
    extract_boundaries, load_model_bundle, post_process_frames, predict_frames,
)
from ecgcode.stage2.model import FrameClassifierViT
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBPuFullDataset, _decimate_to_250, _normalize,
)
from ecgcode.stage2.train import TrainConfig, fit, load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
FS = 250
SEED = 42
EDGE_MARGIN_MS = 100
QTDB_EVAL_SEED = 42
QTDB_PU0_WINDOWS_PER_RECORD = 5

BASE_KWARGS = dict(
    patch_size=5,
    d_model=128,
    n_heads=4,
    n_layers=8,
    ff=256,
    dropout=0.1,
    use_lead_emb=False,
    pos_type="learnable",
)

VARIANTS = {
    "v6_pure":     {**BASE_KWARGS, "conv_stem": False},
    "v6_convstem": {**BASE_KWARGS, "conv_stem": True},
}


# ---------- Datasets ----------

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


def build_train_dataset():
    print("Building train dataset...", flush=True)
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    print(f"  LUDB train (edge-masked): {len(ludb_train)} sequences", flush=True)
    isp_train = CombinedFrameDataset(["isp_train"])
    print(f"  ISP train: {len(isp_train)} sequences", flush=True)
    qtdb_pu = QTDBPuFullDataset(windows_per_record=20, seed=SEED)
    print(f"  QTDB pu0/pu1 sliding: {len(qtdb_pu)} windows", flush=True)
    combined = _ConcatWithCounts([ludb_train, isp_train, qtdb_pu])
    print(f"  TOTAL: {len(combined)}", flush=True)
    return combined


# ---------- Eval ----------

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
            if len(sig_250) < WINDOW_SAMPLES:
                continue
            rng_lab = ludb.labeled_range(rid, lead)
            if rng_lab is None:
                continue
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
            except Exception:
                continue
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
            except Exception:
                continue
            first_lead = list(record.keys())[0]
            sig_full = record[first_lead].astype(np.float32)
            n = len(sig_full)
            sig_norm = _normalize(sig_full)
            n_max_windows = n // WINDOW_SAMPLES
            k = min(n_windows, n_max_windows)
            chosen = rng.choice(n_max_windows, size=k, replace=False)
            covered_ranges = []
            for w in sorted(chosen):
                start = int(w) * WINDOW_SAMPLES
                end = start + WINDOW_SAMPLES
                covered_ranges.append((start, end))
                sig_win = sig_norm[start:end].astype(np.float32)
                sig_w = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)
                raw = predict_frames(model, sig_w, lead_id=1, device=device)
                pp = post_process_frames(raw, frame_ms=FRAME_MS)
                for ck, vs in extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS).items():
                    for s in vs:
                        bp[ck].append(int(start + s) + cum)
            for ck in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
                for s in ann.get(ck, []):
                    for lo, hi in covered_ranges:
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


def eval_all_domains(model, device):
    return {
        "ludb_edge_filtered": avg_f1(*eval_model_ludb(model, device)),
        "isp_test":           avg_f1(*eval_model_isp(model, device)),
        "qtdb_pu0_random":    avg_f1(*eval_model_qtdb_pu0(model, device)),
    }


# ---------- Train ----------

def train_one(name, train_ds, val_ds, model_kwargs, device, ckpt_path):
    print(f"\n=== TRAIN {name} ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = FrameClassifierViT(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  config={model_kwargs}", flush=True)
    print(f"  params={n_params:,}, train n={len(train_ds)}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=ckpt_path, use_focal=False)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model)
    return model.to(device).train(False), elapsed, n_params


def main():
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    print(f"LUDB val (edge-masked for early-stop): {len(ludb_val)}", flush=True)
    train_ds = build_train_dataset()
    full = {}

    # --- Reference: v4 C ---
    print("\n" + "=" * 78, flush=True)
    print(f"{'>>> v4 C (reference, no retrain) <<<':^78}", flush=True)
    print("=" * 78, flush=True)
    ref_ckpt = CKPT_DIR / "stage2_v4_C.pt"
    bundle = load_model_bundle(ref_ckpt, device=device)
    ref_model = bundle["model"].train(False)
    ref_n_params = sum(p.numel() for p in ref_model.parameters())
    t0 = time.time()
    full["v4_C_ref"] = {"params": ref_n_params, **eval_all_domains(ref_model, device),
                         "eval_seconds": time.time() - t0}
    for k, v in full["v4_C_ref"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # --- v6 variants ---
    for name, model_kwargs in VARIANTS.items():
        print("\n" + "=" * 78, flush=True)
        print(f"{f'>>> {name} <<<':^78}", flush=True)
        print("=" * 78, flush=True)
        ckpt = CKPT_DIR / f"stage2_{name}.pt"
        model, elapsed, n_params = train_one(name, train_ds, ludb_val,
                                              model_kwargs, device, ckpt)
        t0 = time.time()
        results = eval_all_domains(model, device)
        full[name] = {"params": n_params, "train_seconds": elapsed,
                       "eval_seconds": time.time() - t0,
                       "config": model_kwargs, **results}
        for k, v in full[name].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            elif isinstance(v, int):
                print(f"  {k}: {v:,}")

    # --- Summary ---
    print("\n" + "=" * 100, flush=True)
    print(f"{'v6 ARCH ABLATION SUMMARY: avg Martinez F1 by domain':^100}", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'model':16s} {'params':>10s} | {'LUDB(edge)':>10s} {'ISP':>10s} {'QTDB pu0':>10s}", flush=True)
    for name, r in full.items():
        print(f"  {name:16s} {r['params']:>10,d} | "
              f"{r['ludb_edge_filtered']:>10.3f} {r['isp_test']:>10.3f} "
              f"{r['qtdb_pu0_random']:>10.3f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v6_arch_ablate_{ts}.json"

    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
