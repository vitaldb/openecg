"""Train v4 with QTDB sliding-window + on-the-fly time stretch.

QTDB has 15-min continuous recordings; LUDB and ISP only 10s isolated. The
single-window QTDB loading wastes the continuous context. This script uses:

  - LUDB train + ISP train: existing 10s isolated windows (no time aug, since
    compression with padding hurts LUDB/ISP-aug isolated)
  - QTDB: sliding window over full 15-min recording with on-the-fly stretch
    (0.9-1.2, including compression — safe because we have neighbor samples)

Two variants:
  C_qsliding         : LUDB+ISP (no aug) + QTDBSliding (10 windows/record-lead)
  C_qsliding_more    : same but 30 windows/record-lead (3x more QTDB samples)

Reference: existing C ckpt.
Eval: LUDB val + ISP test, full + masked (boundary 2s excluded for cleaner metric).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ecgcode import eval as ee, isp, ludb
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.infer import (
    BOUNDARY_SHIFT_C, extract_boundaries, post_process_frames, predict_frames,
)
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from ecgcode.stage2.train import TrainConfig, fit, load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
SEED = 42
MASK_SAMPLES = 500  # 2s @ 250Hz, for boundary-masked evaluation
TOL = {"p_on": 50, "p_off": 50, "qrs_on": 40, "qrs_off": 40, "t_on": 50, "t_off": 100}


def signed_metrics(pred, true, tol_ms, fs=250):
    tol_samples = tol_ms * fs / 1000.0
    pred_arr = np.sort(np.array(pred, dtype=int))
    true_arr = np.sort(np.array(true, dtype=int))
    matched = set()
    errs = []
    for t in true_arr:
        best, best_abs = -1, float("inf")
        for jj, p in enumerate(pred_arr):
            if jj in matched: continue
            d = abs(int(p) - int(t))
            if d < best_abs:
                best_abs = d; best = jj
        if best >= 0 and best_abs <= tol_samples:
            matched.add(best)
            errs.append(int(pred_arr[best]) - int(t))
    sens = len(errs) / len(true_arr) if len(true_arr) > 0 else 0.0
    ppv = len(errs) / len(pred_arr) if len(pred_arr) > 0 else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    if errs:
        e = np.array(errs) * 1000.0 / fs
        return {"f1": f1, "sens": sens, "ppv": ppv,
                "mean_signed_ms": float(np.mean(e)), "sd_ms": float(np.std(e))}
    return {"f1": f1, "sens": sens, "ppv": ppv, "mean_signed_ms": 0.0, "sd_ms": 0.0}


def evaluate_dual(model, device, shift, kind):
    """kind in {'ludb', 'isp'}. Returns (full_metrics, masked_metrics, n_seqs)."""
    bp_full, bt_full = defaultdict(list), defaultdict(list)
    bp_mask, bt_mask = defaultdict(list), defaultdict(list)
    cum = 0; n = 0

    if kind == "ludb":
        val_ds = LUDBFrameDataset(ludb.load_split()["val"])
        with torch.no_grad():
            for idx in range(len(val_ds)):
                rid, lead = val_ds.items[idx]
                sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
                sig_250 = sig_250[:WINDOW_SAMPLES_250]
                if len(sig_250) < WINDOW_SAMPLES_250: continue
                pred = predict_frames(model, sig_250, lead_idx, device=device)
                pp = post_process_frames(pred, frame_ms=FRAME_MS)
                bds = extract_boundaries(pp, fs=250, frame_ms=FRAME_MS, boundary_shift_ms=shift)
                for k, v in bds.items():
                    for x in v:
                        bp_full[k].append(int(x) + cum)
                        if MASK_SAMPLES <= int(x) < WINDOW_SAMPLES_250 - MASK_SAMPLES:
                            bp_mask[k].append(int(x) + cum)
                try:
                    gt_ann = ludb.load_annotations(rid, lead)
                    for k, v in gt_ann.items():
                        if k.endswith("_on") or k.endswith("_off"):
                            for s in v:
                                s250 = int(s // 2)
                                if 0 <= s250 < WINDOW_SAMPLES_250:
                                    bt_full[k].append(s250 + cum)
                                    if MASK_SAMPLES <= s250 < WINDOW_SAMPLES_250 - MASK_SAMPLES:
                                        bt_mask[k].append(s250 + cum)
                except Exception:
                    pass
                cum += WINDOW_SAMPLES_250
                n += 1
    else:  # isp
        rec_ids = isp.load_split()["test"]
        with torch.no_grad():
            for rid in rec_ids:
                try:
                    record = isp.load_record(rid, split="test")
                    ann = isp.load_annotations_as_super(rid, split="test")
                except Exception:
                    continue
                for lead_idx, lead in enumerate(isp.LEADS_12):
                    from ecgcode.stage2.multi_dataset import _decimate_to_250, _normalize
                    sig_1000 = record[lead]
                    sig_250 = _decimate_to_250(sig_1000, 1000)
                    sig_n = _normalize(sig_250)
                    if len(sig_n) < WINDOW_SAMPLES_250:
                        pad = np.zeros(WINDOW_SAMPLES_250 - len(sig_n), dtype=sig_n.dtype)
                        sig_n = np.concatenate([sig_n, pad])
                    sig_n = sig_n[:WINDOW_SAMPLES_250]
                    pred = predict_frames(model, sig_n, lead_idx, device=device)
                    pp = post_process_frames(pred, frame_ms=FRAME_MS)
                    bds = extract_boundaries(pp, fs=250, frame_ms=FRAME_MS, boundary_shift_ms=shift)
                    for k, v in bds.items():
                        for x in v:
                            bp_full[k].append(int(x) + cum)
                            if MASK_SAMPLES <= int(x) < WINDOW_SAMPLES_250 - MASK_SAMPLES:
                                bp_mask[k].append(int(x) + cum)
                    for k, v in ann.items():
                        if k.endswith("_on") or k.endswith("_off"):
                            for s in v:
                                s250 = int(s // 4)
                                if 0 <= s250 < WINDOW_SAMPLES_250:
                                    bt_full[k].append(s250 + cum)
                                    if MASK_SAMPLES <= s250 < WINDOW_SAMPLES_250 - MASK_SAMPLES:
                                        bt_mask[k].append(s250 + cum)
                    cum += WINDOW_SAMPLES_250
                    n += 1

    def _m(bp, bt):
        return {k: signed_metrics(bp.get(k, []), bt.get(k, []), TOL[k]) for k in TOL}

    return _m(bp_full, bt_full), _m(bp_mask, bt_mask), n


class _ConcatWithCounts(ConcatDataset):
    """ConcatDataset that aggregates label_counts from sub-datasets."""
    def label_counts(self):
        from ecgcode.stage2.multi_dataset import N_CLASSES
        total = np.zeros(N_CLASSES, dtype=np.int64)
        for d in self.datasets:
            if hasattr(d, "label_counts"):
                lc = d.label_counts()
                if lc is not None:
                    total += lc[:N_CLASSES]
        return np.maximum(total, 1)


def train_model(name, train_ds, ludb_val, device, ckpt):
    print(f"\n=== TRAIN {name} ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    model = FrameClassifier(d_model=128, n_layers=8)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}, train n={len(train_ds)}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=ckpt, use_focal=False)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
    if ckpt and Path(ckpt).exists():
        load_checkpoint(ckpt, model)
    return model.to(device).train(False), elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"])
    ludb_isp = CombinedFrameDataset(["ludb_train", "isp_train"])
    print(f"  LUDB+ISP train: {len(ludb_isp)}, LUDB val: {len(ludb_val)}", flush=True)

    qtdb_sliding_x10 = QTDBSlidingDataset(scale_factors=(0.9, 1.0, 1.1, 1.2),
                                           windows_per_record=10, seed=SEED)
    qtdb_sliding_x30 = QTDBSlidingDataset(scale_factors=(0.9, 1.0, 1.1, 1.2),
                                           windows_per_record=30, seed=SEED+1)

    ds_x10 = _ConcatWithCounts([ludb_isp, qtdb_sliding_x10])
    ds_x30 = _ConcatWithCounts([ludb_isp, qtdb_sliding_x30])
    print(f"  Mixed x10: {len(ds_x10)}, Mixed x30: {len(ds_x30)}", flush=True)

    runs = [
        ("C_qsliding_x10",  ds_x10, CKPT_DIR / "stage2_v4_C_qsliding_x10.pt"),
        ("C_qsliding_x30",  ds_x30, CKPT_DIR / "stage2_v4_C_qsliding_x30.pt"),
    ]

    full = {}

    # Reference (existing C) eval only
    print(f"\n{'='*78}\n>>> C_ref (existing) <<<\n{'='*78}", flush=True)
    ref_ckpt = CKPT_DIR / "stage2_v4_C.pt"
    model = FrameClassifier(d_model=128, n_layers=8)
    load_checkpoint(ref_ckpt, model)
    model = model.to(device).train(False)
    shift = BOUNDARY_SHIFT_C
    lf, lm, _ = evaluate_dual(model, device, shift, "ludb")
    isf, ism, _ = evaluate_dual(model, device, shift, "isp")
    full["C_ref"] = {
        "ludb_full_avg": float(np.mean([lf[k]["f1"] for k in TOL])),
        "ludb_mask_avg": float(np.mean([lm[k]["f1"] for k in TOL])),
        "isp_full_avg":  float(np.mean([isf[k]["f1"] for k in TOL])),
        "isp_mask_avg":  float(np.mean([ism[k]["f1"] for k in TOL])),
        "ludb_full": lf, "ludb_mask": lm, "isp_full": isf, "isp_mask": ism,
    }
    print(f"  LUDB full {full['C_ref']['ludb_full_avg']:.3f}, mask {full['C_ref']['ludb_mask_avg']:.3f}; "
          f"ISP full {full['C_ref']['isp_full_avg']:.3f}, mask {full['C_ref']['isp_mask_avg']:.3f}", flush=True)

    for name, train_ds, ckpt in runs:
        print(f"\n{'='*78}\n>>> {name} <<<\n{'='*78}", flush=True)
        model, elapsed = train_model(name, train_ds, ludb_val, device, ckpt)
        lf, lm, _ = evaluate_dual(model, device, shift, "ludb")
        isf, ism, _ = evaluate_dual(model, device, shift, "isp")
        full[name] = {
            "train_seconds": elapsed,
            "ludb_full_avg": float(np.mean([lf[k]["f1"] for k in TOL])),
            "ludb_mask_avg": float(np.mean([lm[k]["f1"] for k in TOL])),
            "isp_full_avg":  float(np.mean([isf[k]["f1"] for k in TOL])),
            "isp_mask_avg":  float(np.mean([ism[k]["f1"] for k in TOL])),
            "ludb_full": lf, "ludb_mask": lm, "isp_full": isf, "isp_mask": ism,
        }
        print(f"  LUDB full {full[name]['ludb_full_avg']:.3f}, mask {full[name]['ludb_mask_avg']:.3f}; "
              f"ISP full {full[name]['isp_full_avg']:.3f}, mask {full[name]['isp_mask_avg']:.3f}", flush=True)

    print("\n" + "="*86, flush=True)
    print(f"{'QTDB sliding window + time stretch (avg Martinez F1, -22ms p_off shift)':^86}", flush=True)
    print("="*86, flush=True)
    print(f"  {'model':22s}  {'LUDB full':>10s}  {'LUDB mask':>10s}  {'ISP full':>10s}  {'ISP mask':>10s}", flush=True)
    base_lf = full["C_ref"]["ludb_full_avg"]
    base_lm = full["C_ref"]["ludb_mask_avg"]
    base_if = full["C_ref"]["isp_full_avg"]
    base_im = full["C_ref"]["isp_mask_avg"]
    for name, r in full.items():
        dlf = r["ludb_full_avg"] - base_lf
        dlm = r["ludb_mask_avg"] - base_lm
        dif = r["isp_full_avg"] - base_if
        dim = r["isp_mask_avg"] - base_im
        print(f"  {name:22s}  {r['ludb_full_avg']:.3f} ({dlf:+.3f})  {r['ludb_mask_avg']:.3f} ({dlm:+.3f})  "
              f"{r['isp_full_avg']:.3f} ({dif:+.3f})  {r['isp_mask_avg']:.3f} ({dim:+.3f})", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v4_qtdb_sliding_{ts}.json"
    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
