"""Train a bigger v4 model to test if LUDB gap is capacity-limited.

Current C: d=128/L=8 (1.08M params)
Test: d=192/L=10 (~2.5M params, ~2.3x bigger)
Same data (combined LUDB+QTDB+ISP), same loss (CE), same lead_emb=True.

If LUDB val Se improves materially (≥3pp), capacity is the limit and we should
scale up. If not (<1pp), data scale is the limit and we need more LUDB-like
data (Stage 4 SSL, augmentation, etc.).

Eval: boundary F1 with Martinez tolerances on LUDB val + ISP test.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openecg import eval as ee, isp, ludb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.infer import (
    extract_boundaries, post_process_frames, predict_frames,
)
from openecg.stage2.model import FrameClassifier
from openecg.stage2.multi_dataset import CombinedFrameDataset
from openecg.stage2.train import TrainConfig, fit, load_checkpoint

CKPT_DIR = Path("data/checkpoints")
OUT_DIR = Path("out")
WINDOW_SAMPLES_250 = 2500
WINDOW_FRAMES = 500
FRAME_MS = 20
SEED = 42

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


def evaluate_ludb(model, device, shift):
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    with torch.no_grad():
        for idx in range(len(val_ds)):
            rid, lead = val_ds.items[idx]
            sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
            sig_250 = sig_250[:WINDOW_SAMPLES_250]
            if len(sig_250) < WINDOW_SAMPLES_250: continue
            pred = predict_frames(model, sig_250, lead_idx, device=device)
            pp = post_process_frames(pred, frame_ms=FRAME_MS)
            b = extract_boundaries(pp, fs=250, frame_ms=FRAME_MS)
            for k, v in b.items():
                bp[k].extend(int(x) + cum for x in v)
            try:
                gt_ann = ludb.load_annotations(rid, lead)
                for k, v in gt_ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 2)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                bt[k].append(s250 + cum)
            except Exception:
                pass
            cum += WINDOW_SAMPLES_250
    out = {}
    for k in TOL:
        out[k] = signed_metrics(bp.get(k, []), bt.get(k, []), TOL[k])
    return out, len(val_ds)


def evaluate_isp(model, device, shift):
    rec_ids = isp.load_split()["test"]
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    with torch.no_grad():
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split="test")
                ann = isp.load_annotations_as_super(rid, split="test")
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                from openecg.stage2.multi_dataset import _decimate_to_250, _normalize
                sig_1000 = record[lead]
                sig_250 = _decimate_to_250(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                if len(sig_n) < WINDOW_SAMPLES_250:
                    pad = np.zeros(WINDOW_SAMPLES_250 - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                sig_n = sig_n[:WINDOW_SAMPLES_250]
                pred = predict_frames(model, sig_n, lead_idx, device=device)
                pp = post_process_frames(pred, frame_ms=FRAME_MS)
                b = extract_boundaries(pp, fs=250, frame_ms=FRAME_MS)
                for k, v in b.items():
                    bp[k].extend(int(x) + cum for x in v)
                for k, v in ann.items():
                    if k.endswith("_on") or k.endswith("_off"):
                        for s in v:
                            s250 = int(s // 4)
                            if 0 <= s250 < WINDOW_SAMPLES_250:
                                bt[k].append(s250 + cum)
                cum += WINDOW_SAMPLES_250
                n += 1
    out = {}
    for k in TOL:
        out[k] = signed_metrics(bp.get(k, []), bt.get(k, []), TOL[k])
    return out, n


def train_model(name, model_kwargs, train_ds, ludb_val, device, ckpt):
    print(f"\n=== TRAIN {name}: {model_kwargs} ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)
    cfg = TrainConfig(epochs=50, batch_size=64, lr=1e-3, early_stop_patience=10)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    model = FrameClassifier(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=ckpt, use_focal=False)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
    if ckpt and Path(ckpt).exists():
        load_checkpoint(ckpt, model)
    return model.to(device).train(False), n_params, elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"])
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"])
    print(f"  LUDB train/val: {len(ludb_train)}/{len(ludb_val)}", flush=True)
    combined_train = CombinedFrameDataset(["ludb_train", "qtdb", "isp_train"])
    print(f"  Combined train: {len(combined_train)}", flush=True)

    candidates = [
        ("C_d128_L8 (existing)", None, {"d_model": 128, "n_layers": 8},
         CKPT_DIR / "stage2_v4_C.pt"),
        ("Cbig_d192_L10",        combined_train, {"d_model": 192, "n_layers": 10},
         CKPT_DIR / "stage2_v4_C_d192L10.pt"),
        ("Cbig_d256_L8",         combined_train, {"d_model": 256, "n_layers": 8},
         CKPT_DIR / "stage2_v4_C_d256L8.pt"),
    ]

    full = {}
    for name, train_ds, mk, ckpt in candidates:
        print(f"\n{'='*78}\n>>> {name} <<<\n{'='*78}", flush=True)
        if not ckpt.exists() and train_ds is not None:
            model, n_params, elapsed = train_model(name, mk, train_ds, ludb_val, device, ckpt)
        else:
            print(f"  Loading existing ckpt: {ckpt}", flush=True)
            model = FrameClassifier(**mk)
            load_checkpoint(ckpt, model)
            model = model.to(device).train(False)
            n_params = sum(p.numel() for p in model.parameters())
            elapsed = 0
        # Evaluate with C-style p_off shift (it's the dominant pattern)
        t0 = time.time()
        ludb_metrics, n_ludb = evaluate_ludb(model, device, shift)
        print(f"\n  [LUDB val, {n_ludb} seqs, {time.time()-t0:.1f}s, shift={shift}]", flush=True)
        for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
            m = ludb_metrics[k]
            print(f"    {k:8s}: F1={m['f1']:.3f} Se={m['sens']*100:5.1f}% P+={m['ppv']*100:5.1f}% "
                  f"mean={m['mean_signed_ms']:+5.1f}ms SD={m['sd_ms']:5.1f}ms", flush=True)
        avg_f1 = float(np.mean([ludb_metrics[k]["f1"] for k in TOL]))
        print(f"    AVG F1 (Martinez tol): {avg_f1:.3f}", flush=True)

        t0 = time.time()
        isp_metrics, n_isp = evaluate_isp(model, device, shift)
        print(f"\n  [ISP test, {n_isp} seqs, {time.time()-t0:.1f}s]", flush=True)
        for k in ("p_on", "qrs_on", "t_on", "p_off", "qrs_off", "t_off"):
            m = isp_metrics[k]
            print(f"    {k:8s}: F1={m['f1']:.3f} Se={m['sens']*100:5.1f}% P+={m['ppv']*100:5.1f}% "
                  f"mean={m['mean_signed_ms']:+5.1f}ms SD={m['sd_ms']:5.1f}ms", flush=True)
        avg_isp = float(np.mean([isp_metrics[k]["f1"] for k in TOL]))
        print(f"    AVG F1 (Martinez tol): {avg_isp:.3f}", flush=True)

        full[name] = {"n_params": n_params, "train_seconds": elapsed,
                      "ludb_val": ludb_metrics, "isp_test": isp_metrics,
                      "ludb_avg_f1": avg_f1, "isp_avg_f1": avg_isp}

    # Summary
    print("\n" + "="*78, flush=True)
    print(f"{'Model size scan: avg Martinez F1 with -22ms p_off shift':^78}", flush=True)
    print("="*78, flush=True)
    print(f"  {'model':28s} {'params':>10s}  {'LUDB avg F1':>12s}  {'ISP avg F1':>12s}", flush=True)
    for name, r in full.items():
        print(f"  {name:28s} {r['n_params']/1e6:>9.2f}M  {r['ludb_avg_f1']:>11.3f}   {r['isp_avg_f1']:>11.3f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v4_bigger_{ts}.json"
    def _safe(o):
        if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
