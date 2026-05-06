"""Ablate v3 regression: which factor caused LUDB val F1 to drop vs v1.0?

5 settings, all evaluated on LUDB val (in-domain). Reports overall F1 + per-lead F1.
Goal: time-axis region segmentation only (other/P/QRS/T per frame); morphology
variants are explicitly out of scope.

Settings:
  A: LUDB only            | d=128/L=8 | CE        | lead_emb=on
  B: LUDB+QTDB+ISP        | d=64/L=4  | CE        | lead_emb=on
  C: LUDB+QTDB+ISP        | d=128/L=8 | CE        | lead_emb=on
  D: LUDB+QTDB+ISP        | d=128/L=8 | focal+aug | lead_emb=on  (v3 reproduce)
  E: LUDB+QTDB+ISP        | d=64/L=4  | CE        | lead_emb=OFF (lead-agnostic)

Compare to:
  v1.0  : LUDB only       | d=64/L=4  | CE        | lead_emb=on
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openecg import eval as ee
from openecg import ludb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.model import FrameClassifier
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset,
    CombinedFrameDatasetAugmented,
)
from openecg.stage2.train import TrainConfig, fit

OUT_DIR = Path("out")
EPOCHS = 50
PATIENCE = 10
SEED = 42


def per_lead_f1(model, val_ds, device):
    """Compute per-lead frame F1 on LUDB val (each item has known lead_idx)."""
    model.train(False)
    by_lead = {i: {"pred": [], "true": []} for i in range(12)}
    overall_pred, overall_true = [], []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            sig, lead_id, labels = val_ds[idx]
            x = sig.unsqueeze(0).to(device)
            lid = lead_id.unsqueeze(0).to(device)
            logits = model(x, lid)
            pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1).astype(np.uint8)
            true = labels.numpy().reshape(-1).astype(np.uint8)
            li = int(lead_id.item())
            by_lead[li]["pred"].append(pred)
            by_lead[li]["true"].append(true)
            overall_pred.append(pred)
            overall_true.append(true)
    overall = ee.frame_f1(np.concatenate(overall_pred), np.concatenate(overall_true))
    per_lead = {}
    for li, buf in by_lead.items():
        if not buf["pred"]:
            continue
        f1 = ee.frame_f1(np.concatenate(buf["pred"]), np.concatenate(buf["true"]))
        per_lead[ludb.LEADS_12[li]] = {
            "P": f1[ee.SUPER_P]["f1"],
            "QRS": f1[ee.SUPER_QRS]["f1"],
            "T": f1[ee.SUPER_T]["f1"],
            "n_seqs": len(buf["pred"]),
        }
    return {
        "overall": {
            "P": overall[ee.SUPER_P]["f1"],
            "QRS": overall[ee.SUPER_QRS]["f1"],
            "T": overall[ee.SUPER_T]["f1"],
            "other": overall[ee.SUPER_OTHER]["f1"],
        },
        "per_lead": per_lead,
    }


def run_setting(name, train_ds, val_ds, model_kwargs, use_focal, device):
    print(f"\n{'='*70}\nSETTING {name}: model={model_kwargs}, focal={use_focal}\n{'='*70}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    counts = train_ds.label_counts()
    weights = compute_class_weights(counts)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    print(f"  train n_seqs={len(train_ds)}, val n_seqs={len(val_ds)}", flush=True)
    print(f"  class_weights={weights.round(3)}", flush=True)

    cfg = TrainConfig(epochs=EPOCHS, batch_size=64, lr=1e-3, early_stop_patience=PATIENCE)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = FrameClassifier(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}", flush=True)

    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights_t, cfg, device=device,
               ckpt_path=None, use_focal=use_focal)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)

    model = model.to(device)
    metrics = per_lead_f1(model, val_ds, device)
    print(f"  overall F1 P/QRS/T = {metrics['overall']['P']:.3f} / "
          f"{metrics['overall']['QRS']:.3f} / {metrics['overall']['T']:.3f}", flush=True)
    print("  per-lead F1 (P/QRS/T):", flush=True)
    for lead, m in metrics["per_lead"].items():
        print(f"    {lead:5s}: {m['P']:.3f} / {m['QRS']:.3f} / {m['T']:.3f}  (n={m['n_seqs']})", flush=True)

    return {
        "name": name,
        "model_kwargs": model_kwargs,
        "use_focal": use_focal,
        "n_params": n_params,
        "n_train": len(train_ds),
        "train_seconds": elapsed,
        "best_during_training": best,
        "ludb_val": metrics,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("\nLoading datasets (one-time)...", flush=True)
    t0 = time.time()
    train_ids = ludb.load_split()["train"]
    val_ids = ludb.load_split()["val"]
    ludb_train = LUDBFrameDataset(train_ids)
    ludb_val = LUDBFrameDataset(val_ids)
    print(f"  LUDB train: {len(ludb_train)} seqs, val: {len(ludb_val)} seqs ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    combined_train = CombinedFrameDataset(["ludb_train", "qtdb", "isp_train"])
    print(f"  Combined train: {len(combined_train)} seqs ({time.time()-t0:.1f}s)", flush=True)
    print(f"  source_counts: {dict(combined_train.source_counts())}", flush=True)

    t0 = time.time()
    combined_train_aug = CombinedFrameDatasetAugmented(
        ["ludb_train", "qtdb", "isp_train"], n_ops=2, seed=SEED,
    )
    print(f"  Combined train (aug): {len(combined_train_aug)} seqs ({time.time()-t0:.1f}s)", flush=True)

    settings = [
        ("A_ludb_big",        ludb_train,         {"d_model": 128, "n_layers": 8}, False),
        ("B_combined_small",  combined_train,     {"d_model": 64,  "n_layers": 4}, False),
        ("C_combined_big",    combined_train,     {"d_model": 128, "n_layers": 8}, False),
        ("D_v3_repro",        combined_train_aug, {"d_model": 128, "n_layers": 8}, True),
        ("E_no_lead_emb",     combined_train,     {"d_model": 64,  "n_layers": 4, "use_lead_emb": False}, False),
    ]

    results = {}
    for name, train_ds, mk, focal in settings:
        try:
            results[name] = run_setting(name, train_ds, ludb_val, mk, focal, device)
        except Exception as ex:
            results[name] = {"error": repr(ex)}
            print(f"  FAILED: {ex}", flush=True)

    summary = {
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "settings": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"ablate_v3_regression_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("FINAL COMPARISON (LUDB val overall F1)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'setting':22s} {'P':>6s} {'QRS':>6s} {'T':>6s}  notes", flush=True)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:22s} ERROR  {r['error']}", flush=True)
            continue
        o = r["ludb_val"]["overall"]
        print(f"{name:22s} {o['P']:.3f}  {o['QRS']:.3f}  {o['T']:.3f}  "
              f"({r['n_params']/1000:.0f}K params, {r['train_seconds']:.0f}s)",
              flush=True)
    print(f"\nReference v1.0: P=0.604 QRS=0.806 T=0.695 (d=64/L=4, LUDB only, CE)", flush=True)
    print(f"Reference v3:   P=0.458 QRS=0.622 T=0.548 (= setting D)", flush=True)


if __name__ == "__main__":
    main()
