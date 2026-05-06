"""Train F (LUDB only, no lead_emb) with ckpt save, then cross-domain eval F vs G.

F = stage2_v4_ludb_only.pt    : LUDB only,           d=64/L=4, no lead_emb
G = stage2_v4_candidate.pt    : LUDB+QTDB+ISP train, d=64/L=4, no lead_emb (already trained)

Eval all on: LUDB val (in-domain for both), QTDB ext, ISP test.
Question: does F generalize across domains, or is combined training necessary?
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
from openecg.stage2.multi_dataset import CombinedFrameDataset
from openecg.stage2.train import TrainConfig, fit, load_checkpoint

OUT_DIR = Path("out")
CKPT_DIR = Path("data/checkpoints")
F_CKPT = CKPT_DIR / "stage2_v4_ludb_only.pt"
G_CKPT = CKPT_DIR / "stage2_v4_candidate.pt"
EPOCHS = 50
PATIENCE = 10
SEED = 42


def per_lead_f1(model, val_ds, device):
    model.train(False)
    overall_pred, overall_true = [], []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            sig, lead_id, labels = val_ds[idx]
            x = sig.unsqueeze(0).to(device)
            lid = lead_id.unsqueeze(0).to(device)
            logits = model(x, lid)
            pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1).astype(np.uint8)
            true = labels.numpy().reshape(-1).astype(np.uint8)
            overall_pred.append(pred)
            overall_true.append(true)
    overall = ee.frame_f1(np.concatenate(overall_pred), np.concatenate(overall_true))
    return {
        "P": overall[ee.SUPER_P]["f1"],
        "QRS": overall[ee.SUPER_QRS]["f1"],
        "T": overall[ee.SUPER_T]["f1"],
        "other": overall[ee.SUPER_OTHER]["f1"],
        "n_seqs": len(val_ds),
    }


def train_f(ludb_train, ludb_val, device):
    print(f"\n{'='*70}\nTRAIN F: LUDB only, d=64/L=4, no lead_emb, save ckpt\n{'='*70}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    counts = ludb_train.label_counts()
    weights = torch.tensor(compute_class_weights(counts), dtype=torch.float32)
    cfg = TrainConfig(epochs=EPOCHS, batch_size=64, lr=1e-3, early_stop_patience=PATIENCE)
    train_loader = DataLoader(ludb_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = FrameClassifier(d_model=64, n_layers=4, use_lead_emb=False)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights, cfg, device=device,
               ckpt_path=F_CKPT, use_focal=False)
    print(f"  trained in {time.time()-t0:.1f}s, best={best}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading datasets...", flush=True)
    t0 = time.time()
    train_ids = ludb.load_split()["train"]
    val_ids = ludb.load_split()["val"]
    ludb_train = LUDBFrameDataset(train_ids)
    ludb_val = LUDBFrameDataset(val_ids)
    print(f"  LUDB train/val: {len(ludb_train)}/{len(ludb_val)} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    qtdb_test = CombinedFrameDataset(["qtdb"])
    print(f"  QTDB ext: {len(qtdb_test)} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    isp_test = CombinedFrameDataset(["isp_test"])
    print(f"  ISP test: {len(isp_test)} ({time.time()-t0:.1f}s)", flush=True)

    # Step 1: train F (if not already done)
    if not F_CKPT.exists():
        train_f(ludb_train, ludb_val, device)
    else:
        print(f"\nF checkpoint exists at {F_CKPT}, skipping training.", flush=True)

    # Step 2: load both F and G, evaluate across domains
    results = {}
    for label, ckpt in [("F_ludb_only", F_CKPT), ("G_combined", G_CKPT)]:
        print(f"\n{'='*70}\nEVAL {label} from {ckpt}\n{'='*70}", flush=True)
        if not ckpt.exists():
            print(f"  MISSING checkpoint!", flush=True)
            results[label] = {"error": f"missing checkpoint {ckpt}"}
            continue
        model = FrameClassifier(d_model=64, n_layers=4, use_lead_emb=False)
        load_checkpoint(ckpt, model)
        model = model.to(device)
        domain_metrics = {}
        for dname, ds in [("ludb_val", ludb_val), ("qtdb_ext", qtdb_test), ("isp_test", isp_test)]:
            m = per_lead_f1(model, ds, device)
            domain_metrics[dname] = m
            print(f"  {dname:10s} (n={m['n_seqs']:4d}): "
                  f"P={m['P']:.3f} QRS={m['QRS']:.3f} T={m['T']:.3f} other={m['other']:.3f}",
                  flush=True)
        results[label] = domain_metrics

    summary = {"settings": results}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"v4_cross_domain_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"{'CROSS-DOMAIN COMPARISON: F (LUDB-only) vs G (combined), no lead_emb, frame F1':^78}", flush=True)
    print("=" * 78, flush=True)
    print(f"{'domain':12s} {'F P/QRS/T':>22s}    {'G P/QRS/T':>22s}    {'delta P/QRS/T':>22s}", flush=True)
    for d in ("ludb_val", "qtdb_ext", "isp_test"):
        f_m = results.get("F_ludb_only", {}).get(d, {})
        g_m = results.get("G_combined", {}).get(d, {})
        if not f_m or not g_m:
            continue
        f_str = f"{f_m['P']:.3f}/{f_m['QRS']:.3f}/{f_m['T']:.3f}"
        g_str = f"{g_m['P']:.3f}/{g_m['QRS']:.3f}/{g_m['T']:.3f}"
        d_str = f"{g_m['P']-f_m['P']:+.3f}/{g_m['QRS']-f_m['QRS']:+.3f}/{g_m['T']-f_m['T']:+.3f}"
        print(f"{d:12s} {f_str:>22s}    {g_str:>22s}    {d_str:>22s}", flush=True)


if __name__ == "__main__":
    main()
