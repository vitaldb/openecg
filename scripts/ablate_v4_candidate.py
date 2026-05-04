"""Two follow-up ablations after ablate_v3_regression:

  F: LUDB only        | d=64/L=4 | CE | lead_emb=OFF
     -> Does lead_emb help even in single-dataset setup?
     -> v1.0 used lead_emb=ON and got P=0.604 / QRS=0.806 / T=0.695
     -> If F is comparable, lead_emb adds nothing.

  G: LUDB+QTDB+ISP    | d=64/L=4 | CE | lead_emb=OFF | best-ckpt save+load
     -> v4 candidate: best ablate_v3_regression setup (E) but eval on best epoch.
     -> If G beats E (last-epoch), then ckpt loading was the missing piece.

All evaluation: LUDB val (in-domain), overall + per-lead F1.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgcode import eval as ee
from ecgcode import ludb
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.multi_dataset import CombinedFrameDataset
from ecgcode.stage2.train import TrainConfig, fit, load_checkpoint

OUT_DIR = Path("out")
CKPT_DIR = Path("data/checkpoints")
EPOCHS = 50
PATIENCE = 10
SEED = 42


def per_lead_f1(model, val_ds, device):
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


def run_setting(name, train_ds, val_ds, model_kwargs, use_focal, device, ckpt_path=None):
    print(f"\n{'='*70}\nSETTING {name}: model={model_kwargs}, focal={use_focal}, "
          f"ckpt_save={'yes' if ckpt_path else 'no'}\n{'='*70}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    counts = train_ds.label_counts()
    weights = compute_class_weights(counts)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    print(f"  train n_seqs={len(train_ds)}, val n_seqs={len(val_ds)}", flush=True)

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
               ckpt_path=ckpt_path, use_focal=use_focal)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)

    # If checkpoint saved, reload it for true best-epoch evaluation
    if ckpt_path is not None and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model)
        print(f"  reloaded best checkpoint from {ckpt_path}", flush=True)
    model = model.to(device)
    metrics = per_lead_f1(model, val_ds, device)
    print(f"  overall F1 P/QRS/T = {metrics['overall']['P']:.3f} / "
          f"{metrics['overall']['QRS']:.3f} / {metrics['overall']['T']:.3f}", flush=True)
    print("  per-lead F1 (P/QRS/T):", flush=True)
    for lead, m in metrics["per_lead"].items():
        print(f"    {lead:5s}: {m['P']:.3f} / {m['QRS']:.3f} / {m['T']:.3f}", flush=True)

    return {
        "name": name,
        "model_kwargs": model_kwargs,
        "use_focal": use_focal,
        "n_params": n_params,
        "n_train": len(train_ds),
        "train_seconds": elapsed,
        "best_during_training": best,
        "ckpt_loaded_for_eval": ckpt_path is not None,
        "ludb_val": metrics,
    }


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
    print(f"  LUDB train: {len(ludb_train)}, val: {len(ludb_val)} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    combined_train = CombinedFrameDataset(["ludb_train", "qtdb", "isp_train"])
    print(f"  Combined train: {len(combined_train)} ({time.time()-t0:.1f}s)", flush=True)

    settings = [
        # F: LUDB-only without lead_emb. Reference: v1.0 (with lead_emb) = P=0.604/QRS=0.806/T=0.695
        ("F_ludb_no_lead_emb", ludb_train, {"d_model": 64, "n_layers": 4, "use_lead_emb": False}, False, None),
        # G: v4 candidate. Combined + no lead_emb + ckpt save (true best-epoch eval).
        ("G_v4_candidate", combined_train, {"d_model": 64, "n_layers": 4, "use_lead_emb": False}, False,
         CKPT_DIR / "stage2_v4_candidate.pt"),
    ]

    results = {}
    for name, train_ds, mk, focal, ckpt in settings:
        try:
            results[name] = run_setting(name, train_ds, ludb_val, mk, focal, device, ckpt_path=ckpt)
        except Exception as ex:
            results[name] = {"error": repr(ex)}
            print(f"  FAILED: {ex}", flush=True)

    summary = {"epochs": EPOCHS, "patience": PATIENCE, "settings": results}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"ablate_v4_candidate_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("FINAL COMPARISON (LUDB val overall F1)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'setting':22s} {'P':>6s} {'QRS':>6s} {'T':>6s}  notes", flush=True)
    print(f"{'v1.0 ref (lead_emb)':22s}  0.604  0.806  0.695  (LUDB only, last-epoch)", flush=True)
    print(f"{'A_ludb_big (lead_emb)':22s}  0.652  0.812  0.711  (from prior ablate, last-epoch)", flush=True)
    print(f"{'E_no_lead_emb':22s}  0.556  0.743  0.642  (combined, last-epoch)", flush=True)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:22s} ERROR  {r['error']}", flush=True)
            continue
        o = r["ludb_val"]["overall"]
        ckpt_note = "best-ckpt" if r["ckpt_loaded_for_eval"] else "last-epoch"
        print(f"{name:22s} {o['P']:.3f}  {o['QRS']:.3f}  {o['T']:.3f}  ({ckpt_note})", flush=True)


if __name__ == "__main__":
    main()
