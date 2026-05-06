"""v12_hubert_lp / v12_hubert_ft - HuBERT-ECG transfer.

Usage:
    python scripts/train_v12_hubert.py --mode lp
    python scripts/train_v12_hubert.py --mode ft

See docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md sec 5.2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from ecgcode import isp, ludb, qtdb
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from ecgcode.stage2.ssl.head import BackboneWithHeads
from ecgcode.stage2.ssl.hubert import HUBERT_DEFAULT_MODEL_ID, HubertECGAdapter
from ecgcode.stage2.train import (
    TrainConfig, load_checkpoint, run_eval, save_checkpoint,
    score_val_metrics, train_one_epoch,
)
from ecgcode import eval as ecg_eval
from scripts.train_v9_q1c_pu_merge import _ConcatWithCounts, eval_all

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
SEED = 42
EDGE_MARGIN_MS = 100


def _build_train(seed):
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=seed,
                                       q1c_pu_merge=True)
    return _ConcatWithCounts([ludb_train, isp_train, qtdb_merged])


def _fit_with_groups(model, train_loader, val_loader, weights, cfg,
                      device, ckpt_path, param_groups, log_fn=print):
    """Like fit() but accepts pre-built optimizer param_groups (LP / FT split)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * cfg.warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best = -1; best_metrics = None; bad = 0
    for epoch in range(cfg.epochs):
        tl = train_one_epoch(model, train_loader, optimizer, weights, device,
                              use_focal=False, grad_clip=cfg.grad_clip,
                              scheduler=scheduler)
        val = run_eval(model, val_loader, device)
        score = score_val_metrics(val, cfg.early_stop_metric)
        log_fn(f"epoch {epoch:3d} train={tl:.4f} score={score:.3f}")
        if score > best:
            best = score
            best_metrics = {"epoch": epoch, "val_score": score,
                              "val_qrs_f1": val[ecg_eval.SUPER_QRS]["f1"]}
            bad = 0
            if ckpt_path: save_checkpoint(ckpt_path, model, best_metrics, cfg)
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                log_fn(f"Early stop at {epoch}"); break
    return best_metrics or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lp", "ft"), required=True)
    ap.add_argument("--model-id", default=HUBERT_DEFAULT_MODEL_ID)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, mode={args.mode}", flush=True)

    train_ds = _build_train(SEED)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()),
                            dtype=torch.float32)

    torch.manual_seed(SEED); np.random.seed(SEED)
    backbone = HubertECGAdapter(model_id=args.model_id, device=device)
    model = BackboneWithHeads(backbone, hidden_dim=backbone.hidden_dim, use_reg=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model_id={args.model_id} hidden={backbone.hidden_dim} params={n_params:,}",
          flush=True)

    if args.mode == "lp":
        for p in backbone.parameters():
            p.requires_grad = False
        param_groups = [{"params": model.cls_head.parameters(), "lr": 1e-3}]
    else:
        param_groups = [
            {"params": backbone.parameters(), "lr": 1e-5},
            {"params": model.cls_head.parameters(), "lr": 1e-3},
        ]

    cfg = TrainConfig(epochs=args.epochs, batch_size=32, lr=1e-3,
                       early_stop_patience=7, warmup_frac=0.05)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)

    name = f"v12_hubert_{args.mode}"
    ckpt_path = CKPT_DIR / f"stage2_{name}.pt"
    t0 = time.time()
    best = _fit_with_groups(model, train_loader, val_loader, weights, cfg,
                              device, ckpt_path, param_groups)
    elapsed = time.time() - t0

    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)
    res = eval_all(model, device)
    print(f"\n=== {name} eval ===", flush=True)
    for k, v in res.items(): print(f"  {k}: {v:.3f}")

    full = {
        "v9_q1c_pu_merge_ref": {"params": 1126660,
                                  "ludb_edge_filtered": 0.923,
                                  "isp_test": 0.943,
                                  "qtdb_pu0_random": 0.779},
        name: {"params": n_params, "train_seconds": elapsed,
                "model_id": args.model_id,
                "mode": args.mode, "best_metrics": best, **res},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_{name}_{ts}.json"
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
