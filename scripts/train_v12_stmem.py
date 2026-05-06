"""v12_stmem_lp / v12_stmem_ft - ST-MEM transfer (single-lead replicate)."""
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
from ecgcode.stage2.ssl.stmem import STMEMAdapter
from ecgcode.stage2.train import TrainConfig, load_checkpoint
from scripts.train_v12_hubert import _fit_with_groups
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lp", "ft"), required=True)
    ap.add_argument("--weights", default=None,
                     help="Path to ST-MEM pretrained weights .pt")
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
    backbone = STMEMAdapter(weights_path=args.weights, device=device)
    model = BackboneWithHeads(backbone, hidden_dim=backbone.hidden_dim, use_reg=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  hidden={backbone.hidden_dim} params={n_params:,} weights={args.weights}",
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

    name = f"v12_stmem_{args.mode}"
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
                "weights": args.weights, "mode": args.mode,
                "best_metrics": best, **res},
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
