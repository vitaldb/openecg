"""v12_soft - same training data + ViT as v9_q1c_pu_merge, but soft labels at
transitions and KL loss instead of CE.

See docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.1.
"""
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
from ecgcode.stage2.model import FrameClassifierViT
from ecgcode.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from ecgcode.stage2.soft_labels import SoftLabelDataset
from ecgcode.stage2.train import TrainConfig, fit_kl, load_checkpoint
# Reuse v9 eval helpers exactly so numbers are directly comparable
from scripts.train_v9_q1c_pu_merge import _ConcatWithCounts, eval_all, KWARGS

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
SEED = 42
EDGE_MARGIN_MS = 100
ALPHA = 0.7


def main():
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                 mask_unlabeled_edges=True,
                                 edge_margin_ms=EDGE_MARGIN_MS)

    print("Building train dataset (q1c+pu0 merge, soft labels)...", flush=True)
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=SEED,
                                       q1c_pu_merge=True)

    soft_ludb = SoftLabelDataset(ludb_train, alpha=ALPHA)
    soft_isp = SoftLabelDataset(isp_train, alpha=ALPHA)
    soft_qtdb = SoftLabelDataset(qtdb_merged, alpha=ALPHA)
    train_ds = _ConcatWithCounts([soft_ludb, soft_isp, soft_qtdb])

    name = "v12_soft"
    print(f"\n=== TRAIN {name} ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()),
                            dtype=torch.float32)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = FrameClassifierViT(**KWARGS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  config={KWARGS}", flush=True)
    print(f"  params={n_params:,}, train n={len(train_ds)}, alpha={ALPHA}",
          flush=True)
    t0 = time.time()
    ckpt_path = CKPT_DIR / f"stage2_{name}.pt"
    best = fit_kl(model, train_loader, val_loader, weights, cfg, device=device,
                   ckpt_path=ckpt_path)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s, best={best}", flush=True)
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
                "config": KWARGS, "alpha": ALPHA, **res},
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
