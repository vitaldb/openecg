"""Retrain v12_reg (lambda=0.05 only) with the qtdb cluster-window fixes.

Previous lambda sweep found 0.05/0.1/0.5 all within 0.005 F1 of each other on
all three eval sets, so a single lambda is enough to confirm the fixes don't
regress headline numbers and to compare per-record silent-rate before/after.

Outputs
- data/checkpoints/stage2_v12_reg_after_qtdb_fix.pt  (new canonical for the
  fix verification — the previous stage2_v12_reg.pt is left in place so we
  can A/B against it.)
- out/retrain_v12_reg_after_qtdb_fix_<ts>.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import ludb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.train import TrainConfig, fit_reg, load_checkpoint
from scripts.train_v12_reg import (
    EDGE_MARGIN_MS, LAMBDAS, SEED, _build_train_loader, _eval_all,
)
from scripts.train_v9_q1c_pu_merge import KWARGS

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
LAMBDA = 0.05  # previous winner on LUDB val


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_ds = _build_train_loader()
    ludb_val = LUDBFrameDataset(
        ludb.load_split()["val"],
        mask_unlabeled_edges=True,
        edge_margin_ms=EDGE_MARGIN_MS,
    )
    weights = torch.tensor(
        compute_class_weights(train_ds.label_counts()), dtype=torch.float32,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ludb_val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True,
    )
    model = FrameClassifierViTReg(**KWARGS, n_reg=6)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_path = CKPT_DIR / "stage2_v12_reg_after_qtdb_fix.pt"
    print(f"\n=== TRAIN v12_reg (after qtdb fix) lambda={LAMBDA} "
          f"({n_params:,} params) ===", flush=True)

    t0 = time.time()
    best = fit_reg(
        model, train_loader, val_loader, weights, cfg,
        device=device, ckpt_path=ckpt_path, lambda_reg=LAMBDA,
    )
    elapsed = time.time() - t0

    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)

    res = _eval_all(model, device)
    payload = {
        "params": n_params,
        "lambda": LAMBDA,
        "train_seconds": elapsed,
        "best_metrics": best,
        "eval": res,
        "v12_reg_baseline_ref": {
            "ludb_edge_filtered": 0.949,
            "isp_test": 0.967,
            "qtdb_pu0_random": 0.827,
        },
    }

    def _safe(v):
        if isinstance(v, dict):
            return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"retrain_v12_reg_after_qtdb_fix_{ts}.json"
    import json
    out_path.write_text(json.dumps(_safe(payload), indent=2))
    print(f"\nSaved {out_path}", flush=True)
    print(f"\nFinal eval: {res}", flush=True)


if __name__ == "__main__":
    main()
