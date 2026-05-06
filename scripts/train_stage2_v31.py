# scripts/train_stage2_v31.py
"""v3.1: QTDB lead-id fix + no augmentation + weighted CE."""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openecg.stage2.multi_dataset import CombinedFrameDataset
from openecg.stage2.dataset import compute_class_weights
from openecg.stage2.model import FrameClassifier
from openecg.stage2.train import TrainConfig, fit

CKPT_PATH = Path("data/checkpoints/stage2_v31.pt")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading TRAIN (combined LUDB+QTDB+ISP, NO augmentation)...", flush=True)
    t0 = time.time()
    train_ds = CombinedFrameDataset(['ludb_train', 'qtdb', 'isp_train'])
    print(f"  TRAIN: {len(train_ds)} sequences in {time.time()-t0:.1f}s", flush=True)
    print(f"  source_counts: {dict(train_ds.source_counts())}", flush=True)

    print("Loading VAL (LUDB val + ISP test)...", flush=True)
    t0 = time.time()
    val_ds = CombinedFrameDataset(['ludb_val', 'isp_test'])
    print(f"  VAL: {len(val_ds)} sequences in {time.time()-t0:.1f}s", flush=True)

    counts = train_ds.label_counts()
    print(f"Train labels: other={counts[0]} P={counts[1]} QRS={counts[2]} T={counts[3]}", flush=True)
    weights = compute_class_weights(counts)
    print(f"Class weights: {weights.round(3)}", flush=True)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    cfg = TrainConfig(epochs=100, batch_size=64, lr=1e-3, early_stop_patience=20)
    print(f"Config: epochs={cfg.epochs} patience={cfg.early_stop_patience} loss=weighted_CE (no focal, no aug)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model = FrameClassifier(d_model=128, n_layers=8)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    print("Training...", flush=True)
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights_t, cfg,
               device=device, ckpt_path=CKPT_PATH, use_focal=False)
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Best metrics: {best}", flush=True)


if __name__ == "__main__":
    main()
