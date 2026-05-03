"""Train Stage 2 model on LUDB train split, save best checkpoint."""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgcode import ludb
from ecgcode.stage2.dataset import LUDBFrameDataset, compute_class_weights
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import TrainConfig, fit

CKPT_PATH = Path("data/checkpoints/stage2_v1.pt")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    split = ludb.load_split()
    train_ids = split["train"]
    val_ids = split["val"]
    print(f"Train records: {len(train_ids)}, Val records: {len(val_ids)}")

    print("Loading train dataset...")
    t0 = time.time()
    train_ds = LUDBFrameDataset(train_ids)
    print(f"  {len(train_ds)} sequences in {time.time()-t0:.1f}s")

    print("Loading val dataset...")
    t0 = time.time()
    val_ds = LUDBFrameDataset(val_ids)
    print(f"  {len(val_ds)} sequences in {time.time()-t0:.1f}s")

    counts = train_ds.label_counts()
    print(f"Train labels: other={counts[0]} P={counts[1]} QRS={counts[2]} T={counts[3]}")
    weights = compute_class_weights(counts)
    print(f"Class weights: {weights.round(3)}")
    weights_t = torch.tensor(weights, dtype=torch.float32)

    cfg = TrainConfig()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = FrameClassifier()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("Training...")
    t0 = time.time()
    best = fit(model, train_loader, val_loader, weights_t, cfg,
               device=device, ckpt_path=CKPT_PATH)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best metrics: {best}")
    print(f"Checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()
