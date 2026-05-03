import tempfile
from pathlib import Path

import numpy as np
import torch

from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import (
    TrainConfig,
    train_one_epoch,
    save_checkpoint,
    load_checkpoint,
)


def _tiny_loader(n_samples=4):
    sigs = torch.randn(n_samples, 2500)
    leads = torch.randint(0, 12, (n_samples,))
    labels = torch.randint(0, 4, (n_samples, 500))
    return [(sigs, leads, labels)]


def test_train_one_epoch_loss_decreases():
    torch.manual_seed(0)
    model = FrameClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loader = _tiny_loader(n_samples=4)
    losses = []
    for _ in range(150):
        loss = train_one_epoch(model, loader, opt, weights, device="cpu")
        losses.append(loss)
    # Bigger model (1.5M params) overfitting tiny synthetic batch on CPU; require meaningful decrease.
    assert losses[-1] < losses[0] * 0.6, f"loss did not decrease enough: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_checkpoint_roundtrip():
    model = FrameClassifier()
    cfg = TrainConfig(epochs=1)
    metrics = {"val_qrs_f1": 0.7, "epoch": 0}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(path, model, metrics, cfg)
        model2 = FrameClassifier()
        loaded_metrics = load_checkpoint(path, model2)
        assert loaded_metrics == metrics
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
