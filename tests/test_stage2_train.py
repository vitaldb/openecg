import tempfile
from pathlib import Path

import numpy as np
import torch

from openecg.stage2.model import FrameClassifier
from openecg.stage2.train import (
    TrainConfig,
    train_one_epoch,
    save_checkpoint,
    load_checkpoint,
    score_val_metrics,
)
from openecg import eval as ee
from openecg.stage2.infer import load_model_bundle


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
    for _ in range(60):
        loss = train_one_epoch(model, loader, opt, weights, device="cpu")
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.5, f"loss did not halve: {losses[0]:.3f} -> {losses[-1]:.3f}"


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


def test_checkpoint_contains_inference_metadata():
    model = FrameClassifier(d_model=128, n_layers=2)
    cfg = TrainConfig(epochs=1)
    metrics = {"val_score": 0.7, "epoch": 0}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(
            path,
            model,
            metrics,
            cfg,
            postprocess_config={"min_duration_ms": 60},
        )
        bundle = load_model_bundle(path, device="cpu")
        assert bundle["model_config"]["d_model"] == 128
        assert bundle["model_config"]["n_layers"] == 2
        assert bundle["postprocess_config"] == {"min_duration_ms": 60}
        assert bundle["metrics"] == metrics


def test_score_val_metrics_mean_wave_f1():
    metrics = {
        ee.SUPER_OTHER: {"f1": 0.99},
        ee.SUPER_P: {"f1": 0.6},
        ee.SUPER_QRS: {"f1": 0.9},
        ee.SUPER_T: {"f1": 0.75},
    }
    assert score_val_metrics(metrics, "mean_wave_f1") == np.mean([0.6, 0.9, 0.75])
    assert score_val_metrics(metrics, "qrs_f1") == 0.9
