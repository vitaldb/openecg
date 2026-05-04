"""Stage 2 training loop with checkpointing and early stopping."""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecgcode import eval as ecg_eval


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    early_stop_patience: int = 10
    grad_clip: float = 1.0
    seed: int = 42


def focal_cross_entropy(logits, target, weight=None, gamma=2.0):
    """Focal cross-entropy loss for class imbalance.

    logits: [B, C, ...] (already permuted so class axis is dim=1)
    target: [B, ...] integer class labels
    weight: optional per-class weight tensor of shape [C]
    gamma: focusing parameter (paper recommends 2.0)
    """
    log_probs = nn.functional.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    nll = nn.functional.nll_loss(log_probs, target, weight=weight, reduction="none")
    pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    focal_factor = (1.0 - pt).pow(gamma)
    return (focal_factor * nll).mean()


def train_one_epoch(model, loader, optimizer, class_weights, device,
                    use_focal=True, focal_gamma=2.0):
    model.train()
    weights = class_weights.to(device)
    total_loss = 0.0
    n_batches = 0
    for sigs, leads, labels in loader:
        sigs = sigs.to(device)
        leads = leads.to(device)
        labels = labels.to(device)
        logits = model(sigs, leads)
        if use_focal:
            loss = focal_cross_entropy(
                logits.transpose(1, 2), labels, weight=weights, gamma=focal_gamma,
            )
        else:
            loss = nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, weight=weights,
            )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def run_eval(model, loader, device):
    """Run val pass; return per-class F1 (using ecgcode.eval.frame_f1)."""
    model.eval()
    all_pred = []
    all_true = []
    for sigs, leads, labels in loader:
        sigs = sigs.to(device)
        leads = leads.to(device)
        logits = model(sigs, leads)
        pred = logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        true = labels.numpy().astype(np.uint8)
        all_pred.append(pred.reshape(-1))
        all_true.append(true.reshape(-1))
    pred_concat = np.concatenate(all_pred)
    true_concat = np.concatenate(all_true)
    return ecg_eval.frame_f1(pred_concat, true_concat)


def save_checkpoint(path, model, metrics, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": asdict(config),
    }, path)


def load_checkpoint(path, model):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state"])
    return blob["metrics"]


def fit(model, train_loader, val_loader, class_weights, config,
        device="cuda", ckpt_path=None, log_fn=print, use_focal=True):
    """Full training: cosine schedule, early stopping on val QRS F1."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = config.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * config.warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_qrs = -1.0
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, class_weights, device, use_focal=use_focal,
        )
        for _ in range(len(train_loader)):
            scheduler.step()
        val_metrics = run_eval(model, val_loader, device)
        qrs_f1 = val_metrics[ecg_eval.SUPER_QRS]["f1"]
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_F1: P={val_metrics[ecg_eval.SUPER_P]['f1']:.3f} "
            f"QRS={qrs_f1:.3f} T={val_metrics[ecg_eval.SUPER_T]['f1']:.3f}"
        )
        if qrs_f1 > best_qrs:
            best_qrs = qrs_f1
            best_metrics = {
                "epoch": epoch,
                "val_qrs_f1": qrs_f1,
                "val_p_f1": val_metrics[ecg_eval.SUPER_P]["f1"],
                "val_t_f1": val_metrics[ecg_eval.SUPER_T]["f1"],
                "val_other_f1": val_metrics[ecg_eval.SUPER_OTHER]["f1"],
            }
            epochs_without_improvement = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, best_metrics, config)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break

    return best_metrics or {}
