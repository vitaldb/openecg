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
    early_stop_metric: str = "mean_wave_f1"
    grad_clip: float = 1.0
    seed: int = 42


def focal_cross_entropy(logits, target, weight=None, gamma=2.0, ignore_index=255):
    """Focal cross-entropy loss for class imbalance, with ignore_index support.

    logits: [B, C, ...] (already permuted so class axis is dim=1)
    target: [B, ...] integer class labels (ignore_index frames excluded from loss)
    weight: optional per-class weight tensor of shape [C]
    gamma: focusing parameter (paper recommends 2.0)
    """
    log_probs = nn.functional.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0  # safe zero gradient
    # Replace ignored frames with class 0 to avoid index errors; we'll mask out below.
    safe_target = target.clone()
    safe_target[~valid] = 0
    nll = nn.functional.nll_loss(log_probs, safe_target, weight=weight, reduction="none")
    pt = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    focal_factor = (1.0 - pt).pow(gamma)
    per_frame = focal_factor * nll
    return per_frame[valid].mean()


def train_one_epoch(model, loader, optimizer, class_weights, device,
                    use_focal=True, focal_gamma=2.0, ignore_index=255,
                    grad_clip=1.0, scheduler=None):
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
                ignore_index=ignore_index,
            )
        else:
            loss = nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, weight=weights,
                ignore_index=ignore_index,
            )
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
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


def save_checkpoint(path, model, metrics, config, model_config=None,
                    postprocess_config=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_config = model_config
    if resolved_model_config is None:
        resolved_model_config = getattr(model, "model_config", None)
    torch.save({
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": asdict(config),
        "model_config": resolved_model_config,
        "postprocess_config": postprocess_config or {},
        "extra": extra or {},
    }, path)


def load_checkpoint_blob(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path, model):
    blob = load_checkpoint_blob(path)
    model.load_state_dict(blob["model_state"])
    return blob["metrics"]


def score_val_metrics(metrics, metric_name="mean_wave_f1"):
    """Return scalar early-stop score from frame metrics."""
    if metric_name == "qrs_f1":
        return metrics[ecg_eval.SUPER_QRS]["f1"]
    if metric_name == "mean_wave_f1":
        return float(np.mean([
            metrics[ecg_eval.SUPER_P]["f1"],
            metrics[ecg_eval.SUPER_QRS]["f1"],
            metrics[ecg_eval.SUPER_T]["f1"],
        ]))
    raise ValueError(f"unknown early_stop_metric: {metric_name}")


def fit(model, train_loader, val_loader, class_weights, config,
        device="cuda", ckpt_path=None, log_fn=print, use_focal=True):
    """Full training: cosine schedule, early stopping on validation F1."""
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

    best_score = -1.0
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, class_weights, device, use_focal=use_focal,
            grad_clip=config.grad_clip, scheduler=scheduler,
        )
        val_metrics = run_eval(model, val_loader, device)
        val_score = score_val_metrics(val_metrics, config.early_stop_metric)
        qrs_f1 = val_metrics[ecg_eval.SUPER_QRS]["f1"]
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_F1: P={val_metrics[ecg_eval.SUPER_P]['f1']:.3f} "
            f"QRS={qrs_f1:.3f} T={val_metrics[ecg_eval.SUPER_T]['f1']:.3f} "
            f"score={val_score:.3f}"
        )
        if val_score > best_score:
            best_score = val_score
            best_metrics = {
                "epoch": epoch,
                "early_stop_metric": config.early_stop_metric,
                "val_score": val_score,
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


def kl_cross_entropy(logits, soft_target, weight=None):
    """Soft-target cross-entropy (-Σ target · log_softmax(logits)).

    logits:      [B, T, C] raw model output (cls_head over batch_first sequence).
    soft_target: [B, T, C] non-negative target weights. Rows whose sum is 0 are
                 masked out (no contribution to the loss).
    weight:      optional [C] tensor; per-class re-weight applied to target
                 before renormalisation so loss stays in CE-equivalent scale.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    if weight is not None:
        soft_target = soft_target * weight.view(1, 1, -1)
    row_sum = soft_target.sum(dim=-1, keepdim=True)
    valid = row_sum.squeeze(-1) > 0
    if not valid.any():
        return logits.sum() * 0.0
    target_norm = soft_target / row_sum.clamp(min=1e-8)
    per_frame = -(target_norm * log_probs).sum(dim=-1)
    return per_frame[valid].mean()


def train_one_epoch_kl(model, loader, optimizer, class_weights, device,
                       grad_clip=1.0, scheduler=None):
    """Per-epoch training loop using soft-target KL on a [B, T, C] target."""
    model.train()
    weights = class_weights.to(device)
    total = 0.0
    n = 0
    for sigs, leads, soft in loader:
        sigs = sigs.to(device)
        leads = leads.to(device)
        soft = soft.to(device).float()
        logits = model(sigs, leads)
        loss = kl_cross_entropy(logits, soft, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def fit_kl(model, train_loader, val_loader, class_weights, config,
           device="cuda", ckpt_path=None, log_fn=print):
    """fit() variant using KL on soft training targets; eval still uses hard labels."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * config.warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_score = -1.0
    best_metrics = None
    bad = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch_kl(
            model, train_loader, optimizer, class_weights, device,
            scheduler=scheduler, grad_clip=config.grad_clip,
        )
        val_metrics = run_eval(model, val_loader, device)
        score = score_val_metrics(val_metrics, config.early_stop_metric)
        log_fn(
            f"epoch {epoch:3d}  train_kl={train_loss:.4f}  "
            f"val_F1: P={val_metrics[ecg_eval.SUPER_P]['f1']:.3f} "
            f"QRS={val_metrics[ecg_eval.SUPER_QRS]['f1']:.3f} "
            f"T={val_metrics[ecg_eval.SUPER_T]['f1']:.3f} "
            f"score={score:.3f}"
        )
        if score > best_score:
            best_score = score
            best_metrics = {
                "epoch": epoch,
                "early_stop_metric": config.early_stop_metric,
                "val_score": score,
                "val_qrs_f1": val_metrics[ecg_eval.SUPER_QRS]["f1"],
                "val_p_f1": val_metrics[ecg_eval.SUPER_P]["f1"],
                "val_t_f1": val_metrics[ecg_eval.SUPER_T]["f1"],
                "val_other_f1": val_metrics[ecg_eval.SUPER_OTHER]["f1"],
            }
            bad = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, best_metrics, config)
        else:
            bad += 1
            if bad >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break
    return best_metrics or {}


def boundary_l1_loss(reg_offsets, reg_targets, reg_mask):
    """Mean masked L1 over reg head outputs.

    reg_offsets, reg_targets: [B, T, 6] float
    reg_mask: [B, T, 6] bool
    """
    if reg_mask.dtype != torch.bool:
        reg_mask = reg_mask.bool()
    if not reg_mask.any():
        return reg_offsets.sum() * 0.0
    diff = (reg_offsets - reg_targets).abs()
    return diff[reg_mask].mean()


def train_one_epoch_reg(model, loader, optimizer, class_weights, device,
                         scheduler=None, grad_clip=1.0,
                         ignore_index=255, lambda_reg=0.1):
    """Per-epoch training for a (cls, reg) tuple model on (sig, lead, labels,
    reg_targets, reg_mask) batches."""
    model.train()
    weights = class_weights.to(device)
    total = 0.0
    n = 0
    for sigs, leads, labels, reg_t, reg_m in loader:
        sigs = sigs.to(device)
        leads = leads.to(device)
        labels = labels.to(device)
        reg_t = reg_t.to(device).float()
        reg_m = reg_m.to(device).bool()
        cls_logits, reg_off = model(sigs, leads)
        cls_loss = nn.functional.cross_entropy(
            cls_logits.transpose(1, 2), labels, weight=weights,
            ignore_index=ignore_index,
        )
        reg_loss = boundary_l1_loss(reg_off, reg_t, reg_m)
        loss = cls_loss + lambda_reg * reg_loss
        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def run_eval_reg(model, loader, device):
    """Same as run_eval but unwraps the (cls, reg) tuple model output."""
    model.train(False)
    all_pred = []
    all_true = []
    for sigs, leads, labels in loader:
        sigs = sigs.to(device)
        leads = leads.to(device)
        cls_logits, _ = model(sigs, leads)
        pred = cls_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        true = labels.numpy().astype(np.uint8)
        all_pred.append(pred.reshape(-1))
        all_true.append(true.reshape(-1))
    return ecg_eval.frame_f1(np.concatenate(all_pred), np.concatenate(all_true))


def fit_reg(model, train_loader, val_loader, class_weights, config,
            device="cuda", ckpt_path=None, log_fn=print, lambda_reg=0.1):
    """fit() variant for FrameClassifierViTReg-style models."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * config.warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_score = -1.0
    best_metrics = None
    bad = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch_reg(
            model, train_loader, optimizer, class_weights, device,
            scheduler=scheduler, grad_clip=config.grad_clip,
            lambda_reg=lambda_reg,
        )
        val_metrics = run_eval_reg(model, val_loader, device)
        score = score_val_metrics(val_metrics, config.early_stop_metric)
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"score={score:.3f}  lambda={lambda_reg}"
        )
        if score > best_score:
            best_score = score
            best_metrics = {
                "epoch": epoch,
                "early_stop_metric": config.early_stop_metric,
                "val_score": score,
                "lambda_reg": lambda_reg,
                "val_qrs_f1": val_metrics[ecg_eval.SUPER_QRS]["f1"],
                "val_p_f1": val_metrics[ecg_eval.SUPER_P]["f1"],
                "val_t_f1": val_metrics[ecg_eval.SUPER_T]["f1"],
            }
            bad = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, best_metrics, config)
        else:
            bad += 1
            if bad >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break
    return best_metrics or {}
