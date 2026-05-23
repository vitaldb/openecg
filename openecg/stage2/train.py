"""Stage 2 training loop with checkpointing and early stopping."""

from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from openecg import eval as ecg_eval
from openecg.stage2 import afib_mask as _afm


def _flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


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
    """Run val pass; return per-class F1 (using openecg.eval.frame_f1)."""
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
        device="cuda", ckpt_path=None, log_fn=_flush_print, use_focal=True):
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
           device="cuda", ckpt_path=None, log_fn=_flush_print):
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
                         ignore_index=255, lambda_reg=0.1,
                         use_afib_mask=False):
    """Per-epoch training for a (cls, reg) tuple model on (sig, lead, labels,
    reg_targets, reg_mask) batches.

    ``use_afib_mask``: when True the loader is expected to yield 6-tuples
    ``(sig, lead, labels, reg_t, reg_m, afib_flag[B])``. AFib rows have:
      - main logits P-folded into OTHER (model P output absorbed silently)
      - main labels P -> OTHER
      - reg_mask p_on/p_off cleared (no L1 penalty on P boundaries)
    """
    model.train()
    weights = class_weights.to(device)
    total = 0.0
    n = 0
    for batch in loader:
        if use_afib_mask:
            sigs, leads, labels, reg_t, reg_m, afib_flag = batch
            afib_flag = afib_flag.to(device).bool()
        else:
            sigs, leads, labels, reg_t, reg_m = batch
            afib_flag = None
        sigs = sigs.to(device)
        leads = leads.to(device)
        labels = labels.to(device)
        reg_t = reg_t.to(device).float()
        reg_m = reg_m.to(device).bool()
        cls_logits, reg_off = model(sigs, leads)
        if afib_flag is not None:
            cls_logits = _afm.pfold_logits(cls_logits, afib_flag)
            labels = _afm.pfold_labels(labels, afib_flag)
            reg_m = _afm.pfold_reg_mask(reg_m, afib_flag)
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
def run_eval_reg(model, loader, device, use_afib_mask=False):
    """Same as run_eval but unwraps the (cls, reg, [aux]) tuple model output.

    Accepts both the 2-tuple (cls, reg) returned by FrameClassifierViTReg and
    the 3-tuple (cls, reg, aux) returned by FrameClassifierViTRegAux.

    ``use_afib_mask``: when True the loader yields 4-tuples
    ``(sig, lead, labels, afib_flag[B])``. AFib frames are P-folded
    (pred/true both: P -> OTHER) before frame F1 — so any P prediction
    or P label inside an AFib window contributes 0 to TP/FP/FN.
    """
    model.train(False)
    all_pred = []
    all_true = []
    all_afib = []
    for batch in loader:
        if use_afib_mask:
            sigs, leads, labels, afib_flag = batch
            afib_flag_np = afib_flag.bool().numpy()
        else:
            sigs, leads, labels = batch
            afib_flag_np = None
        sigs = sigs.to(device)
        leads = leads.to(device)
        out = model(sigs, leads)
        cls_logits = out[0]
        pred = cls_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        true = labels.numpy().astype(np.uint8)
        all_pred.append(pred.reshape(-1))
        all_true.append(true.reshape(-1))
        if afib_flag_np is not None:
            T = pred.shape[1] if pred.ndim == 2 else pred.size // afib_flag_np.size
            all_afib.append(np.repeat(afib_flag_np, T))
    pred_concat = np.concatenate(all_pred)
    true_concat = np.concatenate(all_true)
    if use_afib_mask and all_afib:
        afib_concat = np.concatenate(all_afib)
        pred_concat, true_concat = _afm.pfold_predictions_arrays(
            pred_concat, true_concat, afib_concat,
        )
    return ecg_eval.frame_f1(pred_concat, true_concat)


def _aux_targets_from_main(labels, aux_target, ignore_index=255):
    """Map main-task labels to aux-target labels.

    For binary aux heads (qrs_binary / p_binary), collapse the multi-class
    label into "is target" / "is not target". IGNORE_INDEX positions stay
    ignored. For v18+, qrs_binary covers SUPER_QRS *and* SUPER_PACED_QRS
    so the lower stack learns "any QRS event" rather than "non-paced QRS
    only" — the paced-vs-sinus distinction is the upper stack's job.

    Returns a long tensor with the same shape as `labels`.
    """
    if aux_target == "all":
        return labels
    from openecg import eval as _ee
    if aux_target == "qrs_binary":
        is_target = (
            (labels == _ee.SUPER_QRS) | (labels == _ee.SUPER_PACED_QRS)
        ).long()
    elif aux_target == "p_binary":
        is_target = (labels == _ee.SUPER_P).long()
    else:
        raise ValueError(f"unknown aux_target: {aux_target!r}")
    return torch.where(labels == ignore_index,
                        torch.full_like(is_target, ignore_index),
                        is_target)


def _mask_p_aux_for_afib(p_labels, afib_flag, ignore_index):
    """Replace every frame of P-binary aux labels with ``ignore_index`` on
    AFib rows. After this, CE on the P aux head contributes 0 loss (and
    0 gradient) for any model output, P or not, within AFib windows —
    matching the strict don't-care semantics of the main P-fold.
    """
    if afib_flag is None or not afib_flag.any():
        return p_labels
    flag = afib_flag.view(-1, 1).expand_as(p_labels)
    return torch.where(
        flag, torch.full_like(p_labels, ignore_index), p_labels,
    )


def train_one_epoch_reg_aux(model, loader, optimizer, class_weights, device,
                              scheduler=None, grad_clip=1.0,
                              ignore_index=255, lambda_reg=0.1, alpha_aux=0.3,
                              use_afib_mask=False, scaler=None,
                              pfold_logits_enable=True,
                              t_soft_alpha=1.0, t_soft_radius=1):
    """Per-epoch training loop for an auxiliary-head model returning
    (cls_logits, reg_offsets, aux_logits).

    Loss = main_cls_loss + alpha_aux * aux_cls_loss + lambda_reg * reg_loss.

    The aux loss target depends on `model.aux_target`:
      * "all"        → 4-class CE with the same labels & weights as the main
                       head (v13 / v15 behaviour: deep supervision).
      * "qrs_binary" → 2-class CE on (label==SUPER_QRS); no class weight.
      * "p_binary"   → 2-class CE on (label==SUPER_P);  no class weight.
      * "dual_binary"→ summed QRS-binary + P-binary CE on the (aux_qrs, aux_p)
                       tuple emitted by ``FrameClassifierMambaDualAux``.

    Binary aux uses no class weight: alpha_aux already gates total
    contribution, and the natural class imbalance lets the aux head fire
    only when confident — exactly what the upper-stack concat path needs.

    ``use_afib_mask``: when True the loader yields 6-tuples
    ``(sig, lead, labels, reg_t, reg_m, afib_flag[B])``. AFib rows trigger:
      - main logits/labels P-fold (P -> OTHER, P channel suppressed)
      - reg_mask p_on/p_off cleared
      - P-binary aux labels (single or dual) set to ignore_index on the
        whole AFib row, so the P aux head's output is loss-irrelevant.
        QRS-binary aux is untouched (its target is independent of P).
      - 4-class aux ("all") is folded the same way as the main head.
    """
    model.train()
    weights = class_weights.to(device)
    aux_target = getattr(model, "aux_target", "all")
    aux_weights = weights if aux_target == "all" else None
    total = 0.0
    n = 0
    for batch in loader:
        if use_afib_mask:
            sigs, leads, labels, reg_t, reg_m, afib_flag = batch
            afib_flag = afib_flag.to(device).bool()
        else:
            sigs, leads, labels, reg_t, reg_m = batch
            afib_flag = None
        sigs = sigs.to(device)
        leads = leads.to(device)
        labels = labels.to(device)
        reg_t = reg_t.to(device).float()
        reg_m = reg_m.to(device).bool()
        amp_ctx = (torch.cuda.amp.autocast(dtype=torch.float16)
                    if scaler is not None else _nullcontext())
        with amp_ctx:
            out = model(sigs, leads)
            cls_logits, reg_off = out[0], out[1]
            if afib_flag is not None:
                # v54g: pfold_logits_enable=False keeps the strict N/A
                # treatment via pfold_labels (P frames → IGNORE so CE skips
                # them) but drops the P-logit clamp. The latter forced
                # softmax to renormalize over BG/QRS/T even for non-P
                # frames in AFib rows, which biased P feature learning when
                # is_afib false-positives on a sinus record.
                if pfold_logits_enable:
                    cls_logits = _afm.pfold_logits(cls_logits, afib_flag)
                labels = _afm.pfold_labels(labels, afib_flag)
                reg_m = _afm.pfold_reg_mask(reg_m, afib_flag)
            if t_soft_alpha < 1.0 - 1e-6:
                # v54i: build per-batch soft target that softens T_on/T_off
                # transitions only. Other classes stay one-hot. IGNORE rows
                # stay all-zero so kl_cross_entropy masks them.
                from openecg.stage2.soft_labels import (
                    t_boundary_soft_target_batched,
                )
                soft_t = t_boundary_soft_target_batched(
                    labels, n_classes=cls_logits.shape[-1],
                    alpha=float(t_soft_alpha), radius=int(t_soft_radius),
                    ignore_index=ignore_index,
                )
                cls_loss = kl_cross_entropy(cls_logits, soft_t, weight=weights)
            else:
                cls_loss = nn.functional.cross_entropy(
                    cls_logits.transpose(1, 2), labels, weight=weights,
                    ignore_index=ignore_index,
                )
            if aux_target == "none":
                # v52: NoAux model — skip aux loss entirely.
                aux_loss = torch.zeros((), device=device, dtype=cls_loss.dtype)
            elif aux_target == "dual_binary":
                aux_qrs_logits, aux_p_logits = out[2], out[3]
                qrs_labels = _aux_targets_from_main(labels, "qrs_binary", ignore_index)
                p_labels   = _aux_targets_from_main(labels, "p_binary",   ignore_index)
                p_labels = _mask_p_aux_for_afib(p_labels, afib_flag, ignore_index)
                aux_loss = (
                    nn.functional.cross_entropy(
                        aux_qrs_logits.transpose(1, 2), qrs_labels,
                        ignore_index=ignore_index,
                    )
                    + nn.functional.cross_entropy(
                        aux_p_logits.transpose(1, 2), p_labels,
                        ignore_index=ignore_index,
                    )
                )
            else:
                aux_logits = out[2]
                if (aux_target == "all" and afib_flag is not None
                        and pfold_logits_enable):
                    aux_logits = _afm.pfold_logits(aux_logits, afib_flag)
                aux_labels = _aux_targets_from_main(labels, aux_target, ignore_index)
                if aux_target == "p_binary":
                    aux_labels = _mask_p_aux_for_afib(aux_labels, afib_flag, ignore_index)
                aux_loss = nn.functional.cross_entropy(
                    aux_logits.transpose(1, 2), aux_labels, weight=aux_weights,
                    ignore_index=ignore_index,
                )
            # v50d: reg_head removed → reg_off is None. Skip reg L1 term.
            if reg_off is not None:
                reg_loss = boundary_l1_loss(reg_off, reg_t, reg_m)
                loss = cls_loss + alpha_aux * aux_loss + lambda_reg * reg_loss
            else:
                loss = cls_loss + alpha_aux * aux_loss
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def fit_reg_aux(model, train_loader, val_loader, class_weights, config,
                device="cuda", ckpt_path=None, log_fn=_flush_print,
                lambda_reg=0.1, alpha_aux=0.3, use_afib_mask=False,
                use_amp=False, compile_model=False,
                eval_fn=None, eval_every=1,
                real_only_loader=None, real_only_last_k=0,
                pfold_logits_enable=True,
                t_soft_alpha=1.0, t_soft_radius=1):
    """fit() variant for FrameClassifierViTRegAux: adds an auxiliary CE loss
    on the intermediate-layer aux head with weight `alpha_aux`.

    ``use_afib_mask``: propagates to train_one_epoch_reg_aux and run_eval_reg.
    Train + val loaders must yield AFib-flag-tailed batches when True.

    ``use_amp``: enable torch.cuda.amp mixed-precision training (fp16
    forward + GradScaler). ~1.5-2x speed-up on RTX 4090.
    ``compile_model``: wrap the model in ``torch.compile`` before training
    (PyTorch 2.x graph fusion). First epoch pays compile overhead but
    subsequent epochs gain 1.5-2x throughput. Disable if hitting Triton
    issues on Windows.

    ``val_loader`` may be a single ``DataLoader`` (legacy: one validation
    set, score = ``score_val_metrics(metrics, early_stop_metric)``) OR a
    ``dict[str, DataLoader]`` (multi-dataset val: each loader is scored
    separately, the early-stop score is the *unweighted mean* of the
    per-set scores, and per-set scores are logged each epoch). Use the
    dict form when the model is selected against the same multi-dataset
    metric it will be reported on (e.g. v45g's LUDB+ISP+QTDB mean) so
    selection isn't myopic to one set.

    ``eval_fn(model) -> dict[str, float]``: when provided, OVERRIDES the
    val_loader-based score. Called every ``eval_every`` epochs (skipping
    the cheap frame-F1 eval on the intervening epochs). The returned
    dict's per-set values are logged and their mean becomes the
    early-stop score. Use this to drive selection by the exact final
    eval metric (e.g. ``score_all_1ch`` boundary-F1) instead of the
    cheap frame-F1 proxy. Trade-off: each call costs ~30 s on a 4090,
    so eval_every=3 adds ~10 min per 80-ep run.

    ``real_only_loader`` + ``real_only_last_k``: v45m1 last-K curriculum.
    When both supplied (loader != None and K > 0), the final K epochs of
    training swap ``train_loader`` for ``real_only_loader`` (which must
    yield batches with the SAME tuple shape — caller is responsible for
    wrapping LUDB+ISP+QTDB real-train with the same RegLabel/AFibMask/
    rank-norm wrappers as ``train_loader``). Designed to recover the
    early-epoch QTDB peak that drifts away during synth-dominated late
    epochs. K=0 (default) keeps the legacy single-loader behavior.
    """
    model = model.to(device)
    if compile_model:
        model = torch.compile(model)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None
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
    multi_val = isinstance(val_loader, dict)
    use_eval_fn = eval_fn is not None
    real_only_active = real_only_loader is not None and real_only_last_k > 0
    real_only_start_epoch = (
        config.epochs - int(real_only_last_k) if real_only_active else None
    )
    for epoch in range(config.epochs):
        # v45m1 curriculum: swap to real-only loader for final K epochs.
        if real_only_active and epoch >= real_only_start_epoch:
            epoch_loader = real_only_loader
            if epoch == real_only_start_epoch:
                log_fn(
                    f"[v45m1] entering real-only phase at epoch {epoch} "
                    f"(remaining {real_only_last_k} epochs use "
                    f"LUDB+ISP+QTDB real-train only)"
                )
        else:
            epoch_loader = train_loader
        train_loss = train_one_epoch_reg_aux(
            model, epoch_loader, optimizer, class_weights, device,
            scheduler=scheduler, grad_clip=config.grad_clip,
            lambda_reg=lambda_reg, alpha_aux=alpha_aux,
            use_afib_mask=use_afib_mask, scaler=scaler,
            pfold_logits_enable=pfold_logits_enable,
            t_soft_alpha=t_soft_alpha, t_soft_radius=t_soft_radius,
        )
        if use_eval_fn:
            last_epoch = (epoch == config.epochs - 1)
            if (epoch % eval_every != 0) and not last_epoch:
                log_fn(
                    f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                    f"(eval skipped — runs every {eval_every} epochs)"
                )
                continue
            model.train(False)
            with torch.no_grad():
                bf1 = eval_fn(model)
            model.train(True)
            score = float(np.mean(list(bf1.values())))
            val_metrics = {"boundary_f1": dict(bf1)}
            per_set_scores = dict(bf1)
            set_str = "  ".join(
                f"{n}={s:.3f}" for n, s in bf1.items()
            )
            log_fn(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"bf1_mean={score:.3f}  [{set_str}]  "
                f"lambda={lambda_reg}  alpha_aux={alpha_aux}"
            )
        elif multi_val:
            val_metrics = {}
            per_set_scores = {}
            for name, loader in val_loader.items():
                m = run_eval_reg(
                    model, loader, device, use_afib_mask=use_afib_mask,
                )
                val_metrics[name] = m
                per_set_scores[name] = score_val_metrics(
                    m, config.early_stop_metric,
                )
            score = float(np.mean(list(per_set_scores.values())))
            set_str = "  ".join(
                f"{n}={s:.3f}" for n, s in per_set_scores.items()
            )
            log_fn(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"mean={score:.3f}  [{set_str}]  "
                f"lambda={lambda_reg}  alpha_aux={alpha_aux}"
            )
        else:
            val_metrics = run_eval_reg(
                model, val_loader, device, use_afib_mask=use_afib_mask,
            )
            score = score_val_metrics(val_metrics, config.early_stop_metric)
            per_set_scores = None
            log_fn(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"score={score:.3f}  lambda={lambda_reg}  alpha_aux={alpha_aux}"
            )
        if score > best_score:
            best_score = score
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "score": score,
                "metrics": val_metrics,
                "early_stop_metric": config.early_stop_metric,
            }
            if per_set_scores is not None:
                best_metrics["per_set_scores"] = per_set_scores
            bad = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, best_metrics, config,
                                extra={"alpha_aux": alpha_aux,
                                       "lambda_reg": lambda_reg})
        else:
            bad += 1
            if bad >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break
    return best_metrics


def _masked_window_ce(logits, labels, mask):
    """Cross-entropy averaged over the valid samples in a batch.

    logits: [B, C], labels: [B] long, mask: [B] bool. Samples where
    mask=False are excluded from both numerator and denominator. If no
    sample is valid, returns a zero loss with a graph for safe backprop.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], labels[mask])


def train_one_epoch_multitask(model, loader, optimizer, class_weights, device,
                                scheduler=None, grad_clip=1.0,
                                ignore_index=255, lambda_reg=0.1, alpha_aux=0.3,
                                alpha_rr=0.3, alpha_wide=0.3,
                                alpha_rhythm=0.0, alpha_avb=0.0,
                                log_components=False):
    """v17 training loop for `FrameClassifierViTRegMultiTask`.

    Loader items are 7-tuples (sig, lead, frame_labels, reg_t, reg_m,
    window_labels[K], window_mask[K]).

    Total loss
      = main_cls
      + alpha_aux    * aux_cls            (binary or 4-class per model.aux_target)
      + lambda_reg   * reg_l1
      + alpha_rr     * rr_regular_ce      (window-level, masked)
      + alpha_wide   * qrs_wide_ce        (window-level, masked)
      [+ alpha_rhythm * rhythm_ce          (window-level, masked, if model.head_rhythm is not None)]
      [+ alpha_avb    * avb_ce             (window-level, masked, if model.head_avb is not None)]
    """
    model.train()
    weights = class_weights.to(device)
    aux_target = getattr(model, "aux_target", "all")
    aux_weights = weights if aux_target == "all" else None
    has_rhythm = getattr(model, "head_rhythm", None) is not None
    has_avb = getattr(model, "head_avb", None) is not None
    total = 0.0
    n = 0
    component_acc = {k: 0.0 for k in
                      ("cls", "aux", "reg", "rr", "wide", "rhythm", "avb")}

    for batch in loader:
        sigs, leads, labels, reg_t, reg_m, win_l, win_m = batch
        sigs = sigs.to(device)
        leads = leads.to(device)
        labels = labels.to(device)
        reg_t = reg_t.to(device).float()
        reg_m = reg_m.to(device).bool()
        win_l = win_l.to(device).long()
        win_m = win_m.to(device).bool()

        cls_logits, reg_off, aux_logits, win_logits = model(sigs, leads)

        cls_loss = nn.functional.cross_entropy(
            cls_logits.transpose(1, 2), labels, weight=weights,
            ignore_index=ignore_index,
        )
        aux_labels = _aux_targets_from_main(labels, aux_target, ignore_index)
        aux_loss = nn.functional.cross_entropy(
            aux_logits.transpose(1, 2), aux_labels, weight=aux_weights,
            ignore_index=ignore_index,
        )
        reg_loss = boundary_l1_loss(reg_off, reg_t, reg_m)

        rr_loss = _masked_window_ce(
            win_logits["rr_regular"], win_l[:, 0], win_m[:, 0],
        )
        wide_loss = _masked_window_ce(
            win_logits["qrs_wide"], win_l[:, 1], win_m[:, 1],
        )
        loss = (cls_loss + alpha_aux * aux_loss + lambda_reg * reg_loss
                + alpha_rr * rr_loss + alpha_wide * wide_loss)

        rhythm_loss_v = 0.0
        avb_loss_v = 0.0
        if has_rhythm and "rhythm" in win_logits and win_l.shape[1] > 2:
            rhythm_loss = _masked_window_ce(
                win_logits["rhythm"], win_l[:, 2], win_m[:, 2],
            )
            loss = loss + alpha_rhythm * rhythm_loss
            rhythm_loss_v = float(rhythm_loss.item())
        if has_avb and "avb" in win_logits and win_l.shape[1] > 3:
            avb_loss = _masked_window_ce(
                win_logits["avb"], win_l[:, 3], win_m[:, 3],
            )
            loss = loss + alpha_avb * avb_loss
            avb_loss_v = float(avb_loss.item())

        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
        n += 1
        if log_components:
            component_acc["cls"] += float(cls_loss.item())
            component_acc["aux"] += float(aux_loss.item())
            component_acc["reg"] += float(reg_loss.item())
            component_acc["rr"] += float(rr_loss.item())
            component_acc["wide"] += float(wide_loss.item())
            component_acc["rhythm"] += rhythm_loss_v
            component_acc["avb"] += avb_loss_v

    avg = total / max(1, n)
    if log_components:
        return avg, {k: v / max(1, n) for k, v in component_acc.items()}
    return avg


@torch.no_grad()
def run_eval_multitask(model, loader, device):
    """Validation eval for the multi-task model.

    The val loader is a plain frame-only LUDB val set (3-tuple). We unwrap
    only the first (cls_logits) tensor of the 4-tuple model output and
    compute frame F1 — same protocol as the v15 / v16 sweeps so the
    headline number stays comparable.
    """
    model.train(False)
    all_pred = []
    all_true = []
    for batch in loader:
        if len(batch) == 3:
            sigs, leads, labels = batch
        else:
            sigs, leads, labels = batch[0], batch[1], batch[2]
        sigs = sigs.to(device)
        leads = leads.to(device)
        out = model(sigs, leads)
        cls_logits = out[0]
        pred = cls_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        true = labels.numpy().astype(np.uint8)
        all_pred.append(pred.reshape(-1))
        all_true.append(true.reshape(-1))
    return ecg_eval.frame_f1(np.concatenate(all_pred), np.concatenate(all_true))


def fit_multitask(model, train_loader, val_loader, class_weights, config,
                   device="cuda", ckpt_path=None, log_fn=_flush_print,
                   lambda_reg=0.1, alpha_aux=0.3,
                   alpha_rr=0.3, alpha_wide=0.3,
                   alpha_rhythm=0.0, alpha_avb=0.0):
    """fit() variant for the v17 multi-task model.

    Early stopping uses the same `mean_wave_f1` as v15 / v16 so headline
    LUDB val numbers stay comparable across versions.
    """
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
        train_loss, comp = train_one_epoch_multitask(
            model, train_loader, optimizer, class_weights, device,
            scheduler=scheduler, grad_clip=config.grad_clip,
            lambda_reg=lambda_reg, alpha_aux=alpha_aux,
            alpha_rr=alpha_rr, alpha_wide=alpha_wide,
            alpha_rhythm=alpha_rhythm, alpha_avb=alpha_avb,
            log_components=True,
        )
        val_metrics = run_eval_multitask(model, val_loader, device)
        score = score_val_metrics(val_metrics, config.early_stop_metric)
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"score={score:.3f}  | "
            f"cls={comp['cls']:.3f}  aux={comp['aux']:.3f}  "
            f"reg={comp['reg']:.3f}  rr={comp['rr']:.3f}  wide={comp['wide']:.3f}"
        )
        if score > best_score:
            best_score = score
            best_metrics = {
                "epoch": epoch, "train_loss": train_loss,
                "score": score, "metrics": val_metrics,
                "components": comp,
                "early_stop_metric": config.early_stop_metric,
            }
            bad = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, best_metrics, config,
                                extra={"alpha_aux": alpha_aux,
                                       "alpha_rr": alpha_rr,
                                       "alpha_wide": alpha_wide,
                                       "lambda_reg": lambda_reg})
        else:
            bad += 1
            if bad >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break
    return best_metrics


def fit_reg(model, train_loader, val_loader, class_weights, config,
            device="cuda", ckpt_path=None, log_fn=_flush_print, lambda_reg=0.1,
            use_afib_mask=False, eval_fn=None, eval_every=1,
            real_only_loader=None, real_only_last_k=0):
    """fit() variant for FrameClassifierViTReg-style models.

    ``use_afib_mask``: propagates to train_one_epoch_reg and run_eval_reg.
    Loaders must yield AFib-flag-tailed batches when True.

    ``eval_fn(model) -> dict[str, float]``: optional boundary-F1 multi-val
    evaluator (e.g. ``score_all_1ch``). When provided, OVERRIDES the
    val_loader-based score. Called every ``eval_every`` epochs.

    ``real_only_loader`` + ``real_only_last_k``: v45m1 last-K curriculum
    (see fit_reg_aux for full doc). K=0 (default) keeps legacy behavior.
    """
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
    multi_val = isinstance(val_loader, dict)
    use_eval_fn = eval_fn is not None
    real_only_active = real_only_loader is not None and real_only_last_k > 0
    real_only_start_epoch = (
        config.epochs - int(real_only_last_k) if real_only_active else None
    )
    for epoch in range(config.epochs):
        if real_only_active and epoch >= real_only_start_epoch:
            epoch_loader = real_only_loader
            if epoch == real_only_start_epoch:
                log_fn(
                    f"[v45m1] entering real-only phase at epoch {epoch} "
                    f"(remaining {real_only_last_k} epochs use "
                    f"LUDB+ISP+QTDB real-train only)"
                )
        else:
            epoch_loader = train_loader
        train_loss = train_one_epoch_reg(
            model, epoch_loader, optimizer, class_weights, device,
            scheduler=scheduler, grad_clip=config.grad_clip,
            lambda_reg=lambda_reg, use_afib_mask=use_afib_mask,
        )
        if use_eval_fn:
            last_epoch = (epoch == config.epochs - 1)
            if (epoch % eval_every != 0) and not last_epoch:
                log_fn(
                    f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                    f"(eval skipped — runs every {eval_every} epochs)"
                )
                continue
            model.train(False)
            with torch.no_grad():
                bf1 = eval_fn(model)
            model.train(True)
            score = float(np.mean(list(bf1.values())))
            val_metrics = {"boundary_f1": dict(bf1)}
            per_set_scores = dict(bf1)
            set_str = "  ".join(f"{n}={s:.3f}" for n, s in bf1.items())
            log_fn(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"bf1_mean={score:.3f}  [{set_str}]  lambda={lambda_reg}"
            )
            rep = None
        elif multi_val:
            val_metrics = {}
            per_set_scores = {}
            for name, loader in val_loader.items():
                m = run_eval_reg(
                    model, loader, device, use_afib_mask=use_afib_mask,
                )
                val_metrics[name] = m
                per_set_scores[name] = score_val_metrics(
                    m, config.early_stop_metric,
                )
            score = float(np.mean(list(per_set_scores.values())))
            set_str = "  ".join(
                f"{n}={s:.3f}" for n, s in per_set_scores.items()
            )
            log_fn(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"mean={score:.3f}  [{set_str}]  lambda={lambda_reg}"
            )
            # Pick representative per-class F1s from the LUDB head so the
            # legacy best_metrics fields stay populated.
            rep = val_metrics.get("ludb", next(iter(val_metrics.values())))
        else:
            val_metrics = run_eval_reg(
                model, val_loader, device, use_afib_mask=use_afib_mask,
            )
            score = score_val_metrics(val_metrics, config.early_stop_metric)
            per_set_scores = None
            rep = val_metrics
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
            }
            if rep is not None:
                best_metrics.update({
                    "val_qrs_f1": rep[ecg_eval.SUPER_QRS]["f1"],
                    "val_p_f1": rep[ecg_eval.SUPER_P]["f1"],
                    "val_t_f1": rep[ecg_eval.SUPER_T]["f1"],
                })
            if per_set_scores is not None:
                best_metrics["per_set_scores"] = per_set_scores
            if use_eval_fn:
                best_metrics["metrics"] = val_metrics
            bad = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, best_metrics, config)
        else:
            bad += 1
            if bad >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break
    return best_metrics or {}
