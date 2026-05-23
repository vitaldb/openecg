# openecg/stage2/cascade.py
"""Two-pass cascade: a frozen frame model (pass 1) feeds its per-frame
class probabilities and boundary regression into a small refiner
(pass 2) that re-decides the labels.

Pass 1 emits per-frame predictions independently; pass 2 sees the full
sequence plus the raw signal and learns to second-guess based on global
structure (e.g. suppressing P bands that fire 200 ms before every paced
QRS as the BUT PDB rid=3 paced FP signature).

Pieces:
* CascadeDataset   — wraps a base dataset, precomputes pass-1 outputs.
* FrameRefinerViT  — small ViT consuming (signal + pass1 features).
* CascadedModel    — wrapper exposing the legacy (cls, reg) API.
* fit_refiner / train_one_epoch_refiner / run_eval_refiner.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from openecg import eval as ecg_metrics
from openecg.stage2 import train as _train

WINDOW_FRAMES = 500
N_REG = 6


class CascadeDataset(Dataset):
    """Wrap a frame-labelled dataset; pre-compute pass-1 outputs once and
    emit them as trailing tensors per item.

    Memory budget: N items × 500 frames × (K_p1 + n_reg) channels × float32.
    For the v17/v18 train mix (~20 K items × 5+6 channels) that is ~440 MB.
    """

    def __init__(self, base, pass1_model, *, device: str = "cuda",
                 batch_size: int = 64, verbose: bool = True):
        self.base = base
        N = len(base)
        K = int(pass1_model.model_config.get("n_classes", 5))
        self._cls = np.zeros((N, WINDOW_FRAMES, K), dtype=np.float32)
        self._reg = np.zeros((N, WINDOW_FRAMES, N_REG), dtype=np.float32)
        pass1_model = pass1_model.to(device).train(False)
        if verbose:
            print(f"[CascadeDataset] precomputing pass-1 over {N:,} items "
                  f"(K={K}, batch={batch_size})...", flush=True)
        with torch.no_grad():
            i = 0
            while i < N:
                hi = min(N, i + batch_size)
                batch = [base[j] for j in range(i, hi)]
                sigs = torch.stack([t[0] for t in batch]).to(device)
                leads = torch.stack([t[1] for t in batch]).to(device)
                out = pass1_model(sigs, leads)
                cls_logits = out[0]
                reg = out[1]
                cls_softmax = torch.softmax(cls_logits, dim=-1)
                self._cls[i:hi] = cls_softmax.cpu().numpy()
                self._reg[i:hi] = reg.cpu().numpy()
                i = hi
                if verbose and (i % (batch_size * 50) == 0 or i == N):
                    print(f"  pass-1 {i:>6,}/{N:,}", flush=True)
        self.K = K

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        cls_p1 = torch.from_numpy(self._cls[idx])
        reg_p1 = torch.from_numpy(self._reg[idx])
        return (*item, cls_p1, reg_p1)

    def label_counts(self):
        if hasattr(self.base, "label_counts"):
            return self.base.label_counts()
        return None


class FrameRefinerViT(nn.Module):
    """Small ViT-style frame refiner.

    Each per-frame token = (signal patch[patch_size] + pass1 softmax[K_p1]
    + pass1 reg[n_reg]). Refiner does NOT predict its own boundary
    regression; CascadedModel reuses pass-1 reg downstream.
    """

    def __init__(self, *, n_pass1_classes: int, n_pass1_reg: int = 6,
                 n_classes_out: int = 5,
                 patch_size: int = 5, n_leads: int = 12,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                 ff: int = 256, dropout: float = 0.1, max_seq_len: int = 512,
                 use_lead_emb: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.use_lead_emb = use_lead_emb
        self.n_pass1_classes = int(n_pass1_classes)
        self.n_pass1_reg = int(n_pass1_reg)
        self.n_classes_out = int(n_classes_out)
        token_dim = patch_size + self.n_pass1_classes + self.n_pass1_reg
        self.token_proj = nn.Linear(token_dim, d_model)
        if use_lead_emb:
            self.lead_emb = nn.Embedding(n_leads, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, self.n_classes_out)
        self.model_config = {
            "n_pass1_classes": self.n_pass1_classes,
            "n_pass1_reg": self.n_pass1_reg,
            "n_classes_out": self.n_classes_out,
            "patch_size": patch_size, "n_leads": n_leads,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "ff": ff, "dropout": dropout, "max_seq_len": max_seq_len,
            "use_lead_emb": use_lead_emb,
            "arch": "frame_refiner_vit",
        }

    def forward(self, x, lead_id, p1_cls, p1_reg):
        B, N = x.shape
        n_patches = N // self.patch_size
        sig_patches = x.view(B, n_patches, self.patch_size)
        token_in = torch.cat([sig_patches, p1_cls, p1_reg], dim=-1)
        h = self.token_proj(token_in)
        h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)


class AsymmetricFrameRefiner(nn.Module):
    """Asymmetric refiner: can ONLY suppress pass-1's target class (P by
    default). Fixes the v19 failure mode where the full 5-class refiner
    over-generalised "P near wide QRS = FP" and started removing real
    blocked P in BUT PDB rid=3 (recall dropped to 0.185).

    Architecturally a thin wrapper around FrameRefinerViT(n_classes_out=1).
    The single-channel output is interpreted as a "keep" gate logit g.
    Forward emits modified pass-1 log-probs where:

        modified[:, :, target] = pass1_log_prob[:, :, target] - softplus(-g)
        modified[:, :, other]  = pass1_log_prob[:, :, other]

    softplus(-g) is non-negative, so the target channel can only DECREASE
    relative to pass-1. The refiner can mute existing firings but cannot
    create new ones. Trained as a drop-in replacement for FrameRefinerViT
    in fit_refiner; the loss is unchanged (CE on modified logits).
    """

    def __init__(self, *, target_class: int = 1, **backbone_kwargs):
        super().__init__()
        backbone_kwargs["n_classes_out"] = 1
        self.backbone = FrameRefinerViT(**backbone_kwargs)
        self.target_class = int(target_class)
        self.n_classes_out = int(backbone_kwargs.get("n_pass1_classes", 5))
        self.model_config = dict(self.backbone.model_config)
        self.model_config["arch"] = "asymmetric_frame_refiner"
        self.model_config["target_class"] = self.target_class
        self.model_config["n_classes_out"] = self.n_classes_out

    def forward(self, x, lead_id, p1_cls, p1_reg):
        gate_logit = self.backbone(x, lead_id, p1_cls, p1_reg).squeeze(-1)
        suppression = nn.functional.softplus(-gate_logit)
        # Recover logits from the precomputed softmax up to a per-frame
        # additive constant — fine for CE / argmax which are
        # constant-invariant.
        p1_logits = torch.log(p1_cls.clamp(min=1e-8))
        modified = p1_logits.clone()
        modified[:, :, self.target_class] = (
            p1_logits[:, :, self.target_class] - suppression
        )
        return modified


class CascadedModel(nn.Module):
    """Wrap (pass1, refiner) so the public forward signature matches the
    legacy (cls_logits, reg_offsets) tuple. Pass-1 runs in no-grad inside
    forward; pass-2 produces the refined cls; reg is forwarded from pass-1.
    """

    def __init__(self, pass1, refiner):
        super().__init__()
        self.pass1 = pass1
        self.refiner = refiner
        self.model_config = {
            "arch": "cascaded",
            "pass1_arch": pass1.model_config.get("arch"),
            "refiner_arch": refiner.model_config.get("arch"),
            "n_classes": refiner.n_classes_out,
            "n_reg": N_REG,
        }

    def forward(self, x, lead_id):
        with torch.no_grad():
            out_p1 = self.pass1(x, lead_id)
            cls_logits_p1 = out_p1[0]
            reg_p1 = out_p1[1]
            p1_cls_softmax = torch.softmax(cls_logits_p1, dim=-1)
        cls_logits_refined = self.refiner(x, lead_id, p1_cls_softmax, reg_p1)
        return cls_logits_refined, reg_p1


def train_one_epoch_refiner(model, loader, optimizer, class_weights, device,
                              ignore_index: int = 255, scheduler=None,
                              grad_clip: float = 1.0):
    model.train()
    weights = class_weights.to(device)
    total = 0.0
    n = 0
    for batch in loader:
        sigs = batch[0].to(device)
        leads = batch[1].to(device)
        labels = batch[2].to(device)
        p1_cls = batch[-2].to(device).float()
        p1_reg = batch[-1].to(device).float()
        cls_logits = model(sigs, leads, p1_cls, p1_reg)
        loss = nn.functional.cross_entropy(
            cls_logits.transpose(1, 2), labels, weight=weights,
            ignore_index=ignore_index,
        )
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
def run_eval_refiner(model, loader, device):
    model.train(False)
    all_pred = []
    all_true = []
    for batch in loader:
        sigs = batch[0].to(device)
        leads = batch[1].to(device)
        labels = batch[2]
        p1_cls = batch[-2].to(device).float()
        p1_reg = batch[-1].to(device).float()
        cls_logits = model(sigs, leads, p1_cls, p1_reg)
        pred = cls_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        true = labels.numpy().astype(np.uint8)
        all_pred.append(pred.reshape(-1))
        all_true.append(true.reshape(-1))
    return ecg_metrics.frame_f1(np.concatenate(all_pred), np.concatenate(all_true))


def fit_refiner(refiner, train_loader, val_loader, class_weights, config,
                 device: str = "cuda", ckpt_path=None, log_fn=print):
    refiner = refiner.to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=config.lr, weight_decay=config.weight_decay,
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
        train_loss = train_one_epoch_refiner(
            refiner, train_loader, optimizer, class_weights, device,
            scheduler=scheduler, grad_clip=config.grad_clip,
        )
        val_metrics = run_eval_refiner(refiner, val_loader, device)
        score = _train.score_val_metrics(val_metrics, config.early_stop_metric)
        log_fn(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  score={score:.3f}")
        if score > best_score:
            best_score = score
            best_metrics = {
                "epoch": epoch, "train_loss": train_loss,
                "score": score, "metrics": val_metrics,
                "early_stop_metric": config.early_stop_metric,
            }
            bad = 0
            if ckpt_path is not None:
                _train.save_checkpoint(ckpt_path, refiner, best_metrics, config)
        else:
            bad += 1
            if bad >= config.early_stop_patience:
                log_fn(f"Early stop at epoch {epoch}")
                break
    return best_metrics
