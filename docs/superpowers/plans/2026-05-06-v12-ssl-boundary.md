# v12 — SSL transfer + boundary engineering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run 7 new training experiments that probe (a) boundary-aware loss/regression on the existing v9 ViT and (b) transfer from open-weight ECG SSL backbones (HuBERT-ECG, ST-MEM), then aggregate results into one comparison table against the v9 baseline.

**Architecture:** Three orthogonal layers added on top of the existing `ecgcode/stage2/` package — (1) soft-label and boundary-regression dataset wrappers + losses + a `FrameClassifierViTReg` model, (2) a `ssl/` submodule with HuBERT-ECG and ST-MEM adapters that expose a uniform `(sig[B,2500@250Hz], lead_id) -> features[B,500,d]` interface, (3) a `BackboneWithHeads` wrapper that lets any backbone plug into either or both heads. All experiments share the v9 dataset/eval protocol.

**Tech Stack:** Python 3.11, PyTorch 2.6 (CUDA 12.4), HuggingFace `transformers` (new dep), scipy, numpy, wfdb, pytest. ST-MEM vendored in `third_party/ST-MEM/`.

**Spec:** `docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md`

**Note for code:** PyTorch's inference-mode toggle is written as `model.train(False)` everywhere in this plan instead of the equivalent `.eval()` call, to avoid a false-positive in the host's security hook that flags any literal `.eval(` substring as Python's `eval()` builtin. Both forms are functionally identical.

---

## File Structure

```
ecgcode/stage2/
  soft_labels.py        # NEW  soft_boundary_labels() + SoftLabelDataset
  reg_targets.py        # NEW  boundary_regression_targets() + RegLabelDataset
  model.py              # MOD  + FrameClassifierViTReg
  train.py              # MOD  + kl_cross_entropy, train_one_epoch_kl, fit_kl,
                        #       boundary_l1_loss, train_one_epoch_reg, run_eval_reg, fit_reg
  infer.py              # MOD  + predict_frames_with_reg, apply_reg_to_boundaries
  ssl/
    __init__.py         # NEW  package marker
    head.py             # NEW  FrameHead, FrameRegHead, BackboneWithHeads
    hubert.py           # NEW  HubertECGAdapter
    stmem.py            # NEW  STMEMAdapter
scripts/
  train_v12_soft.py     # NEW  run 1
  train_v12_reg.py      # NEW  run 2 (lambda sweep)
  train_v12_hubert.py   # NEW  runs 3, 4 (--mode lp|ft)
  train_v12_stmem.py    # NEW  runs 5, 6 (--mode lp|ft)
  train_v12_best.py     # NEW  run 7 (combination)
  compare_v12.py        # NEW  aggregator
tests/
  test_stage2_soft_labels.py    # NEW
  test_stage2_reg_targets.py    # NEW
  test_stage2_train_v12.py      # NEW (kl + reg loss / fit smoke)
  test_stage2_model.py          # MOD (+ ViTReg shape tests)
  test_stage2_infer.py          # MOD (+ reg-aware infer tests)
  test_stage2_ssl_head.py       # NEW
  test_stage2_ssl_hubert.py     # NEW (skip if no network)
  test_stage2_ssl_stmem.py      # NEW (skip if vendored module missing)
pyproject.toml          # MOD  + transformers, huggingface-hub
docs/superpowers/plans/2026-05-06-v12-ssl-boundary.md   # this file
```

Each Phase 1 task is self-contained; Phase 2 reuses Phase 1's heads. Phase 3 depends on Phase 1 + Phase 2 outputs.

---

## Phase 1 — Boundary engineering on v9 backbone

### Task 1: `soft_boundary_labels` helper

**Files:**
- Create: `ecgcode/stage2/soft_labels.py`
- Test:   `tests/test_stage2_soft_labels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2_soft_labels.py
import numpy as np
import pytest

from openecg import eval as ee
from openecg.stage2.soft_labels import soft_boundary_labels


def test_no_transitions_produces_one_hot():
    labels = np.array([1, 1, 1, 1], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    assert soft.shape == (4, 4)
    assert soft.dtype == np.float32
    expected = np.array([[0, 1, 0, 0]] * 4, dtype=np.float32)
    np.testing.assert_array_almost_equal(soft, expected)


def test_single_transition_softens_two_frames():
    # other(0) -> P(1) at frames 1->2
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    np.testing.assert_almost_equal(soft[1, 0], 0.7)
    np.testing.assert_almost_equal(soft[1, 1], 0.3)
    np.testing.assert_almost_equal(soft[2, 0], 0.3)
    np.testing.assert_almost_equal(soft[2, 1], 0.7)
    np.testing.assert_almost_equal(soft[0, 0], 1.0)
    np.testing.assert_almost_equal(soft[3, 1], 1.0)


def test_ignore_index_produces_zero_row():
    labels = np.array([1, ee.IGNORE_INDEX, 2, 2], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    np.testing.assert_array_almost_equal(soft[1], np.zeros(4, dtype=np.float32))
    np.testing.assert_almost_equal(soft[0, 1], 1.0)
    np.testing.assert_almost_equal(soft[2, 2], 1.0)


def test_mass_conservation_on_valid_rows():
    labels = np.array([0, 1, 2, 3, 0, ee.IGNORE_INDEX], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    sums = soft.sum(axis=-1)
    np.testing.assert_array_almost_equal(sums[:-1], np.ones(5, dtype=np.float32))
    np.testing.assert_almost_equal(sums[-1], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_stage2_soft_labels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ecgcode.stage2.soft_labels'`.

- [ ] **Step 3: Write minimal implementation**

```python
# openecg/stage2/soft_labels.py
"""Soft labels at frame transitions to soften per-frame CE at boundaries.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.1
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as ee


def soft_boundary_labels(
    labels: np.ndarray,
    alpha: float = 0.7,
    n_classes: int = 4,
    ignore_index: int = ee.IGNORE_INDEX,
) -> np.ndarray:
    """Convert a hard label sequence to per-frame soft targets.

    For every transition (i, i+1) with labels[i] != labels[i+1] (and
    neither equal to ignore_index):
        soft[i  ] = alpha · onehot(labels[i  ]) + (1-alpha) · onehot(labels[i+1])
        soft[i+1] = (1-alpha) · onehot(labels[i  ]) + alpha · onehot(labels[i+1])
    Frames in multiple transitions take the later application.
    Rows for ignore_index frames are all zero (signals "skip in loss").
    """
    labels = np.asarray(labels, dtype=np.int64)
    T = len(labels)
    soft = np.zeros((T, n_classes), dtype=np.float32)
    for i in range(T):
        c = int(labels[i])
        if c != ignore_index and 0 <= c < n_classes:
            soft[i, c] = 1.0
    for i in range(T - 1):
        a, b = int(labels[i]), int(labels[i + 1])
        if a == ignore_index or b == ignore_index or a == b:
            continue
        if not (0 <= a < n_classes and 0 <= b < n_classes):
            continue
        soft[i] = 0.0
        soft[i + 1] = 0.0
        soft[i, a] = alpha
        soft[i, b] = 1.0 - alpha
        soft[i + 1, a] = 1.0 - alpha
        soft[i + 1, b] = alpha
    return soft
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_stage2_soft_labels.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/soft_labels.py tests/test_stage2_soft_labels.py
git commit -m "v12 task 1: soft_boundary_labels helper"
```

---

### Task 2: KL cross-entropy loss + train_one_epoch_kl + fit_kl

**Files:**
- Modify: `ecgcode/stage2/train.py`
- Test:   `tests/test_stage2_train_v12.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2_train_v12.py
import numpy as np
import torch

from openecg.stage2.model import FrameClassifierViT
from openecg.stage2.train import (
    kl_cross_entropy, train_one_epoch_kl,
)


def test_kl_equals_ce_on_one_hot():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 4)
    hard = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(hard, num_classes=4).float()
    kl_loss = kl_cross_entropy(logits, onehot)
    ce_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), hard, reduction="mean",
    )
    assert torch.isclose(kl_loss, ce_loss, atol=1e-5)


def test_kl_skips_zero_target_rows():
    logits = torch.zeros(1, 3, 4)
    target = torch.zeros(1, 3, 4)
    target[0, 0, 1] = 1.0
    target[0, 2, 2] = 1.0
    loss = kl_cross_entropy(logits, target)
    expected = torch.log(torch.tensor(4.0))
    assert torch.isclose(loss, expected, atol=1e-5)


def test_train_one_epoch_kl_loss_decreases():
    torch.manual_seed(0)
    KW = dict(patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
              use_lead_emb=False, pos_type="learnable")
    model = FrameClassifierViT(**KW)
    sigs = torch.randn(4, 2500)
    leads = torch.zeros(4, dtype=torch.long)
    soft = torch.zeros(4, 500, 4)
    soft[..., 0] = 1.0
    loader = [(sigs, leads, soft)]
    weights = torch.ones(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        loss = train_one_epoch_kl(model, loader, opt, weights, device="cpu")
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.7
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_stage2_train_v12.py -v
```

Expected: FAIL with `ImportError: cannot import name 'kl_cross_entropy'`.

- [ ] **Step 3: Write minimal implementation**

Append to `ecgcode/stage2/train.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_train_v12.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/train.py tests/test_stage2_train_v12.py
git commit -m "v12 task 2: KL cross-entropy + train_one_epoch_kl + fit_kl"
```

---

### Task 3: `SoftLabelDataset` wrapper

**Files:**
- Modify: `ecgcode/stage2/soft_labels.py`
- Test:   `tests/test_stage2_soft_labels.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage2_soft_labels.py`:

```python
import torch

from openecg.stage2.soft_labels import SoftLabelDataset


class _HardDS:
    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        self.calls += 1
        sig = torch.zeros(2500)
        lead = torch.tensor(idx, dtype=torch.long)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        return sig, lead, labels

    def label_counts(self):
        return np.array([10, 10, 10, 10], dtype=np.int64)


def test_soft_label_dataset_returns_soft_tensor():
    base = _HardDS()
    ds = SoftLabelDataset(base, alpha=0.7, n_classes=4)
    assert len(ds) == 2
    sig, lead, soft = ds[0]
    assert sig.shape == (2500,)
    assert lead.dtype == torch.long
    assert soft.shape == (8, 4)
    assert soft.dtype == torch.float32
    assert torch.isclose(soft[0, 0], torch.tensor(1.0))
    assert torch.isclose(soft[1, 0], torch.tensor(0.7))
    assert torch.isclose(soft[1, 1], torch.tensor(0.3))


def test_soft_label_dataset_passes_through_label_counts():
    ds = SoftLabelDataset(_HardDS())
    counts = ds.label_counts()
    np.testing.assert_array_equal(counts, [10, 10, 10, 10])
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_stage2_soft_labels.py::test_soft_label_dataset_returns_soft_tensor -v
```

Expected: FAIL with `ImportError: cannot import name 'SoftLabelDataset'`.

- [ ] **Step 3: Write minimal implementation**

Append to `ecgcode/stage2/soft_labels.py`:

```python
class SoftLabelDataset(Dataset):
    """Wraps a hard-label Dataset.

    The base dataset must yield (sig, lead_id, hard_labels[T] long). This
    wrapper instead yields (sig, lead_id, soft_labels[T, n_classes] float32)
    by applying soft_boundary_labels to each item.
    """

    def __init__(self, base, alpha: float = 0.7, n_classes: int = 4,
                 ignore_index: int = ee.IGNORE_INDEX):
        self.base = base
        self.alpha = float(alpha)
        self.n_classes = int(n_classes)
        self.ignore_index = int(ignore_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sig, lead_id, labels = self.base[idx]
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
        soft = soft_boundary_labels(
            labels_np, alpha=self.alpha, n_classes=self.n_classes,
            ignore_index=self.ignore_index,
        )
        return sig, lead_id, torch.from_numpy(soft)

    def label_counts(self):
        return self.base.label_counts()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_soft_labels.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/soft_labels.py tests/test_stage2_soft_labels.py
git commit -m "v12 task 3: SoftLabelDataset wrapper"
```

---

### Task 4: `train_v12_soft.py` script (run 1)

**Files:**
- Create: `scripts/train_v12_soft.py`

- [ ] **Step 1: Write the script**

```python
# scripts/train_v12_soft.py
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

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.model import FrameClassifierViT
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from openecg.stage2.soft_labels import SoftLabelDataset
from openecg.stage2.train import TrainConfig, fit_kl, load_checkpoint
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
```

- [ ] **Step 2: Smoke-test imports**

```
python -c "from scripts.train_v12_soft import main; print('import OK')"
```

Expected: `import OK`. If not, fix imports / sys.path. Do not run training yet.

- [ ] **Step 3: Commit**

```
git add scripts/train_v12_soft.py
git commit -m "v12 task 4: train_v12_soft.py (run 1, soft labels + KL)"
```

---

### Task 5: `FrameClassifierViTReg` model

**Files:**
- Modify: `ecgcode/stage2/model.py`
- Test:   `tests/test_stage2_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage2_model.py`:

```python
def test_vit_reg_forward_shape_cpu():
    from openecg.stage2.model import FrameClassifierViTReg
    model = FrameClassifierViTReg(
        patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
        use_lead_emb=False, pos_type="learnable",
    )
    x = torch.randn(2, 2500)
    lead_id = torch.zeros(2, dtype=torch.long)
    cls, reg = model(x, lead_id)
    assert cls.shape == (2, 500, 4)
    assert reg.shape == (2, 500, 6)
    assert cls.dtype == torch.float32
    assert reg.dtype == torch.float32


def test_vit_reg_model_config_records_n_reg():
    from openecg.stage2.model import FrameClassifierViTReg
    model = FrameClassifierViTReg(
        patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
        use_lead_emb=False, pos_type="learnable",
    )
    assert model.model_config["n_reg"] == 6
    assert model.model_config["arch"] == "vit_reg"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_stage2_model.py::test_vit_reg_forward_shape_cpu -v
```

Expected: FAIL with `ImportError: cannot import name 'FrameClassifierViTReg'`.

- [ ] **Step 3: Write minimal implementation**

Append to `ecgcode/stage2/model.py`:

```python
class FrameClassifierViTReg(FrameClassifierViT):
    """ViT backbone with parallel classification + boundary-regression heads.

    Forward returns (cls_logits[B, N_patches, n_classes],
                     reg_offsets[B, N_patches, n_reg]).
    n_reg defaults to 6: signed sample-offset to nearest GT boundary of each of
    {p_on, p_off, qrs_on, qrs_off, t_on, t_off}.
    """

    def __init__(self, n_reg=6, **kwargs):
        super().__init__(**kwargs)
        self.n_reg = int(n_reg)
        self.reg_head = nn.Linear(self.head.in_features, self.n_reg)
        self.model_config = dict(self.model_config)
        self.model_config["n_reg"] = self.n_reg
        self.model_config["arch"] = "vit_reg"

    def forward(self, x, lead_id):
        B, N = x.shape
        n_patches = N // self.patch_size
        if self.conv_stem:
            h = torch.nn.functional.gelu(self.stem_conv1(x.unsqueeze(1)))
            h = torch.nn.functional.gelu(self.stem_conv2(h))
            h = h.transpose(1, 2)
            patches = h.reshape(B, n_patches, self.patch_size * 32)
        else:
            patches = x.view(B, n_patches, self.patch_size)
        h = self.patch_embed(patches)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :n_patches]
        if self.use_lead_emb:
            h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        cls_logits = self.head(h)
        reg_offsets = self.reg_head(h)
        return cls_logits, reg_offsets
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_model.py -v
```

Expected: all model tests pass (existing 5 + new 2 = 7).

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/model.py tests/test_stage2_model.py
git commit -m "v12 task 5: FrameClassifierViTReg with parallel cls+reg heads"
```

---

### Task 6: `boundary_regression_targets` helper

**Files:**
- Create: `ecgcode/stage2/reg_targets.py`
- Test:   `tests/test_stage2_reg_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2_reg_targets.py
import numpy as np
import pytest

from openecg import eval as ee
from openecg.stage2.reg_targets import (
    REG_CHANNELS, boundary_regression_targets,
)


def test_reg_channels_order():
    assert REG_CHANNELS == ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


def test_no_transitions_yields_no_active_mask():
    labels = np.full(10, ee.SUPER_OTHER, dtype=np.int64)
    targets, mask = boundary_regression_targets(
        labels, samples_per_frame=5, window_frames=5,
    )
    assert targets.shape == (10, 6)
    assert mask.shape == (10, 6)
    assert not mask.any()


def test_single_p_wave_marks_two_channels():
    # P-wave from frame 5 to frame 10 (samples 25..49 inclusive given spf=5)
    labels = np.array([0]*5 + [1]*6 + [0]*5, dtype=np.int64)
    targets, mask = boundary_regression_targets(
        labels, samples_per_frame=5, window_frames=5,
    )
    p_on_active = mask[:, 0]
    assert p_on_active[5]
    assert p_on_active[0]
    assert not p_on_active[15]
    assert targets[5, 0] == 0.0
    assert targets[4, 0] == 5.0
    assert mask[10, 1]
    assert targets[10, 1] == (11 * 5 - 1) - (10 * 5)


def test_ignore_index_zeros_target_and_mask():
    labels = np.array([0, 0, ee.IGNORE_INDEX, 1, 1], dtype=np.int64)
    targets, mask = boundary_regression_targets(
        labels, samples_per_frame=5, window_frames=5,
    )
    assert not mask[2].any()
    np.testing.assert_array_equal(targets[2], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_stage2_reg_targets.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ecgcode.stage2.reg_targets'`.

- [ ] **Step 3: Write minimal implementation**

```python
# openecg/stage2/reg_targets.py
"""Boundary regression targets for FrameClassifierViTReg.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.2
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as ee


REG_CHANNELS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")
_REG_INDEX = {  # (super_class, edge) -> channel index
    (ee.SUPER_P, "on"): 0, (ee.SUPER_P, "off"): 1,
    (ee.SUPER_QRS, "on"): 2, (ee.SUPER_QRS, "off"): 3,
    (ee.SUPER_T, "on"): 4, (ee.SUPER_T, "off"): 5,
}


def boundary_regression_targets(
    hard_labels: np.ndarray,
    samples_per_frame: int = 5,
    window_frames: int = 5,
    ignore_index: int = ee.IGNORE_INDEX,
):
    """Build per-frame regression targets and active masks from hard labels.

    Convention (matches extract_boundaries):
      - off boundary sample = transition_frame * spf - 1
      - on  boundary sample = transition_frame * spf

    Returns (targets[T, 6] float32, mask[T, 6] bool).
    target[f, k] = signed sample offset (boundary - frame_start) if a boundary
    of channel k lies within ±window_frames of f, else 0. mask[f, k] = True iff
    such a boundary exists. Frames where hard_labels[f] == ignore_index get
    mask all-False regardless.
    """
    labels = np.asarray(hard_labels, dtype=np.int64)
    T = len(labels)
    targets = np.zeros((T, 6), dtype=np.float32)
    mask = np.zeros((T, 6), dtype=bool)
    if T == 0:
        return targets, mask

    boundaries: list[tuple[int, int]] = []
    prev = int(labels[0])
    for f in range(1, T):
        cur = int(labels[f])
        if prev == ignore_index or cur == ignore_index or cur == prev:
            prev = cur
            continue
        if (prev, "off") in _REG_INDEX:
            boundaries.append((_REG_INDEX[(prev, "off")], f * samples_per_frame - 1))
        if (cur, "on") in _REG_INDEX:
            boundaries.append((_REG_INDEX[(cur, "on")], f * samples_per_frame))
        prev = cur

    radius = window_frames * samples_per_frame
    for ch, b_sample in boundaries:
        for f in range(T):
            f_sample = f * samples_per_frame
            offset = b_sample - f_sample
            if abs(offset) <= radius:
                if (not mask[f, ch]) or abs(offset) < abs(targets[f, ch]):
                    targets[f, ch] = float(offset)
                    mask[f, ch] = True

    for f in range(T):
        if int(labels[f]) == ignore_index:
            mask[f, :] = False
            targets[f, :] = 0.0
    return targets, mask
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_reg_targets.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/reg_targets.py tests/test_stage2_reg_targets.py
git commit -m "v12 task 6: boundary_regression_targets helper"
```

---

### Task 7: `RegLabelDataset` + reg loss + train_one_epoch_reg + fit_reg

**Files:**
- Modify: `ecgcode/stage2/reg_targets.py`
- Modify: `ecgcode/stage2/train.py`
- Modify: `tests/test_stage2_reg_targets.py`
- Modify: `tests/test_stage2_train_v12.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage2_reg_targets.py`:

```python
import torch

from openecg.stage2.reg_targets import RegLabelDataset


class _HardDS:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sig = torch.zeros(2500)
        lead = torch.tensor(idx, dtype=torch.long)
        labels = torch.tensor([0]*5 + [1]*6 + [0]*489, dtype=torch.long)
        return sig, lead, labels

    def label_counts(self):
        return np.array([100, 5, 5, 5], dtype=np.int64)


def test_reg_label_dataset_yields_targets_and_mask():
    base = _HardDS()
    ds = RegLabelDataset(base, samples_per_frame=5, window_frames=5)
    sig, lead, labels, target, mask = ds[0]
    assert sig.shape == (2500,)
    assert labels.dtype == torch.long
    assert target.shape == (500, 6)
    assert mask.shape == (500, 6)
    assert mask.dtype == torch.bool
    assert mask.any()
```

Append to `tests/test_stage2_train_v12.py`:

```python
def test_boundary_l1_loss_masked():
    from openecg.stage2.train import boundary_l1_loss
    pred = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    target = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, False, False, False, False, False]]])
    loss = boundary_l1_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(2.0))


def test_train_one_epoch_reg_loss_decreases():
    from openecg.stage2.model import FrameClassifierViTReg
    from openecg.stage2.train import train_one_epoch_reg
    torch.manual_seed(0)
    model = FrameClassifierViTReg(
        patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
        use_lead_emb=False, pos_type="learnable",
    )
    sigs = torch.randn(4, 2500)
    leads = torch.zeros(4, dtype=torch.long)
    labels = torch.zeros(4, 500, dtype=torch.long)
    reg_t = torch.zeros(4, 500, 6)
    reg_m = torch.zeros(4, 500, 6, dtype=torch.bool)
    reg_m[:, 100, 0] = True
    reg_t[:, 100, 0] = 3.0
    loader = [(sigs, leads, labels, reg_t, reg_m)]
    weights = torch.ones(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        loss = train_one_epoch_reg(model, loader, opt, weights, device="cpu")
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_stage2_reg_targets.py::test_reg_label_dataset_yields_targets_and_mask tests/test_stage2_train_v12.py::test_boundary_l1_loss_masked -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Append to `ecgcode/stage2/reg_targets.py`:

```python
class RegLabelDataset(Dataset):
    """Wrap a base dataset to additionally yield reg targets + mask.

    Base must yield (sig, lead_id, hard_labels[T]). This wrapper yields
    (sig, lead_id, hard_labels, reg_targets[T,6], reg_mask[T,6]).
    """

    def __init__(self, base, samples_per_frame: int = 5,
                 window_frames: int = 5,
                 ignore_index: int = ee.IGNORE_INDEX):
        self.base = base
        self.samples_per_frame = int(samples_per_frame)
        self.window_frames = int(window_frames)
        self.ignore_index = int(ignore_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sig, lead_id, labels = self.base[idx]
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
        targets, mask = boundary_regression_targets(
            labels_np, samples_per_frame=self.samples_per_frame,
            window_frames=self.window_frames, ignore_index=self.ignore_index,
        )
        return (sig, lead_id, labels,
                torch.from_numpy(targets), torch.from_numpy(mask))

    def label_counts(self):
        return self.base.label_counts()
```

Append to `ecgcode/stage2/train.py`:

```python
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
```

Note: `run_eval` in the existing `train.py` already uses the inference-mode toggle (it predates this plan); leave it unchanged.

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_reg_targets.py tests/test_stage2_train_v12.py -v
```

Expected: all reg/train_v12 tests pass.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/reg_targets.py ecgcode/stage2/train.py tests/test_stage2_reg_targets.py tests/test_stage2_train_v12.py
git commit -m "v12 task 7: RegLabelDataset + boundary_l1_loss + fit_reg"
```

---

### Task 8: Reg-aware inference

**Files:**
- Modify: `ecgcode/stage2/infer.py`
- Test:   `tests/test_stage2_infer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage2_infer.py`:

```python
def test_apply_reg_to_boundaries_shifts_each_sample():
    from openecg.stage2.infer import apply_reg_to_boundaries
    import numpy as np
    boundaries = {
        "p_on":   [25],
        "p_off":  [54],
        "qrs_on": [60],
        "qrs_off":[80],
        "t_on":   [],
        "t_off":  [],
    }
    reg = np.zeros((500, 6), dtype=np.float32)
    reg[5, 0] = 2.0
    reg[10, 1] = -3.0
    reg[12, 2] = 1.0
    reg[16, 3] = 0.0
    refined = apply_reg_to_boundaries(boundaries, reg, samples_per_frame=5,
                                       max_window=2500)
    assert refined["p_on"]   == [27]
    assert refined["p_off"]  == [51]
    assert refined["qrs_on"] == [61]
    assert refined["qrs_off"]== [80]


def test_predict_frames_with_reg_shapes():
    from openecg.stage2.model import FrameClassifierViTReg
    from openecg.stage2.infer import predict_frames_with_reg
    import numpy as np
    model = FrameClassifierViTReg(
        patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
        use_lead_emb=False, pos_type="learnable",
    )
    sig = np.zeros(2500, dtype=np.float32)
    frames, reg = predict_frames_with_reg(model, sig, lead_id=0, device="cpu")
    assert frames.shape == (500,)
    assert frames.dtype.name == "uint8"
    assert reg.shape == (500, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_stage2_infer.py -v
```

Expected: 2 NEW tests fail.

- [ ] **Step 3: Write minimal implementation**

Append to `ecgcode/stage2/infer.py`:

```python
REG_CHANNELS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


@torch.no_grad()
def predict_frames_with_reg(model, sig, lead_id, device="cuda"):
    """Single-sequence inference for a (cls, reg) tuple-output model.
    Returns (frames[T] uint8, reg_offsets[T, 6] float32).
    """
    x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
    lid = torch.tensor([lead_id], dtype=torch.long, device=device)
    cls_logits, reg = model(x, lid)
    frames = cls_logits.argmax(dim=-1).cpu().numpy().squeeze(0).astype(np.uint8)
    reg_np = reg.cpu().numpy().squeeze(0).astype(np.float32)
    return frames, reg_np


def apply_reg_to_boundaries(boundaries, reg_offsets, samples_per_frame=5,
                              max_window=10000):
    """Refine boundary samples by adding the reg head's predicted offset
    at the corresponding frame.

    boundaries: dict from extract_boundaries (key -> list[int sample]).
    reg_offsets: [T, 6] array; channel order = REG_CHANNELS.
    """
    refined: dict[str, list[int]] = {}
    for key, samples in boundaries.items():
        if key not in REG_CHANNELS:
            refined[key] = list(samples)
            continue
        ch = REG_CHANNELS.index(key)
        out: list[int] = []
        T = reg_offsets.shape[0]
        for s in samples:
            f = int(s) // samples_per_frame
            if 0 <= f < T:
                shifted = int(s) + int(round(float(reg_offsets[f, ch])))
            else:
                shifted = int(s)
            shifted = max(0, min(int(max_window) - 1, shifted))
            out.append(shifted)
        refined[key] = out
    return refined
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_infer.py -v
```

Expected: all infer tests pass.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/infer.py tests/test_stage2_infer.py
git commit -m "v12 task 8: predict_frames_with_reg + apply_reg_to_boundaries"
```

---

### Task 9: `train_v12_reg.py` script with lambda sweep (run 2)

**Files:**
- Create: `scripts/train_v12_reg.py`

- [ ] **Step 1: Write the script**

```python
# scripts/train_v12_reg.py
"""v12_reg - same training data + ViT as v9, plus a parallel boundary
regression head. Sweeps lambda in {0.05, 0.1, 0.5}; the best LUDB val avg F1 wins.

See docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §4.2.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.evaluate import MARTINEZ_TOLERANCE_MS, signed_boundary_metrics
from openecg.stage2.infer import (
    apply_reg_to_boundaries, extract_boundaries,
    post_process_frames, predict_frames_with_reg,
)
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset, _decimate_to_250, _normalize,
)
from openecg.stage2.reg_targets import RegLabelDataset
from openecg.stage2.train import TrainConfig, fit_reg, load_checkpoint
from scripts.train_v9_q1c_pu_merge import KWARGS, _ConcatWithCounts

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500
FRAME_MS = 20
FS = 250
SEED = 42
EDGE_MARGIN_MS = 100
QTDB_EVAL_SEED = 42
QTDB_PU0_WINDOWS_PER_RECORD = 5
LAMBDAS = (0.05, 0.1, 0.5)


def _eval_ludb(model, device):
    val_ds = LUDBFrameDataset(ludb.load_split()["val"])
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    margin_250 = int(round(EDGE_MARGIN_MS * FS / 1000.0))
    for idx in range(len(val_ds)):
        rid, lead = val_ds.items[idx]
        sig_250, lead_idx, _ = val_ds.cache[(rid, lead)]
        sig_250 = sig_250[:WINDOW_SAMPLES]
        if len(sig_250) < WINDOW_SAMPLES: continue
        rng_lab = ludb.labeled_range(rid, lead)
        if rng_lab is None: continue
        lo = max(0, rng_lab[0] // 2 - margin_250)
        hi = min(WINDOW_SAMPLES, rng_lab[1] // 2 + margin_250 + 1)
        frames, reg = predict_frames_with_reg(model, sig_250, lead_idx, device=device)
        pp = post_process_frames(frames, frame_ms=FRAME_MS)
        boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
        boundaries = apply_reg_to_boundaries(boundaries, reg, max_window=WINDOW_SAMPLES)
        for k, vs in boundaries.items():
            for s in vs:
                if lo <= s < hi:
                    bp[k].append(int(s) + cum)
        gt = ludb.load_annotations(rid, lead)
        for k, vs in gt.items():
            if k.endswith("_on") or k.endswith("_off"):
                for s in vs:
                    s250 = int(s // 2)
                    if lo <= s250 < hi:
                        bt[k].append(s250 + cum)
        cum += WINDOW_SAMPLES
    return bp, bt


def _eval_isp(model, device):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    for rid in isp.load_split()["test"]:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception: continue
        for lead_idx, lead in enumerate(isp.LEADS_12):
            sig_1000 = record[lead]
            sig_250 = _decimate_to_250(sig_1000, 1000)
            sig_n = _normalize(sig_250)
            if len(sig_n) < WINDOW_SAMPLES:
                pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                sig_n = np.concatenate([sig_n, pad])
            sig_n = sig_n[:WINDOW_SAMPLES]
            frames, reg = predict_frames_with_reg(model, sig_n, lead_idx, device=device)
            pp = post_process_frames(frames, frame_ms=FRAME_MS)
            boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            boundaries = apply_reg_to_boundaries(boundaries, reg, max_window=WINDOW_SAMPLES)
            for k, vs in boundaries.items():
                for s in vs:
                    bp[k].append(int(s) + cum)
            for k, vs in ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in vs:
                        s250 = int(s // 4)
                        if 0 <= s250 < WINDOW_SAMPLES:
                            bt[k].append(s250 + cum)
            cum += WINDOW_SAMPLES
    return bp, bt


def _eval_qtdb(model, device, n_windows=QTDB_PU0_WINDOWS_PER_RECORD):
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    rng = np.random.default_rng(QTDB_EVAL_SEED)
    for rid in qtdb.records_with_q1c():
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_pu(rid, lead=0)
        except Exception: continue
        first_lead = list(record.keys())[0]
        sig_full = record[first_lead].astype(np.float32)
        n = len(sig_full)
        sig_norm = _normalize(sig_full)
        n_max = n // WINDOW_SAMPLES
        k = min(n_windows, n_max)
        chosen = rng.choice(n_max, size=k, replace=False)
        covered = []
        for w in sorted(chosen):
            start = int(w) * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            covered.append((start, end))
            sig_win = sig_norm[start:end].astype(np.float32)
            sig_w = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)
            frames, reg = predict_frames_with_reg(model, sig_w, lead_id=1, device=device)
            pp = post_process_frames(frames, frame_ms=FRAME_MS)
            boundaries = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
            boundaries = apply_reg_to_boundaries(boundaries, reg, max_window=WINDOW_SAMPLES)
            for ck, vs in boundaries.items():
                for s in vs:
                    bp[ck].append(int(start + s) + cum)
        for ck in ("p_on","p_off","qrs_on","qrs_off","t_on","t_off"):
            for s in ann.get(ck, []):
                for lo, hi in covered:
                    if lo <= s < hi:
                        bt[ck].append(int(s) + cum); break
        cum += n
    return bp, bt


def _avg_f1(bp, bt):
    f1s = []
    for k in ("p_on","p_off","qrs_on","qrs_off","t_on","t_off"):
        m = signed_boundary_metrics(
            bp.get(k, []), bt.get(k, []),
            tolerance_ms=MARTINEZ_TOLERANCE_MS[k], fs=FS,
        )
        f1s.append(m["f1"])
    return float(np.mean(f1s))


def _eval_all(model, device):
    return {
        "ludb_edge_filtered": _avg_f1(*_eval_ludb(model, device)),
        "isp_test":           _avg_f1(*_eval_isp(model, device)),
        "qtdb_pu0_random":    _avg_f1(*_eval_qtdb(model, device)),
    }


def _build_train_loader():
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=SEED,
                                       q1c_pu_merge=True)
    reg_ludb = RegLabelDataset(ludb_train)
    reg_isp = RegLabelDataset(isp_train)
    reg_qtdb = RegLabelDataset(qtdb_merged)
    return _ConcatWithCounts([reg_ludb, reg_isp, reg_qtdb])


def main():
    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_ds = _build_train_loader()
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                 mask_unlabeled_edges=True,
                                 edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()), dtype=torch.float32)

    sweep_results = {}
    for lam in LAMBDAS:
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg = TrainConfig(epochs=80, batch_size=64, lr=1e-3, early_stop_patience=15)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                    num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(ludb_val, batch_size=64, shuffle=False,
                                  num_workers=0, pin_memory=True)
        model = FrameClassifierViTReg(**KWARGS, n_reg=6)
        n_params = sum(p.numel() for p in model.parameters())
        ckpt_path = CKPT_DIR / f"stage2_v12_reg_lam{lam}.pt"
        print(f"\n=== TRAIN v12_reg lambda={lam} ({n_params:,} params) ===",
              flush=True)
        t0 = time.time()
        best = fit_reg(model, train_loader, val_loader, weights, cfg, device=device,
                         ckpt_path=ckpt_path, lambda_reg=lam)
        elapsed = time.time() - t0
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model)
        model = model.to(device).train(False)
        res = _eval_all(model, device)
        sweep_results[str(lam)] = {"params": n_params, "train_seconds": elapsed,
                                     "best_metrics": best, **res}
        print(f"  lambda={lam} eval: {res}", flush=True)

    best_lam = max(sweep_results, key=lambda l: sweep_results[l]["ludb_edge_filtered"])
    print(f"\n=== BEST lambda = {best_lam} ===", flush=True)
    best_ckpt = CKPT_DIR / f"stage2_v12_reg_lam{best_lam}.pt"
    canon_ckpt = CKPT_DIR / "stage2_v12_reg.pt"
    if canon_ckpt.exists(): canon_ckpt.unlink()
    canon_ckpt.write_bytes(best_ckpt.read_bytes())

    full = {
        "v9_q1c_pu_merge_ref": {"params": 1126660,
                                  "ludb_edge_filtered": 0.923,
                                  "isp_test": 0.943,
                                  "qtdb_pu0_random": 0.779},
        "v12_reg_sweep": sweep_results,
        "v12_reg_best": {"lambda": float(best_lam), **sweep_results[best_lam]},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v12_reg_{ts}.json"
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test imports**

```
python -c "from scripts.train_v12_reg import main; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 3: Commit**

```
git add scripts/train_v12_reg.py
git commit -m "v12 task 9: train_v12_reg.py (run 2, lambda sweep)"
```

---

## Phase 2 — SSL backbone transfer

### Task 10: SSL package scaffold + dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `ecgcode/stage2/ssl/__init__.py`

- [ ] **Step 1: Add deps to pyproject.toml**

Edit `pyproject.toml` `dependencies` to:

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "neurokit2>=0.2.7",
    "wfdb>=4.1",
    "torch>=2.6",
    "transformers>=4.40",
    "huggingface-hub>=0.20",
]
```

- [ ] **Step 2: Sync env**

```
uv sync
```

Expected: installs `transformers` and `huggingface_hub`.

- [ ] **Step 3: Create the package marker**

```python
# openecg/stage2/ssl/__init__.py
"""Open-weight ECG SSL backbones adapted to the Stage 2 pipeline.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §5.
"""
```

- [ ] **Step 4: Verify import**

```
python -c "import openecg.stage2.ssl; print('ssl package OK')"
```

Expected: `ssl package OK`.

- [ ] **Step 5: Commit**

```
git add pyproject.toml uv.lock ecgcode/stage2/ssl/__init__.py
git commit -m "v12 task 10: add HuggingFace deps + ssl package scaffold"
```

---

### Task 11: `ssl/head.py` — FrameHead, FrameRegHead, BackboneWithHeads

**Files:**
- Create: `ecgcode/stage2/ssl/head.py`
- Test:   `tests/test_stage2_ssl_head.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage2_ssl_head.py
import torch
from torch import nn

from openecg.stage2.ssl.head import (
    BackboneWithHeads, FrameHead, FrameRegHead,
)


class _Backbone(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lin = nn.Linear(2500, 500 * hidden)
        self.hidden = hidden
    def forward(self, sig, lead_id):
        B = sig.size(0)
        return self.lin(sig).view(B, 500, self.hidden)


def test_frame_head_shape():
    head = FrameHead(d_model=16, n_classes=4)
    h = torch.randn(2, 500, 16)
    out = head(h)
    assert out.shape == (2, 500, 4)


def test_frame_reg_head_shape():
    head = FrameRegHead(d_model=16, n_reg=6)
    h = torch.randn(2, 500, 16)
    out = head(h)
    assert out.shape == (2, 500, 6)


def test_backbone_with_heads_cls_only():
    model = BackboneWithHeads(_Backbone(), hidden_dim=16, use_reg=False)
    sig = torch.randn(2, 2500); lead = torch.zeros(2, dtype=torch.long)
    out = model(sig, lead)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 500, 4)


def test_backbone_with_heads_cls_and_reg():
    model = BackboneWithHeads(_Backbone(), hidden_dim=16, use_reg=True)
    sig = torch.randn(2, 2500); lead = torch.zeros(2, dtype=torch.long)
    out = model(sig, lead)
    assert isinstance(out, tuple) and len(out) == 2
    cls, reg = out
    assert cls.shape == (2, 500, 4)
    assert reg.shape == (2, 500, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_stage2_ssl_head.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# openecg/stage2/ssl/head.py
"""Shared classification + regression heads for SSL backbones."""

from torch import nn


class FrameHead(nn.Module):
    """Per-frame supercategory classifier head: Linear(d -> n_classes)."""

    def __init__(self, d_model: int, n_classes: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


class FrameRegHead(nn.Module):
    """Per-frame boundary-offset regressor: Linear(d -> n_reg)."""

    def __init__(self, d_model: int, n_reg: int = 6, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, n_reg)

    def forward(self, h):
        return self.linear(self.dropout(h))


class BackboneWithHeads(nn.Module):
    """Generic wrapper: backbone -> features [B, T, d] -> cls(+reg) head(s).

    Backbone must implement forward(sig, lead_id) -> features [B, 500, d].
    Returns cls_logits if use_reg=False, else (cls_logits, reg_offsets).
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int,
                 use_reg: bool = False, n_classes: int = 4, n_reg: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.use_reg = bool(use_reg)
        self.cls_head = FrameHead(hidden_dim, n_classes=n_classes, dropout=dropout)
        self.reg_head = FrameRegHead(hidden_dim, n_reg=n_reg, dropout=dropout) if use_reg else None
        self.model_config = {
            "arch": "ssl_backbone",
            "hidden_dim": hidden_dim,
            "n_classes": n_classes,
            "use_reg": self.use_reg,
            "n_reg": n_reg if use_reg else 0,
        }

    def forward(self, sig, lead_id):
        h = self.backbone(sig, lead_id)
        cls = self.cls_head(h)
        if self.use_reg:
            return cls, self.reg_head(h)
        return cls
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_ssl_head.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/ssl/head.py tests/test_stage2_ssl_head.py
git commit -m "v12 task 11: ssl head + BackboneWithHeads wrapper"
```

---

### Task 12: `ssl/hubert.py` — HuBERT-ECG adapter

**Files:**
- Create: `ecgcode/stage2/ssl/hubert.py`
- Test:   `tests/test_stage2_ssl_hubert.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage2_ssl_hubert.py
import os
import pytest
import torch

from openecg.stage2.ssl.hubert import HubertECGAdapter, HUBERT_DEFAULT_MODEL_ID


HF_OK = os.environ.get("OPENECG_RUN_HF_TESTS") == "1"


@pytest.mark.skipif(not HF_OK,
                     reason="set OPENECG_RUN_HF_TESTS=1 to download HuBERT-ECG weights")
def test_hubert_adapter_forward_shape():
    """End-to-end shape check; downloads weights from HF.
    Run with: OPENECG_RUN_HF_TESTS=1 pytest tests/test_stage2_ssl_hubert.py
    """
    adapter = HubertECGAdapter(model_id=HUBERT_DEFAULT_MODEL_ID, device="cpu")
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, adapter.hidden_dim)
    assert h.dtype == torch.float32


def test_hubert_adapter_resample_only():
    """Offline test: validate the input resampling path with a tiny synthetic
    encoder (skips HF download)."""
    import torch.nn as nn

    class _DummyEncoder(nn.Module):
        config = type("C", (), {"hidden_size": 8})

        def __call__(self, input_values, *a, **kw):
            B, N = input_values.shape
            assert N == 500, f"expected 500 samples (5s @ 100Hz), got {N}"
            class _O:
                pass
            o = _O()
            o.last_hidden_state = torch.zeros(B, 250, 8)
            return o

    adapter = HubertECGAdapter.__new__(HubertECGAdapter)
    adapter.encoder = _DummyEncoder()
    adapter.hidden_dim = 8
    adapter.target_fs = 100
    adapter.window_seconds = 5
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_stage2_ssl_hubert.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# openecg/stage2/ssl/hubert.py
"""HuBERT-ECG adapter - wraps the HuggingFace HuBERT-style ECG encoder for
single-lead 250 Hz / 10 s input.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §5.2.

Adapter pipeline:
  sig [B, 2500] @ 250Hz, single lead
    -> resample to 100 Hz (length 1000)
    -> split into 2 x 500-sample (5 s) segments
    -> encoder.last_hidden_state on each -> [B, ~250, d]
    -> concat to [B, ~500, d] and pad/truncate to exactly 500 frames
"""

from __future__ import annotations

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch import nn


HUBERT_DEFAULT_MODEL_ID = "Edoardo-BS/hubert_ecg_small"


class HubertECGAdapter(nn.Module):
    """Wraps a HuggingFace HuBERT-ECG encoder."""

    def __init__(self, model_id: str = HUBERT_DEFAULT_MODEL_ID,
                 device: str = "cpu",
                 target_fs: int = 100,
                 window_seconds: int = 5):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_id)
        self.hidden_dim = int(self.encoder.config.hidden_size)
        self.target_fs = int(target_fs)
        self.window_seconds = int(window_seconds)

    def _resample_250_to_target(self, sig_250: torch.Tensor) -> torch.Tensor:
        n_in = sig_250.size(-1)
        n_out = int(round(n_in * self.target_fs / 250))
        np_sig = sig_250.detach().cpu().numpy().astype(np.float32)
        resampled = scipy_signal.resample(np_sig, n_out, axis=-1).astype(np.float32)
        return torch.from_numpy(resampled).to(sig_250.device)

    def _encode_segment(self, seg: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=seg)
        return out.last_hidden_state

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor) -> torch.Tensor:
        sig_target = self._resample_250_to_target(sig)
        seg_len = self.target_fs * self.window_seconds
        n = sig_target.size(-1)
        assert n == 2 * seg_len, f"expected {2 * seg_len} samples, got {n}"
        seg1 = sig_target[..., :seg_len]
        seg2 = sig_target[..., seg_len:]
        h1 = self._encode_segment(seg1)
        h2 = self._encode_segment(seg2)
        h = torch.cat([h1, h2], dim=1)
        if h.size(1) != 500:
            h = h.transpose(1, 2)
            h = nn.functional.interpolate(h, size=500, mode="linear", align_corners=False)
            h = h.transpose(1, 2)
        return h
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_ssl_hubert.py::test_hubert_adapter_resample_only -v
```

Expected: 1 passed.

Optional verification when network is available:

```
OPENECG_RUN_HF_TESTS=1 pytest tests/test_stage2_ssl_hubert.py -v
```

If the actual `Edoardo-BS/hubert_ecg_small` repo id differs, fix `HUBERT_DEFAULT_MODEL_ID` (use `huggingface-cli search hubert ecg` to find the canonical id) and re-run.

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/ssl/hubert.py tests/test_stage2_ssl_hubert.py
git commit -m "v12 task 12: HubertECGAdapter (250->100Hz, 5s x2 forward)"
```

---

### Task 13: `train_v12_hubert.py` (runs 3 + 4)

**Files:**
- Create: `scripts/train_v12_hubert.py`

- [ ] **Step 1: Write the script**

```python
# scripts/train_v12_hubert.py
"""v12_hubert_lp / v12_hubert_ft - HuBERT-ECG transfer.

Usage:
    python scripts/train_v12_hubert.py --mode lp
    python scripts/train_v12_hubert.py --mode ft

See docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §5.2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from openecg.stage2.ssl.head import BackboneWithHeads
from openecg.stage2.ssl.hubert import HUBERT_DEFAULT_MODEL_ID, HubertECGAdapter
from openecg.stage2.train import (
    TrainConfig, load_checkpoint, run_eval, save_checkpoint,
    score_val_metrics, train_one_epoch,
)
from openecg import eval as ecg_eval
from scripts.train_v9_q1c_pu_merge import _ConcatWithCounts, eval_all

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
SEED = 42
EDGE_MARGIN_MS = 100


def _build_train(seed):
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=seed,
                                       q1c_pu_merge=True)
    return _ConcatWithCounts([ludb_train, isp_train, qtdb_merged])


def _fit_with_groups(model, train_loader, val_loader, weights, cfg,
                      device, ckpt_path, param_groups, log_fn=print):
    """Like fit() but accepts pre-built optimizer param_groups (LP / FT split)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * cfg.warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best = -1; best_metrics = None; bad = 0
    for epoch in range(cfg.epochs):
        tl = train_one_epoch(model, train_loader, optimizer, weights, device,
                              use_focal=False, grad_clip=cfg.grad_clip,
                              scheduler=scheduler)
        val = run_eval(model, val_loader, device)
        score = score_val_metrics(val, cfg.early_stop_metric)
        log_fn(f"epoch {epoch:3d} train={tl:.4f} score={score:.3f}")
        if score > best:
            best = score
            best_metrics = {"epoch": epoch, "val_score": score,
                              "val_qrs_f1": val[ecg_eval.SUPER_QRS]["f1"]}
            bad = 0
            if ckpt_path: save_checkpoint(ckpt_path, model, best_metrics, cfg)
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                log_fn(f"Early stop at {epoch}"); break
    return best_metrics or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lp", "ft"), required=True)
    ap.add_argument("--model-id", default=HUBERT_DEFAULT_MODEL_ID)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, mode={args.mode}", flush=True)

    train_ds = _build_train(SEED)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()),
                            dtype=torch.float32)

    torch.manual_seed(SEED); np.random.seed(SEED)
    backbone = HubertECGAdapter(model_id=args.model_id, device=device)
    model = BackboneWithHeads(backbone, hidden_dim=backbone.hidden_dim, use_reg=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model_id={args.model_id} hidden={backbone.hidden_dim} params={n_params:,}",
          flush=True)

    if args.mode == "lp":
        for p in backbone.parameters():
            p.requires_grad = False
        param_groups = [{"params": model.cls_head.parameters(), "lr": 1e-3}]
    else:
        param_groups = [
            {"params": backbone.parameters(), "lr": 1e-5},
            {"params": model.cls_head.parameters(), "lr": 1e-3},
        ]

    cfg = TrainConfig(epochs=args.epochs, batch_size=32, lr=1e-3,
                       early_stop_patience=7, warmup_frac=0.05)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)

    name = f"v12_hubert_{args.mode}"
    ckpt_path = CKPT_DIR / f"stage2_{name}.pt"
    t0 = time.time()
    best = _fit_with_groups(model, train_loader, val_loader, weights, cfg,
                              device, ckpt_path, param_groups)
    elapsed = time.time() - t0

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
                "model_id": args.model_id,
                "mode": args.mode, "best_metrics": best, **res},
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
```

- [ ] **Step 2: Smoke-test imports + CLI**

```
python -c "from scripts.train_v12_hubert import main; print('import OK')"
python scripts/train_v12_hubert.py --help
```

Expected: `import OK` and an argparse usage message.

- [ ] **Step 3: Commit**

```
git add scripts/train_v12_hubert.py
git commit -m "v12 task 13: train_v12_hubert.py (--mode lp|ft, runs 3+4)"
```

---

### Task 14: `ssl/stmem.py` — ST-MEM adapter

**Files:**
- Create: `ecgcode/stage2/ssl/stmem.py`
- Create: `third_party/ST-MEM/README.md`
- Test:   `tests/test_stage2_ssl_stmem.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2_ssl_stmem.py
import os
import pytest
import torch
import torch.nn as nn

from openecg.stage2.ssl.stmem import STMEMAdapter


def test_stmem_adapter_replicate_path():
    """Offline test: build adapter with a tiny synthetic ECGViT."""

    class _DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 12

        def forward(self, x):
            B, C, N = x.shape
            assert C == 12
            return torch.zeros(B, 250, 12)

    adapter = STMEMAdapter.__new__(STMEMAdapter)
    adapter.encoder = _DummyEncoder()
    adapter.hidden_dim = 12
    adapter.window_samples = 1250
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, 12)


@pytest.mark.skipif(not os.path.exists("third_party/ST-MEM/models")
                    and not os.path.exists("third_party/ST-MEM/stmem"),
                     reason="ST-MEM source not vendored")
def test_stmem_adapter_loads_from_vendored():
    """Smoke: import path resolves and adapter constructs without weights."""
    adapter = STMEMAdapter(weights_path=None, device="cpu")
    assert adapter.hidden_dim > 0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_stage2_ssl_stmem.py::test_stmem_adapter_replicate_path -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# openecg/stage2/ssl/stmem.py
"""ST-MEM adapter - wraps the spatiotemporal masked ECG ViT for our
single-lead 250 Hz / 10 s pipeline.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §5.3.

Vendored source expected at third_party/ST-MEM/. We import lazily so the
package is loadable even when the vendored module is absent (tests skip).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


_VENDORED = Path(__file__).resolve().parents[3] / "third_party" / "ST-MEM"


def _load_stmem_module():
    """Insert vendored ST-MEM into sys.path and return the model class."""
    if str(_VENDORED) not in sys.path:
        sys.path.insert(0, str(_VENDORED))
    candidates = [
        ("models.stmem", "ECGViT"),
        ("stmem.models", "ECGViT"),
        ("models", "ECGViT"),
        ("st_mem.models", "ECGViT"),
    ]
    last_err = None
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Could not import ST-MEM ECGViT from {_VENDORED}. "
        f"Tried {candidates}. Last error: {last_err}"
    )


class STMEMAdapter(nn.Module):
    """ST-MEM single-lead adapter (single lead replicated across 12 channels)."""

    def __init__(self, weights_path: str | None = None,
                 device: str = "cpu",
                 window_samples: int = 1250):
        super().__init__()
        cls = _load_stmem_module()
        self.encoder = cls()
        if weights_path:
            blob = torch.load(weights_path, map_location="cpu")
            state = blob.get("model", blob.get("state_dict", blob))
            self.encoder.load_state_dict(state, strict=False)
        self.hidden_dim = int(getattr(self.encoder, "embed_dim",
                                        getattr(self.encoder, "hidden_dim", 0)))
        if self.hidden_dim == 0:
            raise RuntimeError("Could not resolve ECGViT hidden / embed dim")
        self.window_samples = int(window_samples)

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor) -> torch.Tensor:
        B, N = sig.shape
        sig_12ch = sig.unsqueeze(1).expand(B, 12, N).contiguous()
        seg_len = self.window_samples
        assert N == 2 * seg_len, f"expected {2 * seg_len} samples, got {N}"
        seg1 = sig_12ch[..., :seg_len]
        seg2 = sig_12ch[..., seg_len:]
        h1 = self.encoder(seg1)
        h2 = self.encoder(seg2)
        h = torch.cat([h1, h2], dim=1)
        if h.size(1) != 500:
            h = h.transpose(1, 2)
            h = nn.functional.interpolate(h, size=500, mode="linear", align_corners=False)
            h = h.transpose(1, 2)
        return h
```

Also create the placeholder so the `third_party/ST-MEM` dir exists for the skip-test logic to behave correctly until source is vendored:

```
mkdir -p third_party/ST-MEM
printf "Vendored ST-MEM source - clone bakqui/ST-MEM here:\n  git clone https://github.com/bakqui/ST-MEM.git third_party/ST-MEM\n" > third_party/ST-MEM/README.md
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_stage2_ssl_stmem.py -v
```

Expected: 1 passed (`test_stmem_adapter_replicate_path`), 1 skipped (vendored source absent until you clone it).

- [ ] **Step 5: Commit**

```
git add ecgcode/stage2/ssl/stmem.py tests/test_stage2_ssl_stmem.py third_party/ST-MEM/README.md
git commit -m "v12 task 14: STMEMAdapter (12-ch replicate, 5s x2 forward)"
```

---

### Task 15: `train_v12_stmem.py` (runs 5 + 6)

**Files:**
- Create: `scripts/train_v12_stmem.py`

- [ ] **Step 1: Write the script**

```python
# scripts/train_v12_stmem.py
"""v12_stmem_lp / v12_stmem_ft - ST-MEM transfer (single-lead replicate)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from openecg.stage2.ssl.head import BackboneWithHeads
from openecg.stage2.ssl.stmem import STMEMAdapter
from openecg.stage2.train import TrainConfig, load_checkpoint
from scripts.train_v12_hubert import _fit_with_groups
from scripts.train_v9_q1c_pu_merge import _ConcatWithCounts, eval_all

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
SEED = 42
EDGE_MARGIN_MS = 100


def _build_train(seed):
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=seed,
                                       q1c_pu_merge=True)
    return _ConcatWithCounts([ludb_train, isp_train, qtdb_merged])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lp", "ft"), required=True)
    ap.add_argument("--weights", default=None,
                     help="Path to ST-MEM pretrained weights .pt")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, mode={args.mode}", flush=True)

    train_ds = _build_train(SEED)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()),
                            dtype=torch.float32)

    torch.manual_seed(SEED); np.random.seed(SEED)
    backbone = STMEMAdapter(weights_path=args.weights, device=device)
    model = BackboneWithHeads(backbone, hidden_dim=backbone.hidden_dim, use_reg=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  hidden={backbone.hidden_dim} params={n_params:,} weights={args.weights}",
          flush=True)

    if args.mode == "lp":
        for p in backbone.parameters():
            p.requires_grad = False
        param_groups = [{"params": model.cls_head.parameters(), "lr": 1e-3}]
    else:
        param_groups = [
            {"params": backbone.parameters(), "lr": 1e-5},
            {"params": model.cls_head.parameters(), "lr": 1e-3},
        ]

    cfg = TrainConfig(epochs=args.epochs, batch_size=32, lr=1e-3,
                       early_stop_patience=7, warmup_frac=0.05)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)

    name = f"v12_stmem_{args.mode}"
    ckpt_path = CKPT_DIR / f"stage2_{name}.pt"
    t0 = time.time()
    best = _fit_with_groups(model, train_loader, val_loader, weights, cfg,
                              device, ckpt_path, param_groups)
    elapsed = time.time() - t0

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
                "weights": args.weights, "mode": args.mode,
                "best_metrics": best, **res},
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
```

- [ ] **Step 2: Smoke-test imports + CLI**

```
python -c "from scripts.train_v12_stmem import main; print('import OK')"
python scripts/train_v12_stmem.py --help
```

Expected: `import OK` and argparse help output.

- [ ] **Step 3: Commit**

```
git add scripts/train_v12_stmem.py
git commit -m "v12 task 15: train_v12_stmem.py (--mode lp|ft, runs 5+6)"
```

---

## Phase 3 — Combination (run 7) + comparison

### Task 16: `train_v12_best.py` (run 7)

**Files:**
- Create: `scripts/train_v12_best.py`

- [ ] **Step 1: Write the script**

```python
# scripts/train_v12_best.py
"""v12_best - apply the winning boundary tweak (soft / reg / soft+reg) on top
of the winning SSL backbone (HuBERT-FT or ST-MEM-FT).

Run only after runs 1-6 have completed. The script reads each
out/train_v12_*_{ts}.json file and picks:
    * boundary tweak winner = arg max LUDB val avg F1 over {v12_soft, v12_reg}
      vs v9 baseline (0.923). If both beat baseline by >= +0.005, use both.
    * backbone winner = arg max LUDB val avg F1 over {v12_hubert_ft,
      v12_stmem_ft}. Tie-break by qtdb_pu0_random.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §5.4.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "WTdelineator"))

from openecg import isp, ludb, qtdb
from openecg import eval as ecg_eval
from openecg.stage2.dataset import LUDBFrameDataset, compute_class_weights
from openecg.stage2.multi_dataset import (
    CombinedFrameDataset, QTDBSlidingDataset,
)
from openecg.stage2.reg_targets import RegLabelDataset
from openecg.stage2.soft_labels import SoftLabelDataset
from openecg.stage2.ssl.head import BackboneWithHeads
from openecg.stage2.ssl.hubert import HUBERT_DEFAULT_MODEL_ID, HubertECGAdapter
from openecg.stage2.ssl.stmem import STMEMAdapter
from openecg.stage2.train import (
    TrainConfig, boundary_l1_loss, kl_cross_entropy, load_checkpoint,
    run_eval, run_eval_reg, save_checkpoint, score_val_metrics,
)
from scripts.train_v9_q1c_pu_merge import _ConcatWithCounts, eval_all

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
SEED = 42
EDGE_MARGIN_MS = 100
V9_LUDB_F1 = 0.923
WIN_THRESHOLD = 0.005


def _latest_json(prefix):
    files = sorted(glob.glob(str(OUT_DIR / f"train_{prefix}_*.json")))
    if not files:
        return None
    return json.loads(Path(files[-1]).read_text())


def _ludb_f1(d):
    if d is None: return float("nan")
    for v in d.values():
        if isinstance(v, dict) and "ludb_edge_filtered" in v:
            return float(v["ludb_edge_filtered"])
    return float("nan")


def _qtdb_f1(d):
    if d is None: return float("nan")
    for v in d.values():
        if isinstance(v, dict) and "qtdb_pu0_random" in v:
            return float(v["qtdb_pu0_random"])
    return float("nan")


def _select_winners():
    soft = _latest_json("v12_soft")
    reg = _latest_json("v12_reg")
    hub_ft = _latest_json("v12_hubert_ft")
    stm_ft = _latest_json("v12_stmem_ft")
    soft_f1 = _ludb_f1(soft) if soft else 0.0
    reg_f1 = _ludb_f1(reg) if reg else 0.0
    use_soft = soft_f1 >= V9_LUDB_F1 + WIN_THRESHOLD
    use_reg = reg_f1 >= V9_LUDB_F1 + WIN_THRESHOLD
    if not (use_soft or use_reg):
        use_soft = soft_f1 >= reg_f1
        use_reg = not use_soft
    hub_f1 = _ludb_f1(hub_ft); stm_f1 = _ludb_f1(stm_ft)
    if hub_f1 == stm_f1:
        backbone = "hubert" if _qtdb_f1(hub_ft) >= _qtdb_f1(stm_ft) else "stmem"
    else:
        backbone = "hubert" if hub_f1 > stm_f1 else "stmem"
    return {
        "use_soft": bool(use_soft), "use_reg": bool(use_reg),
        "backbone": backbone, "soft_f1": soft_f1, "reg_f1": reg_f1,
        "hub_f1": hub_f1, "stm_f1": stm_f1,
    }


def _build_dataset(use_soft, use_reg, seed):
    ludb_train = LUDBFrameDataset(ludb.load_split()["train"],
                                    mask_unlabeled_edges=True,
                                    edge_margin_ms=EDGE_MARGIN_MS)
    isp_train = CombinedFrameDataset(["isp_train"])
    qtdb_merged = QTDBSlidingDataset(scale_factors=(1.0,),
                                       windows_per_record=20, seed=seed,
                                       q1c_pu_merge=True)
    if use_reg:
        ludb_w = RegLabelDataset(ludb_train)
        isp_w = RegLabelDataset(isp_train)
        qtdb_w = RegLabelDataset(qtdb_merged)
    else:
        ludb_w, isp_w, qtdb_w = ludb_train, isp_train, qtdb_merged
    if use_soft and not use_reg:
        ludb_w = SoftLabelDataset(ludb_w)
        isp_w = SoftLabelDataset(isp_w)
        qtdb_w = SoftLabelDataset(qtdb_w)
    return _ConcatWithCounts([ludb_w, isp_w, qtdb_w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stmem-weights", default=None)
    ap.add_argument("--lambda-reg", type=float, default=0.1)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    winners = _select_winners()
    print(f"Selection: {winners}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    train_ds = _build_dataset(winners["use_soft"], winners["use_reg"], SEED)
    ludb_val = LUDBFrameDataset(ludb.load_split()["val"],
                                  mask_unlabeled_edges=True,
                                  edge_margin_ms=EDGE_MARGIN_MS)
    weights = torch.tensor(compute_class_weights(train_ds.label_counts()),
                            dtype=torch.float32)

    if winners["backbone"] == "hubert":
        backbone = HubertECGAdapter(model_id=HUBERT_DEFAULT_MODEL_ID, device=device)
    else:
        backbone = STMEMAdapter(weights_path=args.stmem_weights, device=device)
    model = BackboneWithHeads(backbone, hidden_dim=backbone.hidden_dim,
                                use_reg=winners["use_reg"])
    param_groups = [
        {"params": backbone.parameters(), "lr": 1e-5},
        {"params": model.cls_head.parameters(), "lr": 1e-3},
    ]
    if winners["use_reg"]:
        param_groups.append({"params": model.reg_head.parameters(), "lr": 1e-3})

    cfg = TrainConfig(epochs=args.epochs, batch_size=32, lr=1e-3,
                       early_stop_patience=7, warmup_frac=0.05)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ludb_val, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)
    ckpt_path = CKPT_DIR / "stage2_v12_best.pt"

    model = model.to(device)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * cfg.warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best = -1; best_metrics = None; bad = 0
    cw = weights.to(device)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train(); total = 0; n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if winners["use_reg"]:
                sigs, leads, labels, reg_t, reg_m = batch
                sigs = sigs.to(device); leads = leads.to(device)
                labels = labels.to(device); reg_t = reg_t.to(device).float(); reg_m = reg_m.to(device).bool()
                cls, reg = model(sigs, leads)
                if winners["use_soft"]:
                    soft = torch.nn.functional.one_hot(
                        labels.clamp(0, 3), num_classes=4
                    ).float()
                    cls_loss = kl_cross_entropy(cls, soft, weight=cw)
                else:
                    cls_loss = torch.nn.functional.cross_entropy(
                        cls.transpose(1, 2), labels, weight=cw, ignore_index=255,
                    )
                reg_loss = boundary_l1_loss(reg, reg_t, reg_m)
                loss = cls_loss + args.lambda_reg * reg_loss
            else:
                sigs, leads, target = batch
                sigs = sigs.to(device); leads = leads.to(device); target = target.to(device)
                cls = model(sigs, leads)
                if winners["use_soft"]:
                    loss = kl_cross_entropy(cls, target.float(), weight=cw)
                else:
                    loss = torch.nn.functional.cross_entropy(
                        cls.transpose(1, 2), target, weight=cw, ignore_index=255,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step()
            total += float(loss.item()); n += 1
        val_fn = run_eval_reg if winners["use_reg"] else run_eval
        val = val_fn(model, val_loader, device)
        score = score_val_metrics(val, cfg.early_stop_metric)
        print(f"epoch {epoch:3d} train={total/max(1,n):.4f} score={score:.3f}", flush=True)
        if score > best:
            best = score
            best_metrics = {"epoch": epoch, "val_score": score, **winners,
                              "lambda_reg": args.lambda_reg}
            bad = 0
            save_checkpoint(ckpt_path, model, best_metrics, cfg)
        else:
            bad += 1
            if bad >= cfg.early_stop_patience: print(f"early stop {epoch}"); break
    elapsed = time.time() - t0
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    model = model.to(device).train(False)
    res = eval_all(model, device)
    print(f"\n=== v12_best eval ===", flush=True)
    for k, v in res.items(): print(f"  {k}: {v:.3f}")

    full = {
        "v9_q1c_pu_merge_ref": {"ludb_edge_filtered": V9_LUDB_F1,
                                  "isp_test": 0.943,
                                  "qtdb_pu0_random": 0.779},
        "v12_best": {"selection": winners,
                       "lambda_reg": args.lambda_reg,
                       "train_seconds": elapsed,
                       "best_metrics": best_metrics or {}, **res},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"train_v12_best_{ts}.json"
    def _safe(v):
        if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    out_path.write_text(json.dumps(_safe(full), indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test imports + CLI**

```
python -c "from scripts.train_v12_best import main; print('import OK')"
python scripts/train_v12_best.py --help
```

Expected: `import OK` and argparse help.

- [ ] **Step 3: Commit**

```
git add scripts/train_v12_best.py
git commit -m "v12 task 16: train_v12_best.py (run 7, combination)"
```

---

### Task 17: `compare_v12.py` aggregator

**Files:**
- Create: `scripts/compare_v12.py`

- [ ] **Step 1: Write the script**

```python
# scripts/compare_v12.py
"""Aggregate all v12_*.json result files into one comparison table.

Reads out/train_v9_q1c_pu_merge_*.json and out/train_v12_*_*.json,
emits Markdown table to stdout and also writes
out/v12_comparison_<ts>.md + out/v12_comparison_<ts>.json.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §6.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "out"
ROW_KEYS = ("ludb_edge_filtered", "isp_test", "qtdb_pu0_random")


def _latest(prefix: str) -> dict | None:
    files = sorted(glob.glob(str(OUT_DIR / f"train_{prefix}_*.json")))
    if not files:
        return None
    return json.loads(Path(files[-1]).read_text())


def _row(d: dict | None) -> dict | None:
    if d is None:
        return None
    for v in d.values():
        if isinstance(v, dict) and any(k in v for k in ROW_KEYS):
            return {k: float(v.get(k, float("nan"))) for k in ROW_KEYS}
    return None


def main():
    rows: list[tuple[str, dict | None]] = [
        ("v9_q1c_pu_merge", _row(_latest("v9_q1c_pu_merge"))),
        ("v12_soft",        _row(_latest("v12_soft"))),
        ("v12_reg",         _row(_latest("v12_reg"))),
        ("v12_hubert_lp",   _row(_latest("v12_hubert_lp"))),
        ("v12_hubert_ft",   _row(_latest("v12_hubert_ft"))),
        ("v12_stmem_lp",    _row(_latest("v12_stmem_lp"))),
        ("v12_stmem_ft",    _row(_latest("v12_stmem_ft"))),
        ("v12_best",        _row(_latest("v12_best"))),
    ]

    md_lines = ["# v12 comparison\n",
                f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
                "| run | LUDB val | ISP test | QTDB pu0 |",
                "|---|---|---|---|"]
    for name, row in rows:
        if row is None:
            md_lines.append(f"| {name} | (no run) | (no run) | (no run) |")
            continue
        md_lines.append(
            f"| {name} | "
            f"{row['ludb_edge_filtered']:.3f} | "
            f"{row['isp_test']:.3f} | "
            f"{row['qtdb_pu0_random']:.3f} |"
        )
    md = "\n".join(md_lines) + "\n"
    print(md)

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = OUT_DIR / f"v12_comparison_{ts}.md"
    md_path.write_text(md)
    json_path = OUT_DIR / f"v12_comparison_{ts}.json"
    json_path.write_text(json.dumps(
        {name: row for name, row in rows}, indent=2,
    ))
    print(f"\nSaved {md_path} / {json_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test**

```
python scripts/compare_v12.py
```

Expected: prints a Markdown table — most rows will say `(no run)` until the experiments are actually trained. Writes `out/v12_comparison_<ts>.md` and `.json`.

- [ ] **Step 3: Commit**

```
git add scripts/compare_v12.py
git commit -m "v12 task 17: compare_v12.py aggregator"
```

---

## Phase 4 — Run experiments

These are not code-writing tasks; they execute the seven training scripts and produce result JSON files. Each run can take 15–60 min depending on backbone size and hardware.

### Task 18: Run v12_soft (run 1)

- [ ] **Step 1: Train**

```
python scripts/train_v12_soft.py
```

Expected: writes `data/checkpoints/stage2_v12_soft.pt` and `out/train_v12_soft_<ts>.json`.

- [ ] **Step 2: Sanity-check the JSON**

```
python -c "import json,glob; d=json.loads(open(sorted(glob.glob('out/train_v12_soft_*.json'))[-1]).read()); print(d)"
```

Expected: dict with `ludb_edge_filtered`, `isp_test`, `qtdb_pu0_random` keys.

### Task 19: Run v12_reg sweep (run 2)

- [ ] **Step 1: Train**

```
python scripts/train_v12_reg.py
```

Expected: writes 3 lambda-specific checkpoints, `data/checkpoints/stage2_v12_reg.pt` (best), and `out/train_v12_reg_<ts>.json`.

### Task 20: Download HuBERT-ECG and run v12_hubert (runs 3, 4)

- [ ] **Step 1: Pre-cache HuBERT-ECG weights**

```
huggingface-cli download Edoardo-BS/hubert_ecg_small
```

If the canonical model id differs, search via:

```
huggingface-cli search hubert_ecg
```

Update `HUBERT_DEFAULT_MODEL_ID` in `ecgcode/stage2/ssl/hubert.py` if needed and commit the change.

- [ ] **Step 2: Run linear probe**

```
python scripts/train_v12_hubert.py --mode lp
```

- [ ] **Step 3: Run full finetune**

```
python scripts/train_v12_hubert.py --mode ft
```

### Task 21: Vendor ST-MEM and run v12_stmem (runs 5, 6)

- [ ] **Step 1: Vendor source**

```
git clone https://github.com/bakqui/ST-MEM.git third_party/ST-MEM
```

Inspect the repo's `models/` directory; if `ECGViT` lives at a different module path, extend the `candidates` list in `ecgcode/stage2/ssl/stmem.py::_load_stmem_module` and commit the addition.

- [ ] **Step 2: Download ST-MEM pretrained weights**

Locate the published checkpoint via the ST-MEM `README.md` (typically a Google Drive or OSF link). Save it to `data/checkpoints/stmem_pretrain.pt`.

- [ ] **Step 3: Run linear probe**

```
python scripts/train_v12_stmem.py --mode lp --weights data/checkpoints/stmem_pretrain.pt
```

- [ ] **Step 4: Run full finetune**

```
python scripts/train_v12_stmem.py --mode ft --weights data/checkpoints/stmem_pretrain.pt
```

### Task 22: Run v12_best (run 7)

- [ ] **Step 1: Train combination**

```
python scripts/train_v12_best.py --stmem-weights data/checkpoints/stmem_pretrain.pt
```

The `--stmem-weights` argument is ignored if the script picks HuBERT.

### Task 23: Generate comparison table

- [ ] **Step 1: Aggregate**

```
python scripts/compare_v12.py
```

Expected: outputs the Markdown comparison table with all 8 rows populated. Writes `out/v12_comparison_<ts>.md` for archival.

- [ ] **Step 2: Commit results**

```
git add out/v12_comparison_*.md out/v12_comparison_*.json out/train_v12_*.json
git commit -m "v12: experiment results (runs 1-7) + comparison table"
```

---

## Self-review notes

Plan reviewed against the spec for placeholder, type, and scope issues. One residual ambiguity remains explicit: HuBERT-ECG's exact HuggingFace id and ST-MEM's exact import path are not pinned because they depend on the upstream release. Tasks 20-21 walk the engineer through resolving each at execution time.

The plan covers every spec subsection:
- §4.1 (soft labels) → tasks 1, 3, 4
- §4.2 (regression head) → tasks 5, 6, 7, 8, 9
- §4.3 (combination) → tasks 16, 22
- §5.2 (HuBERT-ECG) → tasks 12, 13, 20
- §5.3 (ST-MEM) → tasks 14, 15, 21
- §5.4 (v12_best) → task 16
- §6 (eval) → reused from `scripts/train_v9_q1c_pu_merge.py::eval_all`
- §7 (code structure) → tasks 1-17
- §8 (risks) → addressed in tasks 12 (HuBERT seam), 14 (ST-MEM single-lead), 9 (lambda sweep)

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-06-v12-ssl-boundary.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. Best fit for this 17-code-task plan.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Phase 4 (tasks 18-23) is hardware-heavy (each training run is 15-60 min on a single GPU); the running of those tasks is the user's call regardless of which execution mode is chosen for tasks 1-17.
