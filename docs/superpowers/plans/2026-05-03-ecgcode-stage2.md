# ECGCode Stage 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a per-frame 4-class classifier (other/P/QRS/T) on LUDB cardiologist labels and beat Stage 1 baseline + NK direct on val split.

**Architecture:** PyTorch Conv (2 layers) + Transformer (4 layers, d=64) + Linear head. Single-lead with lead_id embedding. ~330K params. Input: 250Hz × 2500 samples (10s). Output: 50Hz × 500 frames × 4 classes.

**Tech Stack:** Python 3.11+, uv, PyTorch 2.6 + CUDA 12.4 (RTX 4090 24GB), numpy, scipy, pytest. Spec: `docs/superpowers/specs/2026-05-03-ecgcode-stage2-design.md`.

**Environment**:
```powershell
$env:UV_LINK_MODE = "copy"
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
```

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | add torch dep |
| `ecgcode/codec.py` | add `from_frames` (closes Stage 1 gap) |
| `ecgcode/stage2/__init__.py` | package marker |
| `ecgcode/stage2/dataset.py` | `LUDBFrameDataset` PyTorch Dataset + `compute_class_weights` |
| `ecgcode/stage2/model.py` | `FrameClassifier` Conv+Transformer |
| `ecgcode/stage2/train.py` | Training loop + checkpointing |
| `ecgcode/stage2/infer.py` | Checkpoint to per-frame predictions for a batch of records |
| `scripts/train_stage2.py` | CLI: train end-to-end + save checkpoint |
| `scripts/validate_stage2.py` | Validate checkpoint on val + comparison vs NK direct |
| `tests/test_codec.py` | Add `from_frames` round-trip tests |
| `tests/test_stage2_dataset.py` | Dataset shapes + class weights |
| `tests/test_stage2_model.py` | Forward pass shape + param count + GPU/CPU |
| `tests/test_stage2_train.py` | 1-epoch synthetic loss-decrease + checkpoint round-trip |
| `data/checkpoints/stage2_v1.pt` | trained checkpoint (gitignored - large) |

---

### Task 1: Add PyTorch dependency + GPU smoke

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add torch dep**

Edit `pyproject.toml` `[project.dependencies]` block, add `"torch>=2.6"` to the dependencies list. Final block:

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "neurokit2>=0.2.7",
    "wfdb>=4.1",
    "torch>=2.6",
]
```

- [ ] **Step 2: uv sync**

Run: `$env:UV_LINK_MODE = "copy"; uv sync`
Expected: torch 2.6+ installed.

- [ ] **Step 3: GPU smoke**

```powershell
uv run python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda: {torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA required'
x = torch.randn(2, 32, device='cuda')
print(f'tensor on {x.device}, shape {x.shape}')
"
```
Expected: `torch: 2.6.x+cu124`, `cuda: True`, tensor on cuda:0.

- [ ] **Step 4: Run full test suite (no regressions)**

```powershell
uv run pytest -v
```
Expected: 53 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "Add torch>=2.6 dependency for Stage 2"
```

---

### Task 2: Implement codec.from_frames

**Files:**
- Modify: `ecgcode/codec.py` (append function)
- Modify: `tests/test_codec.py` (append tests)

This closes the gap from Stage 1 Task 3. Stage 2 inference needs to convert per-frame model output back to RLE events for boundary extraction.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_codec.py`:

```python
def test_from_frames_basic():
    frames = np.array([vocab.ID_ISO]*5 + [vocab.ID_P]*5, dtype=np.uint8)
    events = codec.from_frames(frames, frame_ms=20)
    assert events == [(vocab.ID_ISO, 100), (vocab.ID_P, 100)]


def test_from_frames_single_frame_event():
    frames = np.array([vocab.ID_ISO, vocab.ID_PACER, vocab.ID_ISO], dtype=np.uint8)
    events = codec.from_frames(frames, frame_ms=20)
    assert events == [(vocab.ID_ISO, 20), (vocab.ID_PACER, 20), (vocab.ID_ISO, 20)]


def test_from_frames_roundtrip_with_to_frames():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 100), (vocab.ID_ISO, 80),
              (vocab.ID_R, 60), (vocab.ID_T, 200)]
    frames = codec.to_frames(events, frame_ms=20)
    recovered = codec.from_frames(frames, frame_ms=20)
    assert [s for s, _ in recovered] == [s for s, _ in events]
    assert sum(ms for _, ms in recovered) == sum(ms for _, ms in events)


def test_from_frames_empty():
    frames = np.array([], dtype=np.uint8)
    assert codec.from_frames(frames, frame_ms=20) == []
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_codec.py -v -k "from_frames"`
Expected: 4 tests FAIL with AttributeError.

- [ ] **Step 3: Implement from_frames**

Append to `ecgcode/codec.py`:

```python
def from_frames(frames: np.ndarray, frame_ms: int = 20) -> list[tuple[int, int]]:
    """Run-length encode per-frame array to list of (symbol_id, length_ms) events.

    Inverse of to_frames at frame granularity. Output durations are multiples of frame_ms.
    """
    if len(frames) == 0:
        return []
    change_idx = np.flatnonzero(np.diff(frames)) + 1
    boundaries = np.concatenate(([0], change_idx, [len(frames)]))
    events = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sym = int(frames[start])
        n_frames = end - start
        events.append((sym, n_frames * frame_ms))
    return events
```

- [ ] **Step 4: Run all codec tests**

Run: `uv run pytest tests/test_codec.py -v`
Expected: 19 passed (15 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add ecgcode/codec.py tests/test_codec.py
git commit -m "Add codec.from_frames (per-frame array to RLE events)"
```

---

### Task 3: Stage 2 Dataset class

**Files:**
- Create: `ecgcode/stage2/__init__.py`
- Create: `ecgcode/stage2/dataset.py`
- Create: `tests/test_stage2_dataset.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_stage2_dataset.py
import os
import numpy as np
import pytest
import torch

LUDB_AVAILABLE = bool(os.environ.get("ECGCODE_LUDB_ZIP"))
pytestmark = pytest.mark.skipif(not LUDB_AVAILABLE, reason="ECGCODE_LUDB_ZIP not set")


def test_dataset_basic_shapes():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1, 2])
    assert len(ds) > 0
    sig, lead_id, labels = ds[0]
    assert sig.dtype == torch.float32
    assert sig.shape == (2500,)
    assert lead_id.dtype == torch.long
    assert lead_id.shape == ()
    assert 0 <= int(lead_id) < 12
    assert labels.dtype == torch.long
    assert labels.shape == (500,)


def test_dataset_labels_in_supercategory_range():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1, 2])
    sig, lead_id, labels = ds[0]
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 3


def test_dataset_signal_normalized():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1, 2])
    sig, _, _ = ds[0]
    assert abs(float(sig.mean())) < 0.1
    assert abs(float(sig.std()) - 1.0) < 0.1


def test_dataset_covers_all_leads():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1])
    leads_seen = set()
    for i in range(len(ds)):
        _, lead_id, _ = ds[i]
        leads_seen.add(int(lead_id))
    assert leads_seen == set(range(12))


def test_compute_class_weights_inverse_sqrt():
    from ecgcode.stage2.dataset import compute_class_weights
    counts = np.array([600, 100, 100, 200], dtype=np.float64)
    weights = compute_class_weights(counts)
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]
    assert abs(weights.sum() - 4.0) < 1e-6
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_stage2_dataset.py -v`
Expected: 5 tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement stage2 package**

Create `ecgcode/stage2/__init__.py`:

```python
"""ECGCode Stage 2: per-frame supercategory classifier (P/QRS/T/other).

Spec: docs/superpowers/specs/2026-05-03-ecgcode-stage2-design.md
"""
```

Create `ecgcode/stage2/dataset.py`:

```python
# ecgcode/stage2/dataset.py
"""LUDB Stage 2 dataset: signal+lead_id to frame labels (4-class supercategory)."""

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from ecgcode import eval as ee
from ecgcode import ludb

FS_NATIVE = 500
FS_INPUT = 250
FRAME_MS = 20
N_CLASSES = 4


class LUDBFrameDataset(Dataset):
    """Eager-load LUDB train/val sequences. Memory: ~30 MB for 1908 sequences.

    __getitem__ returns: (signal[2500] float32, lead_id scalar long, labels[500] long).
    """

    def __init__(self, record_ids):
        self.items = []
        self.cache = {}

        for rid in record_ids:
            try:
                record = ludb.load_record(rid)
            except Exception:
                continue
            for lead_idx, lead in enumerate(ludb.LEADS_12):
                sig_500 = record[lead]
                sig_250 = scipy_signal.decimate(sig_500, 2, zero_phase=True)
                mean = float(sig_250.mean())
                std = float(sig_250.std()) + 1e-6
                sig_250 = ((sig_250 - mean) / std).astype(np.float32)
                try:
                    gt_ann = ludb.load_annotations(rid, lead)
                except Exception:
                    continue
                labels = ee.gt_to_super_frames(
                    gt_ann, n_samples=len(sig_500), fs=FS_NATIVE, frame_ms=FRAME_MS
                ).astype(np.int64)
                self.cache[(rid, lead)] = (sig_250, lead_idx, labels)
                self.items.append((rid, lead))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rid, lead = self.items[idx]
        sig, lead_idx, labels = self.cache[(rid, lead)]
        return (
            torch.from_numpy(sig),
            torch.tensor(lead_idx, dtype=torch.long),
            torch.from_numpy(labels),
        )

    def label_counts(self):
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for (_, _, labels) in self.cache.values():
            for c in range(N_CLASSES):
                counts[c] += int((labels == c).sum())
        return counts


def compute_class_weights(counts):
    """Soft inverse-sqrt class weights, normalized so sum == n_classes."""
    n = len(counts)
    weights = 1.0 / np.sqrt(counts + 1e-6)
    weights = weights / weights.sum() * n
    return weights.astype(np.float64)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_stage2_dataset.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ecgcode/stage2/__init__.py ecgcode/stage2/dataset.py tests/test_stage2_dataset.py
git commit -m "Add Stage 2 LUDBFrameDataset + class weights helper"
```

---

### Task 4: Stage 2 model (FrameClassifier)

**Files:**
- Create: `ecgcode/stage2/model.py`
- Create: `tests/test_stage2_model.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_stage2_model.py
import torch

from ecgcode.stage2.model import FrameClassifier


def test_forward_shape_cpu():
    model = FrameClassifier()
    x = torch.randn(4, 2500)
    lead_id = torch.zeros(4, dtype=torch.long)
    out = model(x, lead_id)
    assert out.shape == (4, 500, 4)


def test_param_count_under_500k():
    model = FrameClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    assert 200_000 < n_params < 500_000, f"got {n_params} params (target ~330K)"


def test_forward_gpu():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    model = FrameClassifier().cuda()
    x = torch.randn(8, 2500, device="cuda")
    lead_id = torch.randint(0, 12, (8,), device="cuda")
    out = model(x, lead_id)
    assert out.shape == (8, 500, 4)
    assert out.device.type == "cuda"


def test_lead_embedding_changes_output():
    torch.manual_seed(0)
    model = FrameClassifier().eval()
    x = torch.randn(1, 2500)
    out_lead0 = model(x, torch.tensor([0]))
    out_lead5 = model(x, torch.tensor([5]))
    assert not torch.allclose(out_lead0, out_lead5)


def test_softmax_sums_to_1():
    model = FrameClassifier().eval()
    x = torch.randn(2, 2500)
    lead_id = torch.zeros(2, dtype=torch.long)
    logits = model(x, lead_id)
    probs = logits.softmax(dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_stage2_model.py -v`
Expected: 5 tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement model.py**

```python
# ecgcode/stage2/model.py
"""Stage 2 FrameClassifier: Conv + Transformer + Linear -> per-frame 4-class logits."""

import torch
from torch import nn


class FrameClassifier(nn.Module):
    """Input: signal [B, 2500] @ 250Hz, lead_id [B] in {0..11}.
    Output: logits [B, 500, 4] (per-frame supercategory).
    """

    def __init__(
        self,
        n_leads=12,
        d_model=64,
        n_heads=4,
        n_layers=4,
        ff=256,
        n_classes=4,
        dropout=0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=5, padding=7)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2)
        self.lead_emb = nn.Embedding(n_leads, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, lead_id):
        h = torch.nn.functional.gelu(self.conv1(x.unsqueeze(1)))
        h = torch.nn.functional.gelu(self.conv2(h))
        h = h.transpose(1, 2)
        h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_stage2_model.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ecgcode/stage2/model.py tests/test_stage2_model.py
git commit -m "Add Stage 2 FrameClassifier (Conv + Transformer + lead embedding)"
```

---

### Task 5: Stage 2 training loop

**Files:**
- Create: `ecgcode/stage2/train.py`
- Create: `tests/test_stage2_train.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_stage2_train.py
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
    for _ in range(20):
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
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_stage2_train.py -v`
Expected: 2 tests FAIL.

- [ ] **Step 3: Implement train.py**

```python
# ecgcode/stage2/train.py
"""Stage 2 training loop with checkpointing and early stopping."""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecgcode import eval as ee


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


def train_one_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    weights = class_weights.to(device)
    total_loss = 0.0
    n_batches = 0
    for sigs, leads, labels in loader:
        sigs = sigs.to(device)
        leads = leads.to(device)
        labels = labels.to(device)
        logits = model(sigs, leads)
        loss = nn.functional.cross_entropy(
            logits.transpose(1, 2), labels, weight=weights
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
    return ee.frame_f1(pred_concat, true_concat)


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
        device="cuda", ckpt_path=None, log_fn=print):
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
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, device)
        for _ in range(len(train_loader)):
            scheduler.step()
        val_metrics = run_eval(model, val_loader, device)
        qrs_f1 = val_metrics[ee.SUPER_QRS]["f1"]
        log_fn(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_F1: P={val_metrics[ee.SUPER_P]['f1']:.3f} "
            f"QRS={qrs_f1:.3f} T={val_metrics[ee.SUPER_T]['f1']:.3f}"
        )
        if qrs_f1 > best_qrs:
            best_qrs = qrs_f1
            best_metrics = {
                "epoch": epoch,
                "val_qrs_f1": qrs_f1,
                "val_p_f1": val_metrics[ee.SUPER_P]["f1"],
                "val_t_f1": val_metrics[ee.SUPER_T]["f1"],
                "val_other_f1": val_metrics[ee.SUPER_OTHER]["f1"],
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_stage2_train.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ecgcode/stage2/train.py tests/test_stage2_train.py
git commit -m "Add Stage 2 training loop (cosine schedule, early stop, checkpoint)"
```

---

### Task 6: Stage 2 inference helper

**Files:**
- Create: `ecgcode/stage2/infer.py`

- [ ] **Step 1: Implement infer.py**

```python
# ecgcode/stage2/infer.py
"""Stage 2 inference: checkpoint to per-frame predictions for validation."""

import numpy as np
import torch

from ecgcode import codec
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import load_checkpoint


def load_model(ckpt_path, device="cuda"):
    model = FrameClassifier()
    load_checkpoint(ckpt_path, model)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def predict_frames(model, sig, lead_id, device="cuda"):
    """Single-sequence inference: signal[2500] to frame argmax [500] (uint8)."""
    x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
    lid = torch.tensor([lead_id], dtype=torch.long, device=device)
    logits = model(x, lid)
    pred = logits.argmax(dim=-1).cpu().numpy().squeeze(0).astype(np.uint8)
    return pred


def predict_to_events(model, sig, lead_id, device="cuda", frame_ms=20):
    """Single-sequence inference to RLE events (for boundary extraction)."""
    frames = predict_frames(model, sig, lead_id, device=device)
    return codec.from_frames(frames, frame_ms=frame_ms)
```

- [ ] **Step 2: Smoke import**

```powershell
$env:UV_LINK_MODE = "copy"
uv run python -c "from ecgcode.stage2.infer import load_model, predict_frames, predict_to_events; print('infer module imports OK')"
```

- [ ] **Step 3: Commit**

```bash
git add ecgcode/stage2/infer.py
git commit -m "Add Stage 2 inference helpers (load_model, predict_frames, predict_to_events)"
```

---

### Task 7: Train script (CLI) + actually train the model

**Files:**
- Create: `scripts/train_stage2.py`

- [ ] **Step 1: Implement train script**

```python
# scripts/train_stage2.py
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
```

- [ ] **Step 2: Run training**

```powershell
$env:UV_LINK_MODE = "copy"
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
uv run python scripts/train_stage2.py
```
Expected: ~10-30 min on RTX 4090. Console shows per-epoch loss + val F1.

If val F1 flat near 0.25, check class weights, try lr=5e-4 or 2e-3.

- [ ] **Step 3: Inspect checkpoint**

```powershell
uv run python -c "
import torch
ckpt = torch.load('data/checkpoints/stage2_v1.pt', map_location='cpu', weights_only=False)
print('Best metrics:', ckpt['metrics'])
print('Config:', ckpt['config'])
"
```

- [ ] **Step 4: Commit script**

```bash
git add scripts/train_stage2.py
git commit -m "Add train_stage2 script and produce v1 checkpoint"
```

---

### Task 8: Validate script (model vs cardiologist + model vs NK)

**Files:**
- Create: `scripts/validate_stage2.py`

- [ ] **Step 1: Implement validate script**

```python
# scripts/validate_stage2.py
"""Validate Stage 2 checkpoint on LUDB val split.

Reports model frame F1, boundary error, side-by-side vs NK direct, per-lead breakdown.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ecgcode import codec, delineate, eval as ee, labeler, ludb, pacer
from ecgcode.stage2.dataset import LUDBFrameDataset
from ecgcode.stage2.infer import load_model, predict_frames

CKPT_PATH = Path("data/checkpoints/stage2_v1.pt")
OUT_DIR = Path("out")
FS = 500
FRAME_MS = 20
BOUNDARY_TOLERANCES = {
    "p_on": 50, "p_off": 50,
    "qrs_on": 40, "qrs_off": 40,
    "t_on": 50, "t_off": 100,
}


def _extract_pred_boundaries(events, fs=FS):
    out = defaultdict(list)
    cum_samples = 0
    prev_super = ee.SUPER_OTHER
    super_to_name = {ee.SUPER_P: "p", ee.SUPER_QRS: "qrs", ee.SUPER_T: "t"}
    for sym, ms in events:
        n = round(ms * fs / 1000.0)
        cur_super = ee.to_supercategory(np.array([sym], dtype=np.uint8))[0]
        if cur_super != prev_super:
            if prev_super in super_to_name:
                out[f"{super_to_name[prev_super]}_off"].append(cum_samples - 1)
            if cur_super in super_to_name:
                out[f"{super_to_name[cur_super]}_on"].append(cum_samples)
        cum_samples += n
        prev_super = cur_super
    if prev_super in super_to_name:
        out[f"{super_to_name[prev_super]}_off"].append(cum_samples - 1)
    return dict(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {CKPT_PATH}...")
    model = load_model(CKPT_PATH, device=device)

    val_ids = ludb.load_split()["val"]
    print(f"Validating on {len(val_ids)} val records x 12 leads")

    print("Loading val dataset...")
    val_ds = LUDBFrameDataset(val_ids)
    print(f"  {len(val_ds)} sequences cached")

    model_pred_frames = []
    model_true_frames = []
    nk_pred_frames = []
    nk_true_frames = []
    boundary_pred_model = defaultdict(list)
    boundary_pred_nk = defaultdict(list)
    boundary_true = defaultdict(list)
    per_lead_pred = defaultdict(list)
    per_lead_true = defaultdict(list)

    t0 = time.time()
    cum_offset = 0
    for idx in range(len(val_ds)):
        rid, lead = val_ds.items[idx]
        sig_250, lead_idx, true_frames = val_ds.cache[(rid, lead)]

        pred_frames_model = predict_frames(model, sig_250, lead_idx, device=device)
        n_common = min(len(pred_frames_model), len(true_frames))
        model_pred_frames.append(pred_frames_model[:n_common])
        model_true_frames.append(true_frames[:n_common].astype(np.uint8))
        per_lead_pred[lead_idx].append(pred_frames_model[:n_common])
        per_lead_true[lead_idx].append(true_frames[:n_common].astype(np.uint8))

        events_model = codec.from_frames(pred_frames_model, frame_ms=FRAME_MS)
        b_model = _extract_pred_boundaries(events_model)
        for k, v in b_model.items():
            boundary_pred_model[k].extend(int(x) + cum_offset for x in v)

        record = ludb.load_record(rid)
        sig_500 = record[lead]
        dr = delineate.run(sig_500, fs=FS)
        spikes = pacer.detect_spikes(sig_500, fs=FS)
        events_nk = labeler.label(dr, spikes.tolist(), n_samples=len(sig_500), fs=FS)
        nk_pred_super = ee.events_to_super_frames(events_nk, len(sig_500), fs=FS, frame_ms=FRAME_MS)
        n_common_nk = min(len(nk_pred_super), len(true_frames))
        nk_pred_frames.append(nk_pred_super[:n_common_nk])
        nk_true_frames.append(true_frames[:n_common_nk].astype(np.uint8))
        b_nk = _extract_pred_boundaries(events_nk)
        for k, v in b_nk.items():
            boundary_pred_nk[k].extend(int(x) + cum_offset for x in v)

        try:
            gt_ann = ludb.load_annotations(rid, lead)
            for k, v in gt_ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    boundary_true[k].extend(int(x) + cum_offset for x in v)
        except Exception:
            pass

        cum_offset += len(sig_500)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(val_ds)}] {time.time()-t0:.1f}s")

    model_pred_concat = np.concatenate(model_pred_frames)
    model_true_concat = np.concatenate(model_true_frames)
    nk_pred_concat = np.concatenate(nk_pred_frames)
    nk_true_concat = np.concatenate(nk_true_frames)
    f1_model = ee.frame_f1(model_pred_concat, model_true_concat)
    f1_nk = ee.frame_f1(nk_pred_concat, nk_true_concat)

    boundary_metrics_model = {}
    boundary_metrics_nk = {}
    for key, tol in BOUNDARY_TOLERANCES.items():
        boundary_metrics_model[key] = ee.boundary_error(
            boundary_pred_model.get(key, []), boundary_true.get(key, []),
            tolerance_ms=tol, fs=FS,
        )
        boundary_metrics_nk[key] = ee.boundary_error(
            boundary_pred_nk.get(key, []), boundary_true.get(key, []),
            tolerance_ms=tol, fs=FS,
        )

    per_lead_f1 = {}
    for lid, preds in per_lead_pred.items():
        trues = per_lead_true[lid]
        p_concat = np.concatenate(preds)
        t_concat = np.concatenate(trues)
        per_lead_f1[lid] = ee.frame_f1(p_concat, t_concat)

    print("\n== Stage 2 v1.0 vs Stage 1 NK direct (LUDB val, supercategory F1) ==\n")
    print(f"{'Class':6s} | {'Model F1':>10s} | {'NK F1':>10s} | {'Delta':>7s}")
    for sc in (ee.SUPER_P, ee.SUPER_QRS, ee.SUPER_T):
        name = ee.SUPER_NAMES[sc]
        m = f1_model[sc]['f1']
        n = f1_nk[sc]['f1']
        print(f"{name:6s} | {m:10.3f} | {n:10.3f} | {m-n:+7.3f}")

    print("\nBoundary error: model / NK (median ms / sens / PPV)")
    for key in BOUNDARY_TOLERANCES:
        mm = boundary_metrics_model[key]
        nn = boundary_metrics_nk[key]
        print(f"  {key:7s} | model: {mm['median_error_ms']:5.1f}/{mm['sensitivity']:.2f}/{mm['ppv']:.2f}  "
              f"| NK: {nn['median_error_ms']:5.1f}/{nn['sensitivity']:.2f}/{nn['ppv']:.2f}")

    print("\nPer-lead F1 (Model):")
    print(f"{'Lead':5s} | {'P':>6s} | {'QRS':>6s} | {'T':>6s}")
    for lid in sorted(per_lead_f1.keys()):
        lead_name = ludb.LEADS_12[lid]
        f = per_lead_f1[lid]
        print(f"{lead_name:5s} | {f[ee.SUPER_P]['f1']:6.3f} | {f[ee.SUPER_QRS]['f1']:6.3f} | {f[ee.SUPER_T]['f1']:6.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"validation_stage2_{ts}.json"
    out_file.write_text(json.dumps({
        "model_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_model.items()},
        "nk_f1": {ee.SUPER_NAMES[sc]: m for sc, m in f1_nk.items()},
        "boundary_model": boundary_metrics_model,
        "boundary_nk": boundary_metrics_nk,
        "per_lead_f1": {ludb.LEADS_12[lid]: {ee.SUPER_NAMES[sc]: m for sc, m in f.items()}
                        for lid, f in per_lead_f1.items()},
        "n_records": len(val_ids),
    }, indent=2))
    print(f"\nReport: {out_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validation**

```powershell
$env:UV_LINK_MODE = "copy"
$env:ECGCODE_LUDB_ZIP = "G:\Shared drives\datasets\ecg\lobachevsky-university-electrocardiography-database-1.0.1.zip"
uv run python scripts/validate_stage2.py
```
Expected: ~5 min runtime. Console shows model vs NK side-by-side. JSON saved.

Acceptance check (per spec section 7):
- Model P F1 >= 0.80 AND >= NK 0.49
- Model QRS F1 >= 0.90 AND >= NK 0.67
- Model T F1 >= 0.75 AND >= NK 0.51
- All Delta >= 0 (model beats NK on every wave class)

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_stage2.py
git commit -m "Add validate_stage2 script (model vs NK comparison + per-lead F1)"
```

---

### Task 9: Final verification + README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full test suite**

```powershell
uv run pytest -v
```
Expected: All tests pass (Stage 1: 53 + Stage 2: 12 = 65 with env set).

- [ ] **Step 2: Update README**

Append a Stage 2 section to README.md with the actual numbers from `out/validation_stage2_*.json`. Format:

```markdown

### Stage 2 v1.0 metrics (Conv+Transformer trained on LUDB cardiologist labels, 41 val records)

| Metric | Model | NK direct (S1) | Target |
|---|---|---|---|
| P frame F1 | <fill in> | 0.49 | >= 0.80 |
| QRS frame F1 | <fill in> | 0.67 | >= 0.90 |
| T frame F1 | <fill in> | 0.51 | >= 0.75 |

Stage 2 spec: `docs/superpowers/specs/2026-05-03-ecgcode-stage2-design.md`. Train: `scripts/train_stage2.py`. Validate: `scripts/validate_stage2.py`. Checkpoint: `data/checkpoints/stage2_v1.pt` (gitignored).
```

Replace `<fill in>` with actual numbers.

Also append to the Setup section:

```markdown
uv run python scripts/train_stage2.py     # to data/checkpoints/stage2_v1.pt
uv run python scripts/validate_stage2.py  # to out/validation_stage2_*.json
```

- [ ] **Step 3: Final commit + push**

```bash
git add README.md
git commit -m "Stage 2 v1.0: trained model + validation results"
git push
```

## Self-review checklist

After implementing all tasks, verify spec acceptance criteria are met. If model misses targets:

- All metrics flat near random: verify class weights, single-batch overfit, raise lr
- Model F1 < NK F1 (any class): per-lead breakdown, lead embedding gradient check, try focal loss
- Train F1 high val F1 low: add dropout=0.2, gaussian noise augmentation
- Specific lead bad: lead-stratified batch sampler, longer training

If targets missed but design issues identified, document in `docs/superpowers/notes/` and decide: ship v1.0 or block.
