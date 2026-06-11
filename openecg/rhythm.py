"""Rhythm classifier — small CNN router that predicts one of 6 rhythm
buckets per 10-s ECG window. Used as the L1 readout's ``rhythm`` field.

Architecture (from research/scripts/train_moe_router.py):
    Conv1d stem (15-tap stride 2 → 7-tap stride 2, d=64) →
    3× residual blocks (Conv1d 5-tap ×2 + BatchNorm) →
    AdaptiveAvgPool1d → Linear → 6 logits.

Classes (matches openecg.stage2.arrhythmia_classes.CLASS_TO_EXPERT):
    0  sinus       — NSR + bradycardia + tachycardia
    1  avb         — AVB1 + AVB high-grade
    2  paced       — paced rhythm
    3  afib        — AFib + AFlutter + SVT (narrow-QRS supraventricular)
    4  bbb         — RBBB + LBBB (wide-QRS conducted)
    5  ventricular — VPC + VT + VF + TdP (wide-QRS ectopic)

Checkpoint: ``data/checkpoints/moe_router_v2_6class.pt`` (555 KB).

This module reads PyTorch (torch is an optional dep). For TFLite-only
deployment, the router has not yet been exported — call sites that need it
on edge must add torch to their runtime, or wait for a future TFLite export.

Public API:
    >>> from openecg.rhythm import classify
    >>> label, probs = classify(signal_250hz)
    >>> label   # 'sinus' | 'avb' | 'paced' | 'afib' | 'bbb' | 'ventricular'

Input contract: 1-D float array, 2500 samples @ 250 Hz (10 s). The signal is
rank-normalized internally to match the training contract.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from openecg.dsp import rank_normalize

WINDOW_SAMPLES = 2500
CLASS_NAMES = ("sinus", "avb", "paced", "afib", "bbb", "ventricular")
N_CLASSES = len(CLASS_NAMES)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "openecg.rhythm.classify requires torch. Install with "
            "`pip install torch` or `pip install openecg[rhythm]`."
        ) from e
    return torch, nn


def _build_router(d: int = 64, n_experts: int = N_CLASSES):
    """Re-build the RhythmRouter architecture (mirrors research/.../train_moe_router.py)."""
    torch, nn = _require_torch()

    def block(c):
        return nn.Sequential(
            nn.Conv1d(c, c, 5, padding=2), nn.GELU(),
            nn.BatchNorm1d(c),
            nn.Conv1d(c, c, 5, padding=2), nn.GELU(),
        )

    class _Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(1, d // 2, kernel_size=15, stride=2, padding=7),
                nn.GELU(),
                nn.Conv1d(d // 2, d, kernel_size=7, stride=2, padding=3),
                nn.GELU(),
            )
            self.b1 = block(d)
            self.b2 = block(d)
            self.b3 = block(d)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(d, n_experts)

        def forward(self, sig):
            x = sig.unsqueeze(1)
            x = self.stem(x)
            x = self.b1(x) + x
            x = self.b2(x) + x
            x = self.b3(x) + x
            x = self.pool(x).squeeze(-1)
            return self.head(x)

    return _Router()


def _bundled_ckpt_path() -> Path:
    """Return path to the bundled checkpoint, raising clearly if missing."""
    p = (Path(__file__).resolve().parent.parent
         / "data" / "checkpoints" / "moe_router_v2_6class.pt")
    if not p.exists():
        raise FileNotFoundError(
            f"Bundled rhythm-router checkpoint not found at {p}. "
            f"This file ships with the repo at data/checkpoints/; "
            f"if you installed openecg via pip, the router weights are "
            f"not yet bundled in the wheel.")
    return p


@lru_cache(maxsize=1)
def _load_router():
    torch, _ = _require_torch()
    blob = torch.load(_bundled_ckpt_path(), map_location="cpu",
                      weights_only=False)
    state = blob.get("model_state", blob) if isinstance(blob, dict) else blob
    model = _build_router()
    model.load_state_dict(state)
    model.eval()
    return model


def classify(signal_250hz: np.ndarray) -> tuple[str, np.ndarray]:
    """Predict rhythm class for one 10-s window.

    Parameters
    ----------
    signal_250hz : 1-D float array, 2500 samples at 250 Hz. Will be
        rank-normalized internally.

    Returns
    -------
    label : one of the 6 class names (str).
    probs : (6,) float32 array of softmax probabilities, indexed by
        :data:`CLASS_NAMES`.
    """
    torch, _ = _require_torch()
    sig = np.asarray(signal_250hz, dtype=np.float32).ravel()
    if sig.size != WINDOW_SAMPLES:
        raise ValueError(
            f"rhythm.classify expects {WINDOW_SAMPLES} samples "
            f"(10 s @ 250 Hz), got {sig.size}")
    x = rank_normalize(sig)
    model = _load_router()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).unsqueeze(0))   # (1, 6)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs.astype(np.float32)


__all__ = ["classify", "CLASS_NAMES", "N_CLASSES", "WINDOW_SAMPLES"]
