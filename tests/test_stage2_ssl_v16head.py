"""Sanity checks for the ECGFMV16Head hybrid (ECG-FM backbone + v16 head)."""
from pathlib import Path

import pytest
import torch

from openecg.stage2.ssl.ecgfm import ECGFM_DEFAULT_CKPT
from openecg.stage2.ssl.v16head import ECGFMV16Head


@pytest.mark.skipif(not ECGFM_DEFAULT_CKPT.exists(),
                    reason=f"checkpoint not found at {ECGFM_DEFAULT_CKPT}")
def test_v16head_forward_with_weights():
    model = ECGFMV16Head(weights_path=ECGFM_DEFAULT_CKPT)
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    cls, reg, aux = model(sig, lead)
    assert cls.shape == (2, 500, 4)
    assert reg.shape == (2, 500, 6)
    assert aux.shape == (2, 500, 2)
    assert torch.isfinite(cls).all() and torch.isfinite(reg).all() and torch.isfinite(aux).all()


def test_v16head_forward_no_weights():
    model = ECGFMV16Head(weights_path=Path("/nonexistent.pt"))
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    cls, reg, aux = model(sig, lead)
    assert cls.shape == (2, 500, 4)
    assert reg.shape == (2, 500, 6)
    assert aux.shape == (2, 500, 2)


def test_v16head_freeze_backbone():
    model = ECGFMV16Head(weights_path=Path("/nonexistent.pt"))
    model.freeze_backbone()
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_back = sum(p.numel() for p in model.backbone.parameters())
    assert n_train == n_total - n_back
    assert n_train > 0
