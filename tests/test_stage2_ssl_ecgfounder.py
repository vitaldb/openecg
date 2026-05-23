"""ECGFounder adapter sanity check. Loads the actual pretrained checkpoint
because the file is bundled with the dev environment (data/checkpoints/external).
"""
from pathlib import Path

import pytest
import torch

from openecg.stage2.ssl.ecgfounder import (
    ECGFOUNDER_DEFAULT_CKPT,
    ECGFounderAdapter,
)


@pytest.mark.skipif(not ECGFOUNDER_DEFAULT_CKPT.exists(),
                    reason=f"checkpoint not found at {ECGFOUNDER_DEFAULT_CKPT}")
def test_ecgfounder_adapter_forward_shape():
    adapter = ECGFounderAdapter(weights_path=ECGFOUNDER_DEFAULT_CKPT, device="cpu")
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, ECGFounderAdapter.HIDDEN_DIM)
    assert h.dtype == torch.float32
    assert torch.isfinite(h).all()


def test_ecgfounder_adapter_no_weights():
    """Architecture-only forward (skips weight load) for environments without ckpt."""
    adapter = ECGFounderAdapter(weights_path=Path("/nonexistent.pth"), device="cpu")
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, ECGFounderAdapter.HIDDEN_DIM)
    assert torch.isfinite(h).all()
