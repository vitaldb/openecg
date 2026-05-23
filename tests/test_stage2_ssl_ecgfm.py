"""ECG-FM adapter sanity check."""
from pathlib import Path

import pytest
import torch

from openecg.stage2.ssl.ecgfm import ECGFM_DEFAULT_CKPT, ECGFMAdapter


@pytest.mark.skipif(not ECGFM_DEFAULT_CKPT.exists(),
                    reason=f"checkpoint not found at {ECGFM_DEFAULT_CKPT}")
def test_ecgfm_adapter_forward_shape_with_weights():
    adapter = ECGFMAdapter(weights_path=ECGFM_DEFAULT_CKPT, device="cpu")
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, ECGFMAdapter.HIDDEN_DIM)
    assert h.dtype == torch.float32
    assert torch.isfinite(h).all()


def test_ecgfm_adapter_no_weights():
    adapter = ECGFMAdapter(weights_path=Path("/nonexistent.pt"), device="cpu")
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, ECGFMAdapter.HIDDEN_DIM)
    assert torch.isfinite(h).all()
