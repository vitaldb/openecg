import os
import pytest
import torch
import torch.nn as nn

from ecgcode.stage2.ssl.stmem import STMEMAdapter


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
    nn.Module.__init__(adapter)
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
