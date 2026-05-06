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

    # Plain class (not nn.Module) so we can attach it via attribute assignment
    # on a non-initialized adapter without tripping nn.Module's _modules guard.
    class _DummyEncoder:
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
    nn.Module.__init__(adapter)
    adapter.encoder = _DummyEncoder()
    adapter.hidden_dim = 8
    adapter.target_fs = 100
    adapter.window_seconds = 5
    sig = torch.randn(2, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    h = adapter(sig, lead)
    assert h.shape == (2, 500, 8)
