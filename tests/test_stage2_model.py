import torch

from openecg.stage2.model import FrameClassifier


def test_forward_shape_cpu():
    model = FrameClassifier()
    x = torch.randn(4, 2500)
    lead_id = torch.zeros(4, dtype=torch.long)
    out = model(x, lead_id)
    assert out.shape == (4, 500, 4)


def test_param_count_under_500k():
    model = FrameClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    assert 200_000 < n_params < 500_000, f"got {n_params} params"


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
    model = FrameClassifier()
    model.eval()
    x = torch.randn(1, 2500)
    out_lead0 = model(x, torch.tensor([0]))
    out_lead5 = model(x, torch.tensor([5]))
    assert not torch.allclose(out_lead0, out_lead5)


def test_softmax_sums_to_1():
    model = FrameClassifier()
    model.eval()
    x = torch.randn(2, 2500)
    lead_id = torch.zeros(2, dtype=torch.long)
    logits = model(x, lead_id)
    probs = logits.softmax(dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


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
