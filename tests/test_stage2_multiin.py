"""Smoke tests for FrameClassifierViTRegMultiIn."""
import torch

from openecg.stage2.model import (
    FrameClassifierViTReg, FrameClassifierViTRegMultiIn,
)


KW = dict(
    patch_size=5, d_model=128, n_heads=4, n_layers=4, ff=256,
    dropout=0.1, use_lead_emb=False, pos_type="learnable", conv_stem=False,
)


def test_single_channel_matches_parent_shape():
    m = FrameClassifierViTRegMultiIn(n_input_channels=1, **KW, n_reg=6)
    x = torch.randn(2, 1, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    cls, reg = m(x, lead)
    assert cls.shape == (2, 500, 4)
    assert reg.shape == (2, 500, 6)


def test_two_channels_forward():
    m = FrameClassifierViTRegMultiIn(n_input_channels=2, **KW, n_reg=6)
    x = torch.randn(3, 2, 2500)
    lead = torch.zeros(3, dtype=torch.long)
    cls, reg = m(x, lead)
    assert cls.shape == (3, 500, 4)
    assert reg.shape == (3, 500, 6)


def test_three_channels_forward():
    m = FrameClassifierViTRegMultiIn(n_input_channels=3, **KW, n_reg=6)
    x = torch.randn(2, 3, 2500)
    lead = torch.zeros(2, dtype=torch.long)
    cls, reg = m(x, lead)
    assert cls.shape == (2, 500, 4)
    assert reg.shape == (2, 500, 6)


def test_two_channel_backward_compat_2d_rejected():
    m = FrameClassifierViTRegMultiIn(n_input_channels=2, **KW, n_reg=6)
    x = torch.randn(2, 2500)                                 # 2D, ambiguous
    lead = torch.zeros(2, dtype=torch.long)
    try:
        m(x, lead)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_model_config_carries_arch_and_channels():
    m = FrameClassifierViTRegMultiIn(n_input_channels=2, **KW, n_reg=6)
    assert m.model_config["arch"] == "vit_reg_multiin"
    assert m.model_config["n_input_channels"] == 2
