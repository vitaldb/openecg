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
