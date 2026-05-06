import numpy as np
import torch

from openecg.stage2.model import FrameClassifierViT
from openecg.stage2.train import (
    kl_cross_entropy, train_one_epoch_kl,
)


def test_kl_equals_ce_on_one_hot():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 4)
    hard = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(hard, num_classes=4).float()
    kl_loss = kl_cross_entropy(logits, onehot)
    ce_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), hard, reduction="mean",
    )
    assert torch.isclose(kl_loss, ce_loss, atol=1e-5)


def test_kl_skips_zero_target_rows():
    logits = torch.zeros(1, 3, 4)
    target = torch.zeros(1, 3, 4)
    target[0, 0, 1] = 1.0
    target[0, 2, 2] = 1.0
    loss = kl_cross_entropy(logits, target)
    expected = torch.log(torch.tensor(4.0))
    assert torch.isclose(loss, expected, atol=1e-5)


def test_train_one_epoch_kl_loss_decreases():
    torch.manual_seed(0)
    KW = dict(patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
              use_lead_emb=False, pos_type="learnable")
    model = FrameClassifierViT(**KW)
    sigs = torch.randn(4, 2500)
    leads = torch.zeros(4, dtype=torch.long)
    soft = torch.zeros(4, 500, 4)
    soft[..., 0] = 1.0
    loader = [(sigs, leads, soft)]
    weights = torch.ones(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        loss = train_one_epoch_kl(model, loader, opt, weights, device="cpu")
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.7


def test_boundary_l1_loss_masked():
    from openecg.stage2.train import boundary_l1_loss
    pred = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    target = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, False, False, False, False, False]]])
    loss = boundary_l1_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(2.0))


def test_train_one_epoch_reg_loss_decreases():
    from openecg.stage2.model import FrameClassifierViTReg
    from openecg.stage2.train import train_one_epoch_reg
    torch.manual_seed(0)
    model = FrameClassifierViTReg(
        patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
        use_lead_emb=False, pos_type="learnable",
    )
    sigs = torch.randn(4, 2500)
    leads = torch.zeros(4, dtype=torch.long)
    labels = torch.zeros(4, 500, dtype=torch.long)
    reg_t = torch.zeros(4, 500, 6)
    reg_m = torch.zeros(4, 500, 6, dtype=torch.bool)
    reg_m[:, 100, 0] = True
    reg_t[:, 100, 0] = 3.0
    loader = [(sigs, leads, labels, reg_t, reg_m)]
    weights = torch.ones(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        loss = train_one_epoch_reg(model, loader, opt, weights, device="cpu")
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.7
