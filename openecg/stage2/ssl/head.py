"""Shared classification + regression heads for SSL backbones."""

from torch import nn


class FrameHead(nn.Module):
    """Per-frame supercategory classifier head: Linear(d -> n_classes)."""

    def __init__(self, d_model: int, n_classes: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


class FrameRegHead(nn.Module):
    """Per-frame boundary-offset regressor: Linear(d -> n_reg)."""

    def __init__(self, d_model: int, n_reg: int = 6, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, n_reg)

    def forward(self, h):
        return self.linear(self.dropout(h))


class BackboneWithHeads(nn.Module):
    """Generic wrapper: backbone -> features [B, T, d] -> cls(+reg) head(s).

    Backbone must implement forward(sig, lead_id) -> features [B, 500, d].
    Returns cls_logits if use_reg=False, else (cls_logits, reg_offsets).
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int,
                 use_reg: bool = False, n_classes: int = 4, n_reg: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.use_reg = bool(use_reg)
        self.cls_head = FrameHead(hidden_dim, n_classes=n_classes, dropout=dropout)
        self.reg_head = FrameRegHead(hidden_dim, n_reg=n_reg, dropout=dropout) if use_reg else None
        self.model_config = {
            "arch": "ssl_backbone",
            "hidden_dim": hidden_dim,
            "n_classes": n_classes,
            "use_reg": self.use_reg,
            "n_reg": n_reg if use_reg else 0,
        }

    def forward(self, sig, lead_id):
        h = self.backbone(sig, lead_id)
        cls = self.cls_head(h)
        if self.use_reg:
            return cls, self.reg_head(h)
        return cls
