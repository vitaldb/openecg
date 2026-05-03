"""Stage 2 FrameClassifier: Conv + Transformer + Linear -> per-frame 4-class logits."""

import torch
from torch import nn


class FrameClassifier(nn.Module):
    """Input: signal [B, 2500] @ 250Hz, lead_id [B] in {0..11}.
    Output: logits [B, 500, 4] (per-frame supercategory).
    """

    def __init__(
        self,
        n_leads=12,
        d_model=64,
        n_heads=4,
        n_layers=4,
        ff=256,
        n_classes=4,
        dropout=0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=5, padding=7)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2)
        self.lead_emb = nn.Embedding(n_leads, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, lead_id):
        h = torch.nn.functional.gelu(self.conv1(x.unsqueeze(1)))
        h = torch.nn.functional.gelu(self.conv2(h))
        h = h.transpose(1, 2)
        h = h + self.lead_emb(lead_id).unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)
