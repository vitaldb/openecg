"""ST-MEM adapter - wraps the spatiotemporal masked ECG ViT for our
single-lead 250 Hz / 10 s pipeline.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md sec 5.3.

Vendored source expected at third_party/ST-MEM/. We import lazily so the
package is loadable even when the vendored module is absent (tests skip).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


_VENDORED = Path(__file__).resolve().parents[3] / "third_party" / "ST-MEM"


def _load_stmem_module():
    """Insert vendored ST-MEM into sys.path and return the model class."""
    if str(_VENDORED) not in sys.path:
        sys.path.insert(0, str(_VENDORED))
    candidates = [
        ("models.stmem", "ECGViT"),
        ("stmem.models", "ECGViT"),
        ("models", "ECGViT"),
        ("st_mem.models", "ECGViT"),
    ]
    last_err = None
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Could not import ST-MEM ECGViT from {_VENDORED}. "
        f"Tried {candidates}. Last error: {last_err}"
    )


class STMEMAdapter(nn.Module):
    """ST-MEM single-lead adapter (single lead replicated across 12 channels)."""

    def __init__(self, weights_path: str | None = None,
                 device: str = "cpu",
                 window_samples: int = 1250):
        super().__init__()
        cls = _load_stmem_module()
        self.encoder = cls()
        if weights_path:
            blob = torch.load(weights_path, map_location="cpu")
            state = blob.get("model", blob.get("state_dict", blob))
            self.encoder.load_state_dict(state, strict=False)
        self.hidden_dim = int(getattr(self.encoder, "embed_dim",
                                      getattr(self.encoder, "hidden_dim", 0)))
        if self.hidden_dim == 0:
            raise RuntimeError("Could not resolve ECGViT hidden / embed dim")
        self.window_samples = int(window_samples)

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor) -> torch.Tensor:
        B, N = sig.shape
        sig_12ch = sig.unsqueeze(1).expand(B, 12, N).contiguous()
        seg_len = self.window_samples
        assert N == 2 * seg_len, f"expected {2 * seg_len} samples, got {N}"
        seg1 = sig_12ch[..., :seg_len]
        seg2 = sig_12ch[..., seg_len:]
        h1 = self.encoder(seg1)
        h2 = self.encoder(seg2)
        h = torch.cat([h1, h2], dim=1)
        if h.size(1) != 500:
            h = h.transpose(1, 2)
            h = nn.functional.interpolate(h, size=500, mode="linear", align_corners=False)
            h = h.transpose(1, 2)
        return h
