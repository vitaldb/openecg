"""HuBERT-ECG adapter - wraps the HuggingFace HuBERT-style ECG encoder for
single-lead 250 Hz / 10 s input.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md §5.2.

Adapter pipeline:
  sig [B, 2500] @ 250Hz, single lead
    -> resample to 100 Hz (length 1000)
    -> split into 2 x 500-sample (5 s) segments
    -> encoder.last_hidden_state on each -> [B, ~250, d]
    -> concat to [B, ~500, d] and pad/truncate to exactly 500 frames
"""

from __future__ import annotations

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch import nn


HUBERT_DEFAULT_MODEL_ID = "Edoardo-BS/hubert_ecg_small"


class HubertECGAdapter(nn.Module):
    """Wraps a HuggingFace HuBERT-ECG encoder."""

    def __init__(self, model_id: str = HUBERT_DEFAULT_MODEL_ID,
                 device: str = "cpu",
                 target_fs: int = 100,
                 window_seconds: int = 5):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_id)
        self.hidden_dim = int(self.encoder.config.hidden_size)
        self.target_fs = int(target_fs)
        self.window_seconds = int(window_seconds)

    def _resample_250_to_target(self, sig_250: torch.Tensor) -> torch.Tensor:
        n_in = sig_250.size(-1)
        n_out = int(round(n_in * self.target_fs / 250))
        np_sig = sig_250.detach().cpu().numpy().astype(np.float32)
        resampled = scipy_signal.resample(np_sig, n_out, axis=-1).astype(np.float32)
        return torch.from_numpy(resampled).to(sig_250.device)

    def _encode_segment(self, seg: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=seg)
        return out.last_hidden_state

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor) -> torch.Tensor:
        sig_target = self._resample_250_to_target(sig)
        seg_len = self.target_fs * self.window_seconds
        n = sig_target.size(-1)
        assert n == 2 * seg_len, f"expected {2 * seg_len} samples, got {n}"
        seg1 = sig_target[..., :seg_len]
        seg2 = sig_target[..., seg_len:]
        h1 = self._encode_segment(seg1)
        h2 = self._encode_segment(seg2)
        h = torch.cat([h1, h2], dim=1)
        if h.size(1) != 500:
            h = h.transpose(1, 2)
            h = nn.functional.interpolate(h, size=500, mode="linear", align_corners=False)
            h = h.transpose(1, 2)
        return h
