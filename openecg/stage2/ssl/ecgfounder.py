"""ECGFounder adapter - wraps the Net1D supervised foundation model
(Li et al. 2024, arXiv:2410.04133, PKUDigitalHealth/ECGFounder) for our
single-lead 250 Hz / 10 s pipeline.

Net1D config (matches `ptbxl_eval.py` exactly so the released
`1_lead_ECGFounder.pth` state_dict loads cleanly):

    in_channels=1, base_filters=64, ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16, stride=2, groups_width=16,
    use_bn=False, use_do=False, n_classes=150

Total temporal downsample is 2 (first_conv) * 2^7 (one per stage) = 256x.
At the official 500 Hz / 10 s input that maps 5000 samples to ~19 tokens
which is far too coarse for boundary delineation. We therefore tap the
output of stage 1 (after 8x downsample) where the feature has shape
[B, 160, 625]: token rate 62.5 Hz / 16 ms per frame, comfortably within
the QRS 40 ms tolerance and the P / T_on 50 ms / T_off 100 ms tolerances.

Pipeline:
    sig [B, 2500] @ 250 Hz, single lead
      -> resample to 500 Hz (length 5000) on GPU via interpolate
      -> z-score per signal
      -> [B, 1, 5000]
      -> first_conv + bn + act       -> [B, 64,  2500]
      -> stage 0                     -> [B, 64,  1250]
      -> stage 1                     -> [B, 160,  625]
      -> transpose + interp to 500   -> [B, 500, 160]
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
ECGFOUNDER_DEFAULT_CKPT = REPO_ROOT / "data" / "checkpoints" / "external" / "1_lead_ECGFounder.pth"
ECGFOUNDER_VENDOR_PATH = REPO_ROOT / "third_party" / "ECGFounder"


def _load_net1d_class():
    import sys
    if str(ECGFOUNDER_VENDOR_PATH) not in sys.path:
        sys.path.insert(0, str(ECGFOUNDER_VENDOR_PATH))
    from net1d import Net1D
    return Net1D


class ECGFounderAdapter(nn.Module):
    """Single-lead ECGFounder feature extractor tapped at end of stage 1."""

    HIDDEN_DIM = 160
    TAP_STAGE = 1

    def __init__(self, weights_path: str | None = None,
                 device: str = "cpu",
                 native_fs: int = 250,
                 target_fs: int = 500,
                 target_samples: int = 5000):
        super().__init__()
        Net1D = _load_net1d_class()
        self.net1d = Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=150,
        )
        if weights_path is None:
            weights_path = ECGFOUNDER_DEFAULT_CKPT
        weights_path = Path(weights_path)
        if weights_path.exists():
            blob = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = blob.get("state_dict", blob.get("model", blob))
            log = self.net1d.load_state_dict(state, strict=False)
            missing = [k for k in log.missing_keys if not k.startswith("dense.")]
            if missing:
                raise RuntimeError(
                    f"Unexpected missing keys when loading ECGFounder: {missing[:8]}"
                )
        self.hidden_dim = self.HIDDEN_DIM
        self.native_fs = int(native_fs)
        self.target_fs = int(target_fs)
        self.target_samples = int(target_samples)

    def _resample(self, sig: torch.Tensor) -> torch.Tensor:
        """sig [B, N_in] -> [B, target_samples] linear interpolation."""
        x = sig.unsqueeze(1)
        x = F.interpolate(x, size=self.target_samples, mode="linear",
                          align_corners=False)
        return x.squeeze(1)

    def _zscore(self, sig: torch.Tensor) -> torch.Tensor:
        mean = sig.mean(dim=-1, keepdim=True)
        std = sig.std(dim=-1, keepdim=True) + 1e-8
        return (sig - mean) / std

    def _features_through_stage(self, x: torch.Tensor, last_stage: int) -> torch.Tensor:
        out = self.net1d.first_conv(x)
        if self.net1d.use_bn:
            out = self.net1d.first_bn(out)
        out = self.net1d.first_activation(out)
        for i_stage in range(last_stage + 1):
            out = self.net1d.stage_list[i_stage](out)
        return out

    def forward(self, sig: torch.Tensor, lead_id: torch.Tensor) -> torch.Tensor:
        del lead_id
        if sig.dim() != 2:
            raise ValueError(f"Expected sig [B, N], got {tuple(sig.shape)}")
        x = self._resample(sig)
        x = self._zscore(x)
        x = x.unsqueeze(1)
        feat = self._features_through_stage(x, self.TAP_STAGE)
        feat = F.interpolate(feat, size=500, mode="linear", align_corners=False)
        return feat.transpose(1, 2).contiguous()
