# ecgcode/stage2/infer.py
"""Stage 2 inference: checkpoint to per-frame predictions for validation."""

import numpy as np
import torch

from ecgcode import codec
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import load_checkpoint


def load_model(ckpt_path, device="cuda"):
    model = FrameClassifier()
    load_checkpoint(ckpt_path, model)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def predict_frames(model, sig, lead_id, device="cuda"):
    """Single-sequence inference: signal[2500] to frame argmax [500] (uint8)."""
    x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
    lid = torch.tensor([lead_id], dtype=torch.long, device=device)
    logits = model(x, lid)
    pred = logits.argmax(dim=-1).cpu().numpy().squeeze(0).astype(np.uint8)
    return pred


def predict_to_events(model, sig, lead_id, device="cuda", frame_ms=20):
    """Single-sequence inference to RLE events (for boundary extraction)."""
    frames = predict_frames(model, sig, lead_id, device=device)
    return codec.from_frames(frames, frame_ms=frame_ms)
