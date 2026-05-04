# ecgcode/stage2/infer.py
"""Stage 2 inference: checkpoint to per-frame predictions for validation."""

import numpy as np
import torch

from ecgcode import codec
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import load_checkpoint


def load_model(ckpt_path, device="cuda", **model_kwargs):
    """Load a checkpoint into a FrameClassifier.

    Pass the same model hyperparameters as used at training time (e.g.
    `d_model=128, n_layers=8` for the v3 checkpoint).
    """
    model = FrameClassifier(**model_kwargs)
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


def post_process_frames(frames, frame_ms=20, min_duration_ms=60, merge_gap_ms=200):
    """Apply post-processing to per-frame supercategory array.

    1. Remove segments shorter than min_duration_ms (replace with previous-segment label).
    2. Merge same-class segments separated by a gap shorter than merge_gap_ms.

    Defaults tuned via `scripts/tune_postproc_v4.py` on LUDB val (24-combo grid):
    (60, 200) gives best avg boundary F1 across both C (combined) and F (LUDB-only)
    checkpoints. F gains ~+0.022 avg, C gains ~+0.010 avg vs the previous (40, 300)
    defaults. Per-class optima vary (QRS prefers smaller min, P/T prefer larger),
    but (60, 200) is a robust single-default compromise.
    """
    if len(frames) == 0:
        return np.asarray(frames, dtype=np.uint8)
    arr = np.asarray(frames, dtype=np.uint8).copy()
    min_frames = max(1, int(min_duration_ms / frame_ms))
    merge_frames = max(1, int(merge_gap_ms / frame_ms))
    n = len(arr)

    # Step 1: remove short segments (absorb into previous segment if possible).
    i = 0
    while i < n:
        j = i
        while j < n and arr[j] == arr[i]:
            j += 1
        seg_len = j - i
        if seg_len < min_frames and i > 0:
            arr[i:j] = arr[i - 1]
        i = j

    # Step 2: merge close same-class segments. Only merge physiological classes
    # (P=1, QRS=2, T=3); do NOT extend `other` (0) across an event.
    i = 0
    while i < n:
        cls = arr[i]
        # Find end of current run of cls.
        j = i
        while j < n and arr[j] == cls:
            j += 1
        if j >= n:
            break
        if cls == 0:
            i = j
            continue
        # Look ahead for next occurrence of same class within merge_frames.
        k = j
        while k < n and (k - j) < merge_frames and arr[k] != cls:
            k += 1
        if k < n and (k - j) < merge_frames and arr[k] == cls:
            arr[j:k] = cls
            # Continue from k (the merged region is now one big block of cls).
            i = k
        else:
            i = j

    return arr
