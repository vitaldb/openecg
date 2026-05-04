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


def extract_boundaries(frames, fs=250, frame_ms=20, boundary_shift_ms=None):
    """Extract per-wave boundary sample indices from a per-frame supercategory array.

    Returns dict: {p_on, p_off, qrs_on, qrs_off, t_on, t_off} -> list[int sample idx].

    boundary_shift_ms: optional dict with keys like {"p_off": -22} to apply a
    fixed shift in ms to specific boundaries. The v4 C checkpoint has a +22ms
    p_off systematic bias on LUDB val and +20ms on ISP test (model predicts P
    offset late vs cardiologist annotation); recommended shift = {"p_off": -22}.
    F (LUDB-only) has a smaller +14ms p_off bias; recommended shift = {"p_off": -15}.
    See `scripts/fix_p_off_bias.py` for the bias measurements per checkpoint.
    """
    import numpy as _np
    out = {"p_on": [], "p_off": [], "qrs_on": [], "qrs_off": [], "t_on": [], "t_off": []}
    super_to_name = {1: "p", 2: "qrs", 3: "t"}  # SUPER_P, SUPER_QRS, SUPER_T
    spf = int(round(frame_ms * fs / 1000.0))
    shifts = boundary_shift_ms or {}
    shift_samples = {k: int(round(v * fs / 1000.0)) for k, v in shifts.items()}
    prev = 0
    for f_idx, cur in enumerate(frames):
        cur = int(cur)
        if cur != prev:
            sample = f_idx * spf
            if prev in super_to_name:
                key = f"{super_to_name[prev]}_off"
                out[key].append(int(sample - 1 + shift_samples.get(key, 0)))
            if cur in super_to_name:
                key = f"{super_to_name[cur]}_on"
                out[key].append(int(sample + shift_samples.get(key, 0)))
        prev = cur
    if prev in super_to_name:
        sample = len(frames) * spf
        key = f"{super_to_name[prev]}_off"
        out[key].append(int(sample - 1 + shift_samples.get(key, 0)))
    return out


# Recommended per-checkpoint boundary shifts (measured on LUDB val + ISP test)
# Use with `extract_boundaries(..., boundary_shift_ms=BOUNDARY_SHIFT_C)`.
BOUNDARY_SHIFT_C = {"p_off": -22}  # C (combined big +lead_emb): +22ms p_off bias
BOUNDARY_SHIFT_F = {"p_off": -15}  # F (LUDB only no_lead_emb): +14ms p_off bias


def post_process_frames(frames, frame_ms=20, min_duration_ms=60, merge_gap_ms=200,
                        per_class_min_ms=None, per_class_merge_ms=None):
    """Apply post-processing to per-frame supercategory array.

    1. Remove segments shorter than min_duration_ms (replace with previous-segment label).
    2. Merge same-class segments separated by a gap shorter than merge_gap_ms.

    Per-class overrides: pass dicts keyed by class id (1=P, 2=QRS, 3=T) to use
    different thresholds per wave type. Class id 0 (other) does not merge. Tune
    sweep on LUDB val (`scripts/tune_postproc_v4.py`) found per-class optima:
    QRS=(min~20-40, merge~100), P/T=(min~60, merge~100-300). Single-default
    (60, 200) gives +0.01 avg boundary F1 vs old (40, 300); per-class can give
    further +0.005-0.015.
    """
    if len(frames) == 0:
        return np.asarray(frames, dtype=np.uint8)
    arr = np.asarray(frames, dtype=np.uint8).copy()
    n = len(arr)

    def class_min_frames(cls):
        ms = (per_class_min_ms or {}).get(int(cls), min_duration_ms)
        return max(1, int(ms / frame_ms))

    def class_merge_frames(cls):
        ms = (per_class_merge_ms or {}).get(int(cls), merge_gap_ms)
        return max(1, int(ms / frame_ms))

    # Step 1: remove short segments (absorb into previous segment if possible).
    # Threshold depends on the segment's own class.
    i = 0
    while i < n:
        j = i
        while j < n and arr[j] == arr[i]:
            j += 1
        seg_len = j - i
        if seg_len < class_min_frames(arr[i]) and i > 0:
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
        merge_frames = class_merge_frames(cls)
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
