# ecgcode/stage2/infer.py
"""Stage 2 inference: checkpoint to per-frame predictions for validation."""

import numpy as np
import torch

from ecgcode import codec
from ecgcode.stage2.model import FrameClassifier
from ecgcode.stage2.train import load_checkpoint, load_checkpoint_blob


def infer_model_config_from_state_dict(state_dict):
    """Infer FrameClassifier kwargs from a checkpoint state dict."""
    d_model = int(state_dict["conv2.weight"].shape[0])
    n_classes = int(state_dict["head.weight"].shape[0])
    n_leads = 12
    use_lead_emb = "lead_emb.weight" in state_dict
    if use_lead_emb:
        n_leads = int(state_dict["lead_emb.weight"].shape[0])

    layer_prefixes = {
        key.split(".layers.")[1].split(".")[0]
        for key in state_dict
        if key.startswith("transformer.layers.")
    }
    n_layers = len(layer_prefixes)
    ff = int(state_dict["transformer.layers.0.linear1.weight"].shape[0])
    return {
        "n_leads": n_leads,
        "d_model": d_model,
        "n_layers": n_layers,
        "ff": ff,
        "n_classes": n_classes,
        "use_lead_emb": use_lead_emb,
    }


def load_model(ckpt_path, device="cuda", **model_kwargs):
    """Load a checkpoint into a FrameClassifier.

    If the checkpoint contains `model_config`, kwargs are optional. Explicit
    kwargs override the checkpoint config for backward compatibility.
    """
    if model_kwargs:
        config = model_kwargs
    else:
        blob = load_checkpoint_blob(ckpt_path)
        config = blob.get("model_config") or infer_model_config_from_state_dict(blob["model_state"])
    model = FrameClassifier(**config)
    load_checkpoint(ckpt_path, model)
    model = model.to(device).eval()
    return model


def load_model_bundle(ckpt_path, device="cuda", **model_kwargs):
    """Load model plus self-describing inference metadata."""
    blob = load_checkpoint_blob(ckpt_path)
    config = model_kwargs or blob.get("model_config") or infer_model_config_from_state_dict(blob["model_state"])
    model = FrameClassifier(**config)
    model.load_state_dict(blob["model_state"])
    model = model.to(device).eval()
    return {
        "model": model,
        "metrics": blob.get("metrics", {}),
        "model_config": config,
        "postprocess_config": blob.get("postprocess_config", {}),
        "train_config": blob.get("config", {}),
        "extra": blob.get("extra", {}),
    }


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


def predict_to_boundaries(
    model,
    sig,
    lead_id,
    device="cuda",
    fs=250,
    frame_ms=20,
    postprocess=True,
    postprocess_kwargs=None,
    refine=False,
    refine_kwargs=None,
):
    """Single-sequence inference to boundary sample indices.

    Set `refine=True` to apply the optional Stage 3 signal-aware refiner after
    frame post-processing.
    """
    frames = predict_frames(model, sig, lead_id, device=device)
    if postprocess:
        frames = post_process_frames(frames, frame_ms=frame_ms, **(postprocess_kwargs or {}))
    boundaries = extract_boundaries(frames, fs=fs, frame_ms=frame_ms)
    if refine:
        from ecgcode.stage2.refiner import refine_boundaries
        boundaries = refine_boundaries(sig, boundaries, fs=fs, **(refine_kwargs or {}))
    return boundaries


def extract_boundaries(frames, fs=250, frame_ms=20):
    """Extract per-wave boundary sample indices from a per-frame supercategory array.

    Returns dict: {p_on, p_off, qrs_on, qrs_off, t_on, t_off} -> list[int sample idx].
    Boundaries reflect the model's raw frame transitions with no shift applied.
    """
    out = {"p_on": [], "p_off": [], "qrs_on": [], "qrs_off": [], "t_on": [], "t_off": []}
    super_to_name = {1: "p", 2: "qrs", 3: "t"}  # SUPER_P, SUPER_QRS, SUPER_T
    spf = int(round(frame_ms * fs / 1000.0))
    prev = 0
    for f_idx, cur in enumerate(frames):
        cur = int(cur)
        if cur != prev:
            sample = f_idx * spf
            if prev in super_to_name:
                out[f"{super_to_name[prev]}_off"].append(int(sample - 1))
            if cur in super_to_name:
                out[f"{super_to_name[cur]}_on"].append(int(sample))
        prev = cur
    if prev in super_to_name:
        sample = len(frames) * spf
        out[f"{super_to_name[prev]}_off"].append(int(sample - 1))
    return out


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
