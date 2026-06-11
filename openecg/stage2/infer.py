# openecg/stage2/infer.py
"""Stage 2 inference: checkpoint to per-frame predictions for validation."""

import numpy as np
import torch

from openecg.layered import frames_to_events
from openecg.stage2.model import FrameClassifier
from openecg.stage2.train import load_checkpoint, load_checkpoint_blob


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
    """Single-sequence inference to frame-class runs (for boundary extraction)."""
    frames = predict_frames(model, sig, lead_id, device=device)
    return frames_to_events(frames, frame_ms=frame_ms)


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
        from openecg.stage2.refiner import refine_boundaries
        boundaries = refine_boundaries(sig, boundaries, fs=fs, **(refine_kwargs or {}))
    return boundaries


def extract_boundaries(frames, fs=250, frame_ms=20):
    """Extract per-wave boundary sample indices from a per-frame supercategory array.

    Returns dict: {p_on, p_off, qrs_on, qrs_off, t_on, t_off} -> list[int sample idx].
    Boundaries reflect the model's raw frame transitions with no shift applied.

    SUPER_PACED_QRS (= 4, v18+) is folded into SUPER_QRS so paced and
    sinus QRS contribute to the same boundary stream. Callers wanting
    the paced/non-paced distinction should inspect the frame array
    directly before calling this.
    """
    from openecg import eval as _ee
    frames = _ee.fold_paced_to_qrs(np.asarray(frames, dtype=np.uint8))
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
    # Fold SUPER_PACED_QRS to SUPER_QRS so the legacy postprocessing
    # (which reasons about per-class min duration / merge gap) treats
    # paced and sinus QRS uniformly.
    from openecg import eval as _ee
    arr = _ee.fold_paced_to_qrs(np.asarray(frames, dtype=np.uint8))
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


def suppress_p_after_wide_qrs(
    frames,
    *,
    frame_ms: int = 20,
    qrs_wide_ms: float = 120,
    refractory_ms: float = 300,
):
    """Backwards-compatible alias for `suppress_p_around_wide_qrs` with
    only post-QRS suppression. Kept so existing callers keep working.
    """
    return suppress_p_around_wide_qrs(
        frames, frame_ms=frame_ms, qrs_wide_ms=qrs_wide_ms,
        pre_ms=0, post_ms=refractory_ms,
    )


def suppress_p_around_wide_qrs(
    frames,
    *,
    frame_ms: int = 20,
    qrs_wide_ms: float = 120,
    pre_ms: float = 300,
    post_ms: float = 300,
):
    """Convert P runs to Other when they fall in the refractory window
    surrounding a wide QRS.

    Heuristic for paced / BBB / 3°AVB-paced false-positives. A wide QRS
    (≥ 120 ms) is generally paced or aberrant; the next genuine P arrives
    at the atrial cycle (≥ 600 ms apart). Two FP modes are common at
    inference time:

    * **Pre-QRS** — the model fires a "P band" 100-300 ms BEFORE a paced
      QRS, modelling normal sinus PR coupling. In paced rhythms the
      ventricle fires independently of atrial activity, so any P that
      lines up with the paced beat's lead-in is fictitious. (Empirically
      ~97% of v17's BUT PDB rid=3 paced FPs land in [-300, -100) ms.)
    * **Post-QRS** — T-wave / late ST of the wide complex mistaken for P
      in the early refractory phase.

    Both modes are suppressed by this single rule. Setting pre_ms=0
    disables pre-QRS suppression (legacy behaviour); post_ms=0 disables
    post-QRS suppression. Normal sinus is unaffected because narrow QRS
    runs don't trigger the rule, so the standard PR interval P stays.

    Args:
        frames: per-frame supercategory uint8 array.
        qrs_wide_ms: minimum QRS duration (ms) to treat as wide.
        pre_ms:  P runs whose END falls within this window BEFORE a wide
            QRS-on are suppressed.
        post_ms: P runs whose START falls within this window AFTER a wide
            QRS-off are suppressed.

    Returns a new array; `frames` is not modified in place.
    """
    arr = np.asarray(frames, dtype=np.uint8).copy()
    n = len(arr)
    pre_frames = max(0, int(round(pre_ms / frame_ms)))
    post_frames = max(0, int(round(post_ms / frame_ms)))
    qrs_wide_frames = max(1, int(round(qrs_wide_ms / frame_ms)))
    SUPER_OTHER = 0
    SUPER_P = 1
    SUPER_QRS = 2

    # First pass: collect (qrs_on, qrs_off) of all wide-QRS runs.
    wide_runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if arr[i] != SUPER_QRS:
            i += 1
            continue
        j = i
        while j < n and arr[j] == SUPER_QRS:
            j += 1
        if (j - i) >= qrs_wide_frames:
            wide_runs.append((i, j))
        i = j

    if not wide_runs:
        return arr

    # Second pass: walk all P runs, suppress if any wide QRS sits within
    # the (pre, post) refractory windows.
    i = 0
    while i < n:
        if arr[i] != SUPER_P:
            i += 1
            continue
        ps = i
        pe = i
        while pe < n and arr[pe] == SUPER_P:
            pe += 1
        # P run is [ps, pe). Check refractory windows.
        suppressed = False
        for q_on, q_off in wide_runs:
            # Pre-QRS: P ends within `pre_frames` before this wide QRS-on.
            if pre_frames > 0 and ps <= q_on and (q_on - pe) < pre_frames:
                suppressed = True
                break
            # Post-QRS: P starts within `post_frames` after this wide QRS-off.
            if post_frames > 0 and ps >= q_off and (ps - q_off) < post_frames:
                suppressed = True
                break
        if suppressed:
            arr[ps:pe] = SUPER_OTHER
        i = pe
    return arr


REG_CHANNELS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


@torch.no_grad()
def predict_frames_with_reg(model, sig, lead_id, device="cuda"):
    """Single-sequence inference for a (cls, reg, [aux]) tuple-output model.
    Returns (frames[T] uint8, reg_offsets[T, 6] float32).

    ``sig`` may be either:
      * 1-D ndarray of shape [T] — for single-channel models (legacy).
      * 2-D ndarray of shape [C, T] — for multi-input variants such as
        ``FrameClassifierViTRegMultiIn`` (signal + pacer + qrs channels).
    Accepts (cls, reg) or (cls, reg, aux) tuple outputs; aux is ignored.
    """
    arr = np.asarray(sig, dtype=np.float32)
    if arr.ndim == 1:
        x = torch.from_numpy(arr).unsqueeze(0).to(device)        # [1, T]
    elif arr.ndim == 2:
        x = torch.from_numpy(arr).unsqueeze(0).to(device)        # [1, C, T]
    else:
        raise ValueError(f"sig must be 1-D or 2-D, got shape {arr.shape}")
    lid = torch.tensor([lead_id], dtype=torch.long, device=device)
    out = model(x, lid)
    cls_logits = out[0]
    reg = out[1]
    frames = cls_logits.argmax(dim=-1).cpu().numpy().squeeze(0).astype(np.uint8)
    # v50d: model returns reg=None when reg_head is removed (NoReg sibling).
    if reg is None:
        return frames, None
    reg_np = reg.cpu().numpy().squeeze(0).astype(np.float32)
    return frames, reg_np


def apply_reg_to_boundaries(boundaries, reg_offsets, samples_per_frame=5,
                              max_window=10000):
    """Refine boundary samples by adding the reg head's predicted offset
    at the corresponding frame.

    boundaries: dict from extract_boundaries (key -> list[int sample]).
    reg_offsets: [T, 6] array; channel order = REG_CHANNELS.
    """
    refined: dict[str, list[int]] = {}
    for key, samples in boundaries.items():
        if key not in REG_CHANNELS:
            refined[key] = list(samples)
            continue
        ch = REG_CHANNELS.index(key)
        out: list[int] = []
        T = reg_offsets.shape[0]
        for s in samples:
            f = int(s) // samples_per_frame
            if 0 <= f < T:
                shifted = int(s) + int(round(float(reg_offsets[f, ch])))
            else:
                shifted = int(s)
            shifted = max(0, min(int(max_window) - 1, shifted))
            out.append(shifted)
        refined[key] = out
    return refined
