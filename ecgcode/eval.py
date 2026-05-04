"""Evaluation metrics — frame-level F1 (4-class) + boundary error (Martinez-style).

Spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md §7
"""

import numpy as np

from ecgcode import vocab

# Supercategory IDs for LUDB-compat 4-class comparison
SUPER_OTHER = 0
SUPER_P = 1
SUPER_QRS = 2
SUPER_T = 3
SUPER_NAMES = {SUPER_OTHER: "other", SUPER_P: "P", SUPER_QRS: "QRS", SUPER_T: "T"}

# Sentinel for masked frames (boundary regions where the model has one-sided
# context and predictions are unreliable). PyTorch cross_entropy supports
# `ignore_index` natively; our focal_cross_entropy does too.
IGNORE_INDEX = 255

_SUPER_MAP = {
    vocab.ID_PAD: SUPER_OTHER,
    vocab.ID_UNK: SUPER_OTHER,
    vocab.ID_ISO: SUPER_OTHER,
    vocab.ID_PACER: SUPER_OTHER,
    vocab.ID_P: SUPER_P,
    vocab.ID_Q: SUPER_QRS,
    vocab.ID_R: SUPER_QRS,
    vocab.ID_S: SUPER_QRS,
    vocab.ID_W: SUPER_QRS,
    vocab.ID_T: SUPER_T,
    vocab.ID_U: SUPER_T,    # U is repolarization-adjacent
    vocab.ID_D: SUPER_QRS,  # delta is QRS-adjacent
    vocab.ID_J: SUPER_QRS,
}


def to_supercategory(frames: np.ndarray) -> np.ndarray:
    """Map per-frame v1 alphabet IDs → LUDB-compat 4-class."""
    out = np.zeros_like(frames, dtype=np.uint8)
    for src, dst in _SUPER_MAP.items():
        out[frames == src] = dst
    return out


def frame_f1(pred: np.ndarray, true: np.ndarray) -> dict:
    """Per-supercategory precision/recall/F1.

    Frames where `true == IGNORE_INDEX` are excluded from all counts (TP/FP/FN).
    Pred values at those positions are also ignored regardless of their value.

    Returns: {super_id: {'precision': p, 'recall': r, 'f1': f, 'tp', 'fp', 'fn'}}
    """
    valid = true != IGNORE_INDEX
    pred = pred[valid]
    true = true[valid]
    out = {}
    for sc in (SUPER_OTHER, SUPER_P, SUPER_QRS, SUPER_T):
        tp = int(np.sum((pred == sc) & (true == sc)))
        fp = int(np.sum((pred == sc) & (true != sc)))
        fn = int(np.sum((pred != sc) & (true == sc)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[sc] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return out


def boundary_error(
    pred_indices: list[int],
    true_indices: list[int],
    tolerance_ms: float,
    fs: int,
) -> dict:
    """Greedy nearest-match boundary comparison (Martinez 2004 style).

    For each true boundary, find nearest predicted within tolerance.
    Returns sensitivity, PPV, mean/median/p95 error in ms, hit/miss counts.
    """
    tolerance_samples = tolerance_ms * fs / 1000
    pred_arr = np.sort(np.array(pred_indices, dtype=int))
    true_arr = np.sort(np.array(true_indices, dtype=int))

    if len(true_arr) == 0 and len(pred_arr) == 0:
        return _empty_boundary_result()

    matched_pred = set()
    errors_samples = []
    n_hits = 0

    for t in true_arr:
        if len(pred_arr) == 0:
            break
        best_idx = -1
        best_err = float("inf")
        for j, p in enumerate(pred_arr):
            if j in matched_pred:
                continue
            err = abs(int(p) - int(t))
            if err < best_err:
                best_err = err
                best_idx = j
        if best_idx >= 0 and best_err <= tolerance_samples:
            matched_pred.add(best_idx)
            errors_samples.append(best_err)
            n_hits += 1

    if errors_samples:
        errors_ms = np.array(errors_samples) * 1000.0 / fs
        mean_err = float(np.mean(errors_ms))
        median_err = float(np.median(errors_ms))
        p95_err = float(np.percentile(errors_ms, 95))
    else:
        mean_err = median_err = p95_err = 0.0

    sensitivity = n_hits / len(true_arr) if len(true_arr) > 0 else 0.0
    ppv = n_hits / len(pred_arr) if len(pred_arr) > 0 else 0.0

    return {
        "sensitivity": sensitivity,
        "ppv": ppv,
        "n_hits": n_hits,
        "n_true": int(len(true_arr)),
        "n_pred": int(len(pred_arr)),
        "mean_error_ms": mean_err,
        "median_error_ms": median_err,
        "p95_error_ms": p95_err,
    }


def boundary_f1(pred_indices, true_indices, tolerance_ms, fs):
    """Compute F1 from boundary_error sensitivity and PPV.

    Literature standard: each true boundary matched to a predicted boundary within
    tolerance_ms gives a TP. Sensitivity = TP / |true|, PPV = TP / |pred|.
    F1 = 2 * sens * PPV / (sens + PPV).
    """
    res = boundary_error(pred_indices, true_indices, tolerance_ms=tolerance_ms, fs=fs)
    sens = res["sensitivity"]
    ppv = res["ppv"]
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    return {**res, "f1": f1}


def _empty_boundary_result():
    return {
        "sensitivity": 0.0, "ppv": 0.0, "n_hits": 0, "n_true": 0, "n_pred": 0,
        "mean_error_ms": 0.0, "median_error_ms": 0.0, "p95_error_ms": 0.0,
    }


def events_to_super_frames(events, n_samples, fs=500, frame_ms=20):
    """Pipeline events → per-frame supercategory array.
    Used by validate_v1.py and ablate_methods.py."""
    from ecgcode import codec
    total_ms = round(n_samples * 1000.0 / fs)
    frames = codec.to_frames(events, frame_ms=frame_ms, total_ms=total_ms)
    return to_supercategory(frames)


def gt_to_super_frames(gt_ann, n_samples, fs=500, frame_ms=20):
    """LUDB cardiologist annotation dict → per-frame supercategory array (majority per frame).

    samples_per_frame is fixed by physical time (fs * frame_ms / 1000) so each
    output frame represents exactly frame_ms of signal. n_frames = n_samples //
    samples_per_frame; trailing samples that don't fit a full frame are dropped.
    Earlier versions computed samples_per_frame = n_samples // n_frames, which
    introduced cumulative time drift (up to 500ms by frame 499) when n_samples
    was not a clean multiple of samples-per-frame (e.g. ISP records of 9998-9999
    samples at 1000Hz with frame_ms=20).
    """
    samples_per_frame = round(fs * frame_ms / 1000.0)
    if samples_per_frame < 1:
        samples_per_frame = 1
    sample_labels = np.full(n_samples, SUPER_OTHER, dtype=np.uint8)
    for on, off in zip(gt_ann["p_on"], gt_ann["p_off"]):
        sample_labels[on:off + 1] = SUPER_P
    for on, off in zip(gt_ann["qrs_on"], gt_ann["qrs_off"]):
        sample_labels[on:off + 1] = SUPER_QRS
    for on, off in zip(gt_ann["t_on"], gt_ann["t_off"]):
        sample_labels[on:off + 1] = SUPER_T
    n_frames = n_samples // samples_per_frame
    out = np.zeros(n_frames, dtype=np.uint8)
    for f in range(n_frames):
        seg = sample_labels[f * samples_per_frame: (f + 1) * samples_per_frame]
        vals, counts = np.unique(seg, return_counts=True)
        out[f] = vals[np.argmax(counts)]
    return out
