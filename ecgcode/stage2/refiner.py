"""Stage 3 lightweight signal-aware boundary refinement.

The frame classifier produces boundaries quantized to frame transitions. These
helpers optionally move a boundary within a small local search window using the
raw normalized signal. They are opt-in and do not change default inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


BOUNDARY_KEYS = ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) == 0:
        return x
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(x, kernel, mode="same")


def _local_threshold(trace: np.ndarray, floor_quantile=0.2, peak_frac=0.35) -> float:
    if len(trace) == 0:
        return 0.0
    floor = float(np.quantile(trace, floor_quantile))
    peak = float(np.max(trace))
    return floor + peak_frac * max(0.0, peak - floor)


def refine_boundary(
    signal: Sequence[float],
    boundary_sample: int,
    boundary_key: str,
    fs: int = 250,
    search_ms: int = 80,
    qrs_peak_frac: float = 0.35,
    wave_peak_frac: float = 0.25,
) -> int:
    """Refine one boundary sample within a local signal window.

    QRS boundaries use the smoothed absolute derivative. P/T boundaries use the
    smoothed absolute amplitude after removing the local median baseline. The
    returned sample is clamped to the signal extent.
    """
    sig = np.asarray(signal, dtype=float)
    if len(sig) == 0:
        return int(boundary_sample)
    center = int(np.clip(boundary_sample, 0, len(sig) - 1))
    radius = max(1, int(round(search_ms * fs / 1000.0)))
    lo = max(0, center - radius)
    hi = min(len(sig), center + radius + 1)
    if hi - lo < 3:
        return center

    local = sig[lo:hi]
    rel_center = center - lo
    is_on = boundary_key.endswith("_on")
    is_qrs = boundary_key.startswith("qrs_")

    if is_qrs:
        trace = np.abs(np.diff(local, prepend=local[0]))
        trace = _smooth(trace, max(1, int(round(0.012 * fs))))
    else:
        baseline = float(np.median(local))
        trace = _smooth(np.abs(local - baseline), max(1, int(round(0.024 * fs))))

    if is_on:
        search_trace = trace[:rel_center + 1]
        offset = 0
    else:
        search_trace = trace[rel_center:]
        offset = rel_center
    peak_frac = qrs_peak_frac if is_qrs else wave_peak_frac
    threshold = _local_threshold(search_trace, peak_frac=peak_frac)
    active = np.flatnonzero(search_trace >= threshold)
    if len(active) == 0:
        return center

    if is_on:
        refined_rel = int(active[0] + offset)
    else:
        refined_rel = int(active[-1] + offset)
    return int(np.clip(lo + refined_rel, 0, len(sig) - 1))


def refine_boundaries(
    signal: Sequence[float],
    boundaries: Mapping[str, Sequence[int]],
    fs: int = 250,
    search_ms: int = 80,
    refine_p_t: bool = False,
    refine_qrs: bool = True,
) -> dict[str, list[int]]:
    """Refine boundary dict returned by `extract_boundaries`.

    By default only QRS is refined because QRS has a high-SNR derivative cue.
    P/T refinement is available but should be validated per dataset before use.
    """
    out: dict[str, list[int]] = {}
    for key in BOUNDARY_KEYS:
        values = boundaries.get(key, ())
        should_refine = (key.startswith("qrs_") and refine_qrs) or (
            not key.startswith("qrs_") and refine_p_t
        )
        if should_refine:
            out[key] = [
                refine_boundary(signal, int(v), key, fs=fs, search_ms=search_ms)
                for v in values
            ]
        else:
            out[key] = [int(v) for v in values]
    return out
