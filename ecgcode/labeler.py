# ecgcode/labeler.py
"""Convert NK delineate output + pacer spikes -> RLE token stream.

Algorithm (per the spec section 6):
1. Initialize sample-level array as iso
2. Mark P / T regions
3. Decompose each QRS into q/r/s by midpoint, with wide-QRS fallback (w)
4. Override with pacer spikes (priority: spike > wave > iso)
5. Run-length compress to (symbol_id, length_ms) events
"""

import numpy as np

from ecgcode import vocab
from ecgcode.delineate import DelineateResult

WIDE_QRS_THRESHOLD_MS = 120.0


def _safe_int(x, n: int):
    """Cast NK index (float, possibly NaN) to int and clamp to [0, n-1]. Returns None if NaN."""
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except (TypeError, ValueError):
        pass
    return max(0, min(n - 1, int(x)))


def _has(x) -> bool:
    if x is None:
        return False
    try:
        return not np.isnan(x)
    except (TypeError, ValueError):
        return True


def label(
    dr: DelineateResult,
    spike_idx,
    n_samples: int,
    fs: int = 500,
) -> list:
    """Build sample-level label array, then run-length compress to RLE events.

    Returns list of (symbol_id, length_ms) tuples.
    """
    ms_per_sample = 1000.0 / fs

    # NK total failure -> entire signal as one ? event
    if dr.n_beats == 0:
        return [(vocab.ID_UNK, int(round(n_samples * ms_per_sample)))]

    labels = np.full(n_samples, vocab.ID_ISO, dtype=np.uint8)

    # 1. P waves
    for on_f, off_f in zip(dr.p_onsets, dr.p_offsets):
        if not (_has(on_f) and _has(off_f)):
            continue
        on = _safe_int(on_f, n_samples)
        off = _safe_int(off_f, n_samples)
        labels[on:off + 1] = vocab.ID_P

    # 2. T waves
    for on_f, off_f in zip(dr.t_onsets, dr.t_offsets):
        if not (_has(on_f) and _has(off_f)):
            continue
        on = _safe_int(on_f, n_samples)
        off = _safe_int(off_f, n_samples)
        labels[on:off + 1] = vocab.ID_T

    # 3. QRS - q/r/s decomposition with wide-QRS fallback
    n_beats = dr.n_beats
    for i in range(n_beats):
        if not (_has(dr.r_onsets[i]) and _has(dr.r_offsets[i])):
            continue
        on = _safe_int(dr.r_onsets[i], n_samples)
        off = _safe_int(dr.r_offsets[i], n_samples)
        r = _safe_int(dr.r_peaks[i], n_samples)
        q_raw = dr.q_peaks[i] if i < len(dr.q_peaks) else None
        s_raw = dr.s_peaks[i] if i < len(dr.s_peaks) else None
        q = _safe_int(q_raw, n_samples)
        s = _safe_int(s_raw, n_samples)

        # Duration in ms: (off - on) samples gives the span
        qrs_ms = (off - on) * ms_per_sample
        has_q = q is not None
        has_s = s is not None

        # Wide-QRS fallback: no Q peak AND no S peak AND duration > 120ms
        if not has_q and not has_s and qrs_ms > WIDE_QRS_THRESHOLD_MS:
            labels[on:off + 1] = vocab.ID_W
            continue

        # Standard q/r/s decomposition by midpoints
        if r is None:
            # No R peak available; treat whole QRS as r
            labels[on:off + 1] = vocab.ID_R
            continue

        q_end = (q + r) // 2 if has_q else on
        s_start = (r + s) // 2 if has_s else off + 1

        if has_q:
            labels[on:q_end] = vocab.ID_Q
        labels[q_end:s_start] = vocab.ID_R
        if has_s:
            labels[s_start:off + 1] = vocab.ID_S

    # 4. Pacer spikes - highest priority override
    for idx in spike_idx:
        idx_int = int(idx)
        if 0 <= idx_int < n_samples:
            labels[idx_int] = vocab.ID_PACER

    # 5. RLE compress
    return _rle_compress(labels, ms_per_sample)


def _rle_compress(labels: np.ndarray, ms_per_sample: float) -> list:
    """Group consecutive identical labels -> list of (symbol_id, length_ms).

    Lengths are snapped to the codec grid (4ms) using cumulative rounding so
    rounding error never drifts more than 4ms from the true total. Single-sample
    pacer spikes are always emitted as 4ms (the minimum codec quantum).
    """
    from ecgcode.codec import MS_PER_UNIT

    if len(labels) == 0:
        return []
    change_idx = np.flatnonzero(np.diff(labels)) + 1
    boundaries = np.concatenate(([0], change_idx, [len(labels)]))

    events = []
    cum_true_ms = 0.0
    cum_emitted_ms = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sym = int(labels[start])
        n = int(end - start)
        cum_true_ms += n * ms_per_sample
        # Snap cumulative end to 4ms grid; segment length = grid_end - prev_emitted
        grid_end = int(round(cum_true_ms / MS_PER_UNIT)) * MS_PER_UNIT
        ms = grid_end - cum_emitted_ms
        if ms <= 0:
            # Segment too short to land on a new grid line; force 1 quantum so
            # we don't drop the symbol (e.g. a single-sample pacer spike).
            ms = MS_PER_UNIT
            grid_end = cum_emitted_ms + MS_PER_UNIT
        events.append((sym, ms))
        cum_emitted_ms = grid_end
    return events
