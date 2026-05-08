# openecg/stage2/qrs_channel.py
"""Rule-based QRS-position input channel for v22 input-channel-prior models.

Uses ``openecg.detect_qrs`` for R-peak detection (validated at micro-F1
0.994 on MIT-BIH Arrhythmia DB — see scripts/validate_qrs_mitdb.py).
The output is a boxcar-broadened binary indicator at ``target_fs``;
detection happens at ``fs_in`` to keep R-peak resolution intact.
"""
from __future__ import annotations

import numpy as np

from openecg.qrs import detect_qrs as _detect_r_peaks


DEFAULT_BROADEN_MS = 40.0


def qrs_position_channel(
    sig: np.ndarray,
    fs_in: int,
    target_fs: int = 250,
    broaden_ms: float = DEFAULT_BROADEN_MS,
) -> np.ndarray:
    """Return a [N_at_target_fs] float32 binary indicator with 1.0 in a
    ``broaden_ms`` ms boxcar centered on each detected R-peak.

    The broaden window (default 40 ms) is wide enough to cover normal
    QRS half-widths (~30 ms each side) so the indicator stays "on"
    throughout the QRS complex even though detection points to the
    R-peak only. Overlapping boxcars saturate at 1.0.
    """
    sig = np.asarray(sig, dtype=np.float32)
    n_in = len(sig)
    n_out = 0 if n_in == 0 else int(round(n_in * target_fs / fs_in))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)

    try:
        r_peaks = _detect_r_peaks(sig, fs_in)
    except Exception:
        r_peaks = np.empty(0, dtype=np.int64)

    out = np.zeros(n_out, dtype=np.float32)
    if r_peaks.size == 0:
        return out
    half = max(1, int(round(broaden_ms * target_fs / 1000 / 2)))
    r_target = np.round(r_peaks * (target_fs / fs_in)).astype(np.int64)
    for r in r_target:
        lo = max(0, int(r) - half)
        hi = min(n_out, int(r) + half + 1)
        if hi > lo:
            out[lo:hi] = 1.0
    return out


def qrs_position_channel_from_indices(
    qrs_indices: np.ndarray,
    fs_in: int,
    n_in_samples: int,
    target_fs: int = 250,
    broaden_ms: float = DEFAULT_BROADEN_MS,
) -> np.ndarray:
    """Same output as ``qrs_position_channel`` but takes an already-computed
    QRS index array (e.g. ground-truth labels) instead of running detection.
    Useful for unit tests and for trainer paths that want to use GT QRS
    instead of openecg.detect_qrs.
    """
    n_out = 0 if n_in_samples == 0 else int(round(n_in_samples * target_fs / fs_in))
    out = np.zeros(n_out, dtype=np.float32)
    qrs_indices = np.asarray(qrs_indices, dtype=np.int64)
    if qrs_indices.size == 0 or n_out == 0:
        return out
    half = max(1, int(round(broaden_ms * target_fs / 1000 / 2)))
    qrs_target = np.round(qrs_indices * (target_fs / fs_in)).astype(np.int64)
    for q in qrs_target:
        lo = max(0, int(q) - half)
        hi = min(n_out, int(q) + half + 1)
        if hi > lo:
            out[lo:hi] = 1.0
    return out
