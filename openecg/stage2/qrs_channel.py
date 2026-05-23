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

    ``broaden_ms = 0`` produces a strict 1-sample point indicator at
    each R-peak (one sample = 1.0, the rest 0); positive values give a
    boxcar of width ``broaden_ms``. The default 40 ms covers normal
    QRS half-widths (~30 ms each side) so the indicator stays "on"
    throughout the QRS complex. Overlapping boxcars saturate at 1.0.
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
    half = max(0, int(round(broaden_ms * target_fs / 1000 / 2)))
    r_target = np.round(r_peaks * (target_fs / fs_in)).astype(np.int64)
    for r in r_target:
        lo = max(0, int(r) - half)
        hi = min(n_out, int(r) + half + 1)
        if hi > lo:
            out[lo:hi] = 1.0
    return out


def qrs_pointwise_resampled(
    qrs_native_indices: np.ndarray,
    fs_native: int,
    n_native_samples: int,
    target_fs: int = 250,
) -> np.ndarray:
    """Strict 1-sample-per-spike QRS indicator at ``target_fs``, derived from
    native-fs QRS detections via max-pool downsampling.

    Each native R-peak index ``q`` maps to a single target-fs bin
    ``b = floor(q * target_fs / fs_native)`` and ``out[b] = 1.0``. If
    multiple native spikes fall in the same target bin (only happens when
    ``fs_native > target_fs`` and two beats are spaced less than the bin
    width apart — physiologically impossible for QRS), the bin still
    saturates at 1.0 (max-pool semantics).

    Use this when the native signal had higher sampling rate than the
    model's target (e.g. LUDB 500 → 250, ISP 1000 → 250) so detection
    keeps full native temporal resolution while the downstream channel
    is at the model's input fs.
    """
    n_out = 0 if n_native_samples == 0 else int(round(n_native_samples * target_fs / fs_native))
    out = np.zeros(n_out, dtype=np.float32)
    qrs = np.asarray(qrs_native_indices, dtype=np.int64)
    if qrs.size == 0 or n_out == 0:
        return out
    qrs_target = np.floor(qrs.astype(np.float64) * (target_fs / fs_native)).astype(np.int64)
    qrs_target = qrs_target[(qrs_target >= 0) & (qrs_target < n_out)]
    if qrs_target.size:
        out[qrs_target] = 1.0
    return out


DEFAULT_WIDE_THRESHOLD_MS = 120.0


def qrs_box_resampled(
    qrs_on_native: np.ndarray,
    qrs_off_native: np.ndarray,
    fs_native: int,
    n_native_samples: int,
    target_fs: int = 250,
) -> np.ndarray:
    """Per-sample binary indicator at ``target_fs``: 1 inside every
    ``[qrs_on, qrs_off]`` interval, 0 elsewhere. Replaces the 1-sample
    spike convention (``qrs_pointwise_resampled``) when the model wants
    the full QRS region as a prior — patch max-pool then produces a
    "QRS-in-this-patch" boolean per patch, dense and reliable.

    Wide vs narrow QRS information is naturally absorbed into the box
    width — no separate wide channel is needed.
    """
    n_out = 0 if n_native_samples == 0 else int(round(n_native_samples * target_fs / fs_native))
    out = np.zeros(n_out, dtype=np.float32)
    on = np.asarray(qrs_on_native, dtype=np.int64)
    off = np.asarray(qrs_off_native, dtype=np.int64)
    if on.size == 0 or n_out == 0:
        return out
    for lo, hi in zip(on, off):
        lo_t = int(np.floor(lo * target_fs / fs_native))
        hi_t = int(np.ceil(hi * target_fs / fs_native))
        lo_t = max(0, lo_t)
        hi_t = min(n_out, hi_t + 1)
        if hi_t > lo_t:
            out[lo_t:hi_t] = 1.0
    return out


def qrs_wide_channel(
    sig: np.ndarray,
    fs_in: int,
    target_fs: int = 250,
    broaden_ms: float = DEFAULT_BROADEN_MS,
    width_threshold_ms: float = DEFAULT_WIDE_THRESHOLD_MS,
) -> np.ndarray:
    """Per-sample binary indicator: 1.0 wherever a detected QRS is "wide"
    (width >= ``width_threshold_ms``), 0 elsewhere. Output length matches
    the target-fs resampled signal.

    Designed as an input prior channel alongside ``qrs_position_channel``:
    the QRS channel says "where is a QRS", and this channel says "which of
    those QRS are wide" (BBB / PVC / paced). 120 ms is the clinical
    threshold separating normal from wide QRS.
    """
    sig = np.asarray(sig, dtype=np.float32)
    n_in = len(sig)
    n_out = 0 if n_in == 0 else int(round(n_in * target_fs / fs_in))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)

    try:
        from openecg.qrs import detect_qrs as _detect
        r_peaks, widths_ms = _detect(sig, fs_in, return_widths=True)
    except Exception:
        return np.zeros(n_out, dtype=np.float32)

    out = np.zeros(n_out, dtype=np.float32)
    if len(r_peaks) == 0:
        return out
    half = max(0, int(round(broaden_ms * target_fs / 1000 / 2)))
    r_target = np.round(np.asarray(r_peaks) * (target_fs / fs_in)).astype(np.int64)
    for r, w_ms in zip(r_target, widths_ms):
        if float(w_ms) < float(width_threshold_ms):
            continue
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
    instead of openecg.detect_qrs. ``broaden_ms = 0`` gives a strict
    1-sample point indicator at each QRS index.
    """
    n_out = 0 if n_in_samples == 0 else int(round(n_in_samples * target_fs / fs_in))
    out = np.zeros(n_out, dtype=np.float32)
    qrs_indices = np.asarray(qrs_indices, dtype=np.int64)
    if qrs_indices.size == 0 or n_out == 0:
        return out
    half = max(0, int(round(broaden_ms * target_fs / 1000 / 2)))
    qrs_target = np.round(qrs_indices * (target_fs / fs_in)).astype(np.int64)
    for q in qrs_target:
        lo = max(0, int(q) - half)
        hi = min(n_out, int(q) + half + 1)
        if hi > lo:
            out[lo:hi] = 1.0
    return out
