# openecg/pacer.py
"""Pacemaker spike detector - highpass + adaptive threshold.

Spike characteristics: 1-3ms wide, high amplitude (>1mV), sharp slope.
ECG content (<50Hz) suppressed by 80Hz highpass; spike survives.

Spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md §8
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detect_spikes(
    signal: np.ndarray,
    fs: int = 500,
    cutoff_hz: float = 80.0,
    amp_threshold_mad: float = 5.0,
    max_width_ms: float = 4.0,
    refractory_ms: float = 5.0,
    min_amplitude: float = 0.1,
) -> np.ndarray:
    """Detect pacemaker spike sample indices.

    Args:
        signal: 1D ECG samples (assumed in mV for default ``min_amplitude``).
        fs: sampling rate in Hz
        cutoff_hz: highpass cutoff (default 80Hz, suppresses ECG content)
        amp_threshold_mad: peak height threshold = N x robust sigma
            (sigma = 1.4826 * MAD).
        max_width_ms: reject peaks wider than this (likely R waves)
        refractory_ms: dedup window for bipolar artifacts
        min_amplitude: absolute floor on peak height; prevents spurious
            detections from filter ringing when MAD ~= 0.

    Returns:
        Sorted array of sample indices where spikes were detected.
    """
    nyq = fs / 2
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    hp = filtfilt(b, a, signal)
    abs_hp = np.abs(hp)

    # Robust noise estimate: 1.4826 * MAD approximates sigma of Gaussian noise.
    mad = np.median(np.abs(hp - np.median(hp)))
    sigma = 1.4826 * mad
    threshold = max(amp_threshold_mad * sigma, min_amplitude)

    refractory_samples = max(1, int(refractory_ms * fs / 1000))
    # The highpass impulse response rings for ~1/cutoff_hz seconds; use this
    # as the minimum dedup distance so ringing shoulders are suppressed.
    ringing_samples = max(1, int(fs / cutoff_hz))
    dedup_distance = max(refractory_samples, ringing_samples)
    # Filter dispersion widens narrow peaks; allow extra width for that so
    # the ``max_width_ms`` parameter refers to raw spike width, not filtered.
    filter_dispersion = max(1, int(fs / cutoff_hz / 2))
    max_width_samples = max(1, int(max_width_ms * fs / 1000)) + filter_dispersion

    peaks, _ = find_peaks(
        abs_hp,
        height=threshold,
        distance=dedup_distance,
        width=(None, max_width_samples),
    )
    return peaks.astype(np.int64)
