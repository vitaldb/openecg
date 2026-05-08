"""Unit tests for openecg.qrs.detect_qrs."""
from __future__ import annotations

import numpy as np
import pytest

from openecg import detect_qrs


FS = 500


def _synthesize(rr_ms: int = 800, n_beats: int = 8, fs: int = FS,
                inverted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Build a clean sinusoidal-baseline ECG with QRS-like spikes at
    fixed RR intervals. Returns (signal, ground-truth R-peak indices)."""
    rng = np.random.default_rng(seed=0)
    n = int(rr_ms * (n_beats + 1) * fs / 1000)
    t = np.arange(n) / fs
    # Slow baseline + small noise.
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.01, size=n)
    # Hann-shaped QRS pulses ~80 ms wide, amplitude 1.0.
    qrs_w = np.hanning(int(0.080 * fs))
    centers = np.array([int((i + 1) * rr_ms * fs / 1000)
                        for i in range(n_beats)])
    for c in centers:
        lo = c - len(qrs_w) // 2
        sig[lo:lo + len(qrs_w)] += qrs_w
    if inverted:
        sig = -sig
    return sig.astype(np.float64), centers


def _match_count(gt: np.ndarray, det: np.ndarray, tol_ms: float = 100.0,
                 fs: int = FS) -> int:
    """Greedy 1-1 match within tol_ms ms. Same as the MIT-BIH validator."""
    tol = int(round(tol_ms * fs / 1000))
    used = np.zeros(det.size, dtype=bool)
    matched = 0
    j_start = 0
    gt_sorted = np.sort(gt); det_sorted = np.sort(det)
    for g in gt_sorted:
        while j_start < det_sorted.size and det_sorted[j_start] < g - tol:
            j_start += 1
        for j in range(j_start, det_sorted.size):
            if det_sorted[j] > g + tol: break
            if not used[j]:
                used[j] = True; matched += 1; break
    return matched


def test_clean_synthetic_finds_all_beats():
    sig, gt = _synthesize()
    det = detect_qrs(sig, fs=FS)
    matched = _match_count(gt, det)
    assert matched == len(gt)


def test_inverted_signal_works():
    """Negative-polarity R-peaks (e.g. aVR) should still be detected."""
    sig, gt = _synthesize(inverted=True)
    det = detect_qrs(sig, fs=FS)
    assert _match_count(gt, det) == len(gt)


def test_empty_input_returns_empty():
    assert detect_qrs(np.zeros(0), fs=FS).size == 0


def test_nan_only_returns_empty():
    assert detect_qrs(np.full(1000, np.nan), fs=FS).size == 0


def test_nan_interpolation():
    """NaNs in the middle should be interpolated, not crash."""
    sig, gt = _synthesize()
    sig_nan = sig.copy()
    sig_nan[100:120] = np.nan
    det = detect_qrs(sig_nan, fs=FS)
    # All ground-truth beats matched — the NaNs sit on a baseline
    # segment (not on a QRS), so interpolation should preserve detection.
    assert _match_count(gt, det) == len(gt)


def test_too_short_signal_returns_empty():
    short = np.random.default_rng(0).normal(0, 1, size=10)
    assert detect_qrs(short, fs=FS).size == 0


def test_pure_noise_below_threshold():
    """Pure noise: detector should not return many false positives.

    The gradient-threshold method always finds *some* peaks because
    the threshold is a multiple of the rolling mean (so even white
    noise produces above-threshold regions). We just check the count
    stays well below the per-second beat count of a normal heart rate.
    """
    rng = np.random.default_rng(seed=42)
    n = 5 * FS
    sig = rng.normal(0, 1e-4, size=n)  # microvolt-level noise
    det = detect_qrs(sig, fs=FS)
    # 5s @ 60bpm = 5 beats; require fewer FPs than 1 per second.
    assert det.size < 5


def test_input_dtype_robustness():
    """Verify the detector also works on float32 / list inputs."""
    sig, gt = _synthesize()
    det_f32 = detect_qrs(sig.astype(np.float32), fs=FS)
    det_list = detect_qrs(sig.tolist(), fs=FS)
    assert _match_count(gt, det_f32) == len(gt)
    assert _match_count(gt, det_list) == len(gt)


def test_returns_int64():
    sig, _ = _synthesize()
    det = detect_qrs(sig, fs=FS)
    assert det.dtype == np.int64


@pytest.mark.parametrize("fs", [250, 360, 500, 1000])
def test_fs_invariance(fs):
    """All ground-truth beats found across the typical clinical fs range."""
    sig, gt = _synthesize(fs=fs)
    det = detect_qrs(sig, fs=fs)
    assert _match_count(gt, det, fs=fs) == len(gt)
