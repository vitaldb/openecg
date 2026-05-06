# tests/test_pacer.py
import numpy as np
from scipy import signal as scipy_signal

from openecg import pacer

FS = 500


def test_detects_synthetic_spike():
    n = FS * 5  # 5s
    sig = np.zeros(n)
    sig[1000] = 5.0   # spike at 2s, sample 1000
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 1
    assert abs(detected[0] - 1000) <= 1


def test_detects_negative_polarity_spike():
    n = FS * 5
    sig = np.zeros(n)
    sig[1500] = -5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 1


def test_ignores_qrs_like_wave():
    # 60ms hann window (R wave morphology) should not trigger
    sig = np.zeros(FS * 5)
    qrs = scipy_signal.windows.hann(30) * 1.5  # 60ms wide, amp 1.5
    sig[1000:1030] = qrs
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 0


def test_refractory_dedup_bipolar():
    # bipolar artifact (over+undershoot) within 5ms -> 1 detection
    sig = np.zeros(FS * 5)
    sig[1000] = 5.0
    sig[1001] = -3.0  # 2ms apart
    detected = pacer.detect_spikes(sig, fs=FS, refractory_ms=5.0)
    assert len(detected) == 1


def test_multiple_spikes_outside_refractory():
    # Signal length must accommodate index 2500 (FS * 5 = 2500 was OOB).
    sig = np.zeros(FS * 6)
    sig[500] = 5.0
    sig[1500] = 5.0
    sig[2500] = 5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 3


def test_returns_numpy_int_array():
    sig = np.zeros(FS * 5)
    sig[1000] = 5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert isinstance(detected, np.ndarray)
    assert np.issubdtype(detected.dtype, np.integer)


def test_no_spikes_in_pure_noise():
    rng = np.random.default_rng(seed=42)
    sig = rng.normal(0, 0.05, size=FS * 10)  # tiny noise
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 0
