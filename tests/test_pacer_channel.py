"""Unit tests for openecg.stage2.pacer_channel."""
import numpy as np
import pytest

from openecg.stage2.pacer_channel import (
    pacer_channel_from_signal, pacer_detect_pacings_channel,
)


def test_empty():
    out = pacer_channel_from_signal(np.zeros(0, dtype=np.float32),
                                     fs_in=500, target_fs=250)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_zero_signal_zero_channel():
    n = 5000
    out = pacer_channel_from_signal(np.zeros(n, dtype=np.float32),
                                     fs_in=500, target_fs=250)
    assert out.shape == (n // 2,)
    assert (out == 0).all()


def test_synthetic_spike_detected():
    """Place a clear bipolar 2 ms spike inside Gaussian noise; the
    indicator should peak somewhere within 30 ms of the planted center."""
    rng = np.random.default_rng(0)
    fs_in = 500
    n = 5000
    sig = rng.normal(0, 0.01, size=n).astype(np.float32)
    center = 2500
    half_w = 1                                              # ~2 ms total
    sig[center - half_w:center] += 2.0
    sig[center:center + half_w + 1] -= 2.0
    out = pacer_channel_from_signal(sig, fs_in=fs_in, target_fs=250)
    target_center = center // 2
    win_lo = max(0, target_center - 8)                     # 32 ms tolerance
    win_hi = min(len(out), target_center + 8)
    assert out[win_lo:win_hi].max() == pytest.approx(1.0)


def test_lengths_match_target_fs():
    n = 1000
    sig = np.zeros(n, dtype=np.float32)
    out_500 = pacer_channel_from_signal(sig, fs_in=500, target_fs=250)
    assert out_500.shape == (500,)
    out_360 = pacer_channel_from_signal(sig, fs_in=360, target_fs=250)
    assert out_360.shape == (round(n * 250 / 360),)
    out_1000 = pacer_channel_from_signal(sig, fs_in=1000, target_fs=250)
    assert out_1000.shape == (250,)


def test_detect_pacings_channel_zero_signal():
    out = pacer_detect_pacings_channel(np.zeros(5000, dtype=np.float32),
                                       fs_in=500, target_fs=250)
    assert out.shape == (2500,)
    assert out.dtype == np.float32
    assert (out == 0).all()


def test_detect_pacings_channel_synthetic_spike_train():
    """Plant a 4-spike train in noise; the canonical detect_pacings-driven
    channel must fire (>=1 saturated boxcar) somewhere in the window."""
    rng = np.random.default_rng(0)
    fs_in = 500
    n = 5000
    sig = rng.normal(0, 0.01, size=n).astype(np.float64)
    for center in (1000, 2000, 3000, 4000):
        sig[center - 1] += 3.0
        sig[center + 0] -= 3.0
    out = pacer_detect_pacings_channel(sig, fs_in=fs_in, target_fs=250)
    assert out.shape == (2500,)
    assert out.max() == pytest.approx(1.0)


def test_low_fs_input_upsamples():
    """fs_in=250 (<= detect_fs default) should still detect a planted spike
    via the internal upsample-to-500 path."""
    rng = np.random.default_rng(1)
    fs_in = 250
    n = 2500
    sig = rng.normal(0, 0.01, size=n).astype(np.float32)
    center = 1200
    sig[center] = 3.0
    sig[center + 1] = -3.0
    out = pacer_channel_from_signal(sig, fs_in=fs_in, target_fs=250)
    assert out.shape == (n,)
    win_lo = max(0, center - 10)
    win_hi = min(n, center + 10)
    assert out[win_lo:win_hi].max() == pytest.approx(1.0)
