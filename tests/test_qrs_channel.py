"""Unit tests for openecg.stage2.qrs_channel."""
import numpy as np
import pytest

from openecg.stage2.qrs_channel import (
    qrs_position_channel, qrs_position_channel_from_indices,
)


def test_empty():
    out = qrs_position_channel(np.zeros(0, dtype=np.float32),
                                fs_in=500, target_fs=250)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_zero_signal_zero_channel():
    out = qrs_position_channel(np.zeros(5000, dtype=np.float32),
                                fs_in=500, target_fs=250)
    assert out.shape == (2500,)
    assert (out == 0).all()


def test_indices_path_boxcar():
    """Manually pass QRS indices and verify boxcar shape at target_fs."""
    n_in = 5000
    qrs = np.array([1000, 2500, 4000], dtype=np.int64)
    out = qrs_position_channel_from_indices(
        qrs, fs_in=500, n_in_samples=n_in, target_fs=250, broaden_ms=40,
    )
    assert out.shape == (2500,)
    # For each qrs at 500 Hz idx q, target idx = q // 2 ~= q*0.5.
    for q in qrs:
        target = round(q * 250 / 500)
        # broaden 40 ms at 250 Hz = 10 samples total, half=5.
        assert out[target] == 1.0
        for off in (-3, -1, 1, 3):
            assert out[target + off] == 1.0
    # Far from QRS, channel = 0.
    assert out[100] == 0.0
    assert out[2400] == 0.0


def test_lengths_match():
    sig = np.zeros(2000, dtype=np.float32)
    out_500 = qrs_position_channel(sig, fs_in=500, target_fs=250)
    assert out_500.shape == (1000,)
    out_360 = qrs_position_channel(sig, fs_in=360, target_fs=250)
    assert out_360.shape == (round(2000 * 250 / 360),)


def test_real_detection_roundtrip():
    """Plant clear QRS-shaped peaks and verify the QRS detector finds them."""
    fs = 500
    n = 5000
    sig = np.zeros(n, dtype=np.float32)
    qrs_at = np.array([500, 1500, 2500, 3500, 4500], dtype=np.int64)
    # Triangle pulse mimicking a QRS, ~50 ms wide.
    half_w = int(0.025 * fs)
    for q in qrs_at:
        for i in range(-half_w, half_w + 1):
            if 0 <= q + i < n:
                sig[q + i] += (1.0 - abs(i) / half_w)
    out = qrs_position_channel(sig, fs_in=fs, target_fs=250)
    assert out.shape == (2500,)
    # Each planted QRS should turn on the channel within 40 ms.
    for q in qrs_at:
        target = q // 2
        win = out[max(0, target - 10):min(2500, target + 10)]
        assert win.max() == pytest.approx(1.0)
