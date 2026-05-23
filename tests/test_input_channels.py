"""Unit tests for openecg.stage2.input_channels.compose."""
import numpy as np
import pytest

from openecg.stage2.input_channels import (
    channel_names, compute_input_channels,
)


def test_names():
    assert channel_names(False, False) == ("signal",)
    assert channel_names(True, False)  == ("signal", "pacer")
    assert channel_names(False, True)  == ("signal", "qrs")
    assert channel_names(True, True)   == ("signal", "pacer", "qrs")


def test_signal_only_passthrough():
    sig = np.arange(100, dtype=np.float32) / 100
    out = compute_input_channels(sig, target_fs=250,
                                  with_pacer=False, with_qrs=False)
    assert out.shape == (1, 100)
    assert out[0].tolist() == sig.tolist()


def test_pacer_only_two_channels_zeros_when_clean():
    """A pure-zero signal should have a zero pacer channel."""
    sig_t = np.zeros(2500, dtype=np.float32)
    sig_n = np.zeros(5000, dtype=np.float32)
    out = compute_input_channels(
        sig_t, sig_native=sig_n, fs_native=500, target_fs=250,
        with_pacer=True, with_qrs=False,
    )
    assert out.shape == (2, 2500)
    assert (out[1] == 0).all()


def test_detect_pacings_mode_three_channels():
    """detect_pacings mode produces a [2, T] tensor of the right shape and
    leaves the second channel zero on a clean signal."""
    sig_t = np.zeros(2500, dtype=np.float32)
    sig_n = np.zeros(5000, dtype=np.float32)
    out = compute_input_channels(
        sig_t, sig_native=sig_n, fs_native=500, target_fs=250,
        with_pacer=True, with_qrs=False, pacer_mode="detect_pacings",
    )
    assert out.shape == (2, 2500)
    assert (out[1] == 0).all()


def test_pacer_and_qrs_three_channels_with_qrs_indices():
    sig_t = np.zeros(2500, dtype=np.float32)
    sig_n = np.zeros(5000, dtype=np.float32)
    qrs = np.array([1000, 2500, 4000], dtype=np.int64)
    out = compute_input_channels(
        sig_t, sig_native=sig_n, fs_native=500, target_fs=250,
        with_pacer=True, with_qrs=True,
        qrs_indices_native=qrs,
    )
    assert out.shape == (3, 2500)
    # Pacer all-zero (clean signal).
    assert (out[1] == 0).all()
    # QRS channel saturates at indicated positions.
    for q in qrs:
        t = q // 2
        win = out[2, max(0, t - 5):min(2500, t + 5)]
        assert win.max() == pytest.approx(1.0)
