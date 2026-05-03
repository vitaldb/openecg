import numpy as np

from ecgcode.stage2.augment import (
    amplitude_scaling,
    powerline_noise,
    randaugment_ecg,
    sine_noise,
    white_noise,
)


def test_powerline_changes_signal():
    sig = np.zeros(2500, dtype=np.float32)
    out = powerline_noise(sig, fs=250, freq=50.0, amplitude=0.05)
    assert np.abs(out).max() > 0.04


def test_white_noise_zero_mean():
    sig = np.zeros(2500, dtype=np.float32)
    rng = np.random.default_rng(0)
    out = white_noise(sig, sigma=0.05, rng=rng)
    assert abs(out.mean()) < 0.01


def test_amplitude_scaling_range():
    sig = np.ones(2500, dtype=np.float32)
    rng = np.random.default_rng(0)
    out = amplitude_scaling(sig, scale_range=0.2, rng=rng)
    assert 0.8 <= float(out[0]) <= 1.2


def test_randaugment_changes_signal():
    rng = np.random.default_rng(42)
    sig = np.zeros(2500, dtype=np.float32)
    out = randaugment_ecg(sig, fs=250, n_ops=2, rng=rng)
    assert np.abs(out).max() > 0.0
    assert out.dtype == np.float32
    assert out.shape == sig.shape


def test_sine_noise_changes_signal():
    sig = np.zeros(2500, dtype=np.float32)
    rng = np.random.default_rng(1)
    out = sine_noise(sig, fs=250, amplitude=0.03, rng=rng)
    assert np.abs(out).max() > 0.0
