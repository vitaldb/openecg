# ecgcode/stage2/augment.py
"""ECG-specific augmentations per SemiSegECG (arXiv 2507.18323).

Apply during training only. Skip horizontal flip and baseline shift (paper warns against).
"""

import numpy as np


def powerline_noise(sig, fs=250, freq=50.0, amplitude=0.05, rng=None):
    """Add 50Hz (or 60Hz) sinusoidal interference."""
    rng = rng or np.random.default_rng()
    t = np.arange(len(sig)) / fs
    phase = float(rng.uniform(0, 2 * np.pi))
    return sig + amplitude * np.sin(2 * np.pi * freq * t + phase)


def sine_noise(sig, fs=250, freq_range=(2.0, 30.0), amplitude=0.03, rng=None):
    """Random low-frequency sine component."""
    rng = rng or np.random.default_rng()
    freq = float(rng.uniform(*freq_range))
    t = np.arange(len(sig)) / fs
    phase = float(rng.uniform(0, 2 * np.pi))
    return sig + amplitude * np.sin(2 * np.pi * freq * t + phase)


def white_noise(sig, sigma=0.05, rng=None):
    rng = rng or np.random.default_rng()
    return sig + rng.normal(0, sigma, size=sig.shape).astype(sig.dtype)


def amplitude_scaling(sig, scale_range=0.2, rng=None):
    rng = rng or np.random.default_rng()
    scale = 1.0 + float(rng.uniform(-scale_range, scale_range))
    return sig * scale


def randaugment_ecg(sig, fs=250, n_ops=2, rng=None):
    """Apply n_ops randomly chosen ECG augmentations from the safe set."""
    rng = rng or np.random.default_rng()
    ops = [
        lambda s: powerline_noise(s, fs=fs, freq=float(rng.choice([50.0, 60.0])),
                                  amplitude=float(rng.uniform(0.02, 0.08)), rng=rng),
        lambda s: sine_noise(s, fs=fs, amplitude=float(rng.uniform(0.01, 0.05)), rng=rng),
        lambda s: white_noise(s, sigma=float(rng.uniform(0.02, 0.08)), rng=rng),
        lambda s: amplitude_scaling(s, scale_range=float(rng.uniform(0.1, 0.3)), rng=rng),
    ]
    rng.shuffle(ops)
    out = sig.astype(np.float32)
    for op in ops[:n_ops]:
        out = op(out).astype(np.float32)
    return out
