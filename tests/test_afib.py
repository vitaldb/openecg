"""Unit tests for openecg.is_afib."""
from __future__ import annotations

import numpy as np
import pytest

from openecg import is_afib, afib_score


FS = 500


def _synth_window(rr_ms_list, qrs_width_samples=40, amp=1.0, fs=FS,
                  total_s=10.0, rng_seed=0):
    """Construct a synthetic 10s ECG with R-peaks at the given RR offsets."""
    rng = np.random.default_rng(rng_seed)
    n = int(total_s * fs)
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * np.arange(n) / fs)
    sig += rng.normal(0, 0.01, size=n)
    qrs = np.hanning(qrs_width_samples) * amp
    # First R at 0.5 s, then walk by rr_ms_list
    t = int(0.5 * fs)
    for rr in rr_ms_list:
        lo = t - qrs_width_samples // 2
        if lo < 0 or lo + qrs_width_samples >= n:
            break
        sig[lo:lo + qrs_width_samples] += qrs
        t += int(rr * fs / 1000)
    # final beat
    if t - qrs_width_samples // 2 >= 0 and t + qrs_width_samples // 2 < n:
        sig[t - qrs_width_samples // 2:t + qrs_width_samples // 2] += qrs
    return sig.astype(np.float64)


def test_nsr_regular_not_afib():
    """Uniform RR ≈ 800 ms NSR window should not trigger AFib."""
    sig = _synth_window([800] * 10)
    assert is_afib(sig, FS) is False


def test_brady_regular_not_afib():
    """Sinus brady (uniform RR 1200 ms) — not AFib."""
    sig = _synth_window([1200] * 7)
    assert is_afib(sig, FS) is False


def test_chaotic_rr_fires_afib():
    """Highly irregular RR pattern (AFib-like) should fire."""
    afib_rrs = [620, 920, 540, 1100, 780, 480, 1240, 700, 880, 560]
    sig = _synth_window(afib_rrs)
    assert is_afib(sig, FS) is True


def test_score_shape():
    sig = _synth_window([800] * 10)
    out = afib_score(sig, FS)
    assert isinstance(out, dict)
    assert {"is_afib", "reason", "n_beats", "rr_ms", "widths_ms",
            "vetoed", "main_fire", "safety_fire"}.issubset(out.keys())


def test_too_few_beats_safe():
    """Edge case: only 2 R-peaks → no decision possible, return False."""
    sig = _synth_window([800, 850])
    out = afib_score(sig, FS)
    assert out["is_afib"] is False
    assert "insufficient" in out["reason"]


def test_wide_qrs_window_is_vetoed():
    """When most beats have wide QRS (sustained VT/escape), veto fires."""
    # 12 beats all with very wide QRS (~160 ms wide hanning)
    wide_qrs_samples = int(0.160 * FS)
    sig = _synth_window([700] * 12, qrs_width_samples=wide_qrs_samples,
                        amp=2.0)
    out = afib_score(sig, FS)
    # Should be vetoed (most beats wide) and not called AFib.
    assert out["is_afib"] is False
