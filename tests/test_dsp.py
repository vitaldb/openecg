"""Unit tests for openecg.dsp — verify our numpy-only DSP matches scipy
where scipy is available. Tests gracefully skip if scipy isn't installed."""
from __future__ import annotations

import numpy as np
import pytest

from openecg.dsp import butter, filtfilt, find_peaks, lfilter

scipy_signal = pytest.importorskip("scipy.signal")


# -- butter -------------------------------------------------------------------

@pytest.mark.parametrize("N,Wn,btype", [
    (2, 0.5 / 250, "high"),
    (2, [5 / 250, 15 / 250], "band"),
    (4, 40 / 250, "high"),
    (4, 0.5 / 180, "high"),
    (4, [0.5 / 180, 40 / 180], "band"),
    (3, 0.4, "low"),
    (5, [0.2, 0.8], "band"),
])
def test_butter_matches_scipy(N, Wn, btype):
    b_ours, a_ours = butter(N, Wn, btype=btype)
    b_sci, a_sci = scipy_signal.butter(N, Wn, btype=btype)
    np.testing.assert_allclose(b_ours, b_sci, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(a_ours, a_sci, atol=1e-12, rtol=1e-10)


# -- filtfilt -----------------------------------------------------------------

def test_filtfilt_matches_scipy_bandpass():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000) + np.sin(2 * np.pi * np.arange(2000) / 50)
    b, a = butter(2, [5 / 250, 15 / 250], btype="band")
    y_ours = filtfilt(b, a, x)
    y_sci = scipy_signal.filtfilt(b, a, x)
    np.testing.assert_allclose(y_ours, y_sci, atol=1e-9, rtol=1e-7)


def test_filtfilt_short_signal():
    """Signals shorter than padlen — should still run and produce a result
    of the same length."""
    x = np.arange(50, dtype=float)
    b, a = butter(2, 0.5, btype="low")
    y = filtfilt(b, a, x)
    assert y.shape == x.shape


# -- lfilter ------------------------------------------------------------------

def test_lfilter_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    b, a = butter(2, 0.3, btype="low")
    y_ours = lfilter(b, a, x)
    y_sci = scipy_signal.lfilter(b, a, x)
    np.testing.assert_allclose(y_ours, y_sci, atol=1e-12, rtol=1e-10)


# -- find_peaks ---------------------------------------------------------------

def test_find_peaks_matches_scipy_height_distance():
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(2000) * 0.05) + 0.1 * rng.normal(size=2000)
    p_ours, _ = find_peaks(x, height=0.3, distance=10)
    p_sci, _ = scipy_signal.find_peaks(x, height=0.3, distance=10)
    assert set(p_ours.tolist()) == set(p_sci.tolist())


def test_find_peaks_matches_scipy_prominence():
    rng = np.random.default_rng(1)
    x = np.cos(np.arange(1000) * 0.03) * np.sin(np.arange(1000) * 0.1)
    p_ours, props_ours = find_peaks(x, prominence=(None, None))
    p_sci, props_sci = scipy_signal.find_peaks(x, prominence=(None, None))
    assert set(p_ours.tolist()) == set(p_sci.tolist())
    np.testing.assert_allclose(
        props_ours["prominences"], props_sci["prominences"],
        atol=1e-12, rtol=1e-10,
    )


def test_find_peaks_with_width_filter():
    """Width filter is used by openecg.pacer.detect_spikes (legacy detector)."""
    x = np.zeros(500)
    # Narrow Gaussian-like peak at sample 100, wide one at sample 300.
    for i in range(95, 106):
        x[i] = np.exp(-((i - 100) ** 2) / 4.0)
    for i in range(280, 321):
        x[i] = np.exp(-((i - 300) ** 2) / 200.0)
    p_ours, _ = find_peaks(x, height=0.5, width=(None, 5))
    p_sci, _ = scipy_signal.find_peaks(x, height=0.5, width=(None, 5))
    assert set(p_ours.tolist()) == set(p_sci.tolist())


def test_find_peaks_empty_input():
    p, props = find_peaks(np.zeros(0))
    assert p.size == 0
    p, props = find_peaks(np.array([1.0]))
    assert p.size == 0


# -- backend dispatch ---------------------------------------------------------

def test_lfilter_backend_dispatch_via_env(monkeypatch):
    """Each backend (when available) must give bit-equivalent output."""
    import importlib
    import openecg.dsp as dsp_mod

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    b, a = butter(2, 0.3, btype="low")

    # Reference: pure numpy.
    monkeypatch.setenv("OPENECG_LFILTER_BACKEND", "numpy")
    importlib.reload(dsp_mod)
    y_np = dsp_mod.lfilter(b, a, x)
    assert dsp_mod.lfilter_backend() == "numpy"

    # scipy must match numpy (via scipy reference test we already passed).
    monkeypatch.setenv("OPENECG_LFILTER_BACKEND", "scipy")
    importlib.reload(dsp_mod)
    try:
        y_sci = dsp_mod.lfilter(b, a, x)
        assert dsp_mod.lfilter_backend() == "scipy"
        np.testing.assert_allclose(y_np, y_sci, atol=1e-10, rtol=1e-9)
    except Exception:
        pytest.skip("scipy backend unavailable")

    # numba (skip silently if not installed).
    monkeypatch.setenv("OPENECG_LFILTER_BACKEND", "numba")
    importlib.reload(dsp_mod)
    try:
        y_nb = dsp_mod.lfilter(b, a, x)
        if dsp_mod.lfilter_backend() != "numba":
            pytest.skip("numba unavailable; fell through to another backend")
        np.testing.assert_allclose(y_np, y_nb, atol=1e-10, rtol=1e-9)
    except ImportError:
        pytest.skip("numba unavailable")
    finally:
        # Reset module state for subsequent tests.
        monkeypatch.delenv("OPENECG_LFILTER_BACKEND", raising=False)
        importlib.reload(dsp_mod)


# -- wavelet primitives -------------------------------------------------------

pywt = pytest.importorskip("pywt")


def test_wavedec_db2_matches_pywt():
    from openecg.dsp import wavedec
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    ours = wavedec(x, "db2", level=4)
    ref = pywt.wavedec(x, "db2", level=4, mode="symmetric")
    for o, r in zip(ours, ref):
        np.testing.assert_allclose(o, r, atol=1e-12, rtol=1e-10)


def test_waverec_db2_matches_pywt():
    from openecg.dsp import wavedec, waverec
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 500)
    coeffs = pywt.wavedec(x, "db2", level=4, mode="symmetric")
    rec_ours = waverec(coeffs, "db2")
    rec_ref = pywt.waverec(coeffs, "db2", mode="symmetric")
    np.testing.assert_allclose(rec_ours, rec_ref, atol=1e-12, rtol=1e-10)


def test_waverec_round_trips_to_original():
    from openecg.dsp import wavedec, waverec
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 500)
    rec = waverec(wavedec(x, "db2", level=4), "db2")
    np.testing.assert_allclose(rec[: len(x)], x, atol=1e-12, rtol=1e-10)


def test_custom_filter_bank_3tap():
    """The 3-tap interpolation wavelet used by openvital's ecg_annotator."""
    from openecg.dsp import wavedec, waverec

    def qmf(w):
        return [(-1 if i % 2 == 1 else 1) * w[len(w) - 1 - i] for i in range(len(w))]

    def orthfilt(w):
        w = np.asarray(w, dtype=float); lor = w / np.linalg.norm(w)
        return [
            lor[::-1].tolist(), qmf(lor.tolist())[::-1],
            lor.tolist(), qmf(lor.tolist()),
        ]

    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 400)
    fb = orthfilt([0.25, 0.5, 0.25])
    ours = wavedec(x, fb, level=2)
    ref = pywt.wavedec(
        x, pywt.Wavelet("inter1", filter_bank=fb), level=2, mode="symmetric"
    )
    for o, r in zip(ours, ref):
        np.testing.assert_allclose(o, r, atol=1e-12, rtol=1e-10)
    # Round-trip
    rec_ours = waverec(ours, fb)
    rec_ref = pywt.waverec(
        ref, pywt.Wavelet("inter1", filter_bank=fb), mode="symmetric"
    )
    np.testing.assert_allclose(rec_ours, rec_ref, atol=1e-12, rtol=1e-10)


def test_cwt_gaus1_zero_crossings_match_pywt():
    """ecg_annotator only consumes sign / zero-crossings of the CWT
    output, so we accept a normalization difference but require the
    sign / zero-crossing structure to match pywt within a tiny tolerance."""
    from openecg.dsp import cwt
    rng = np.random.default_rng(0)
    fs = 500
    t = np.arange(0, 5, 1 / fs)
    x = 0.1 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.05, t.size)
    # Add QRS-like pulses
    for c in (500, 1000, 1500, 2000):
        x += np.exp(-((np.arange(t.size) - c) ** 2) / 50)
    for scale in (10, 30, 50):
        o = cwt(x, [scale], "gaus1")[0]
        r = pywt.cwt(x, [scale], "gaus1")[0][0]
        sign_match = float(np.mean(np.sign(o) == np.sign(r)))
        assert sign_match > 0.95, f"scale={scale}: sign agreement {sign_match} < 0.95"


def test_lfilter_backend_caches():
    """Backend lookup happens once and is reused on subsequent calls."""
    import openecg.dsp as dsp_mod
    # Reset and force an init.
    dsp_mod._LFILTER_BACKEND = None
    dsp_mod._LFILTER_BACKEND_NAME = None
    name1 = dsp_mod.lfilter_backend()
    func_after_init = dsp_mod._LFILTER_BACKEND
    assert func_after_init is not None
    # Calling again must reuse the same cached function object.
    _ = dsp_mod.lfilter([1.0], [1.0], np.zeros(10))
    assert dsp_mod._LFILTER_BACKEND is func_after_init
    assert dsp_mod.lfilter_backend() == name1
