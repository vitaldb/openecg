import numpy as np

from ecgcode.stage2.refiner import refine_boundaries, refine_boundary


def test_refine_qrs_on_moves_to_derivative_onset():
    sig = np.zeros(120, dtype=np.float32)
    sig[50:61] = np.linspace(0.0, 2.0, 11)
    sig[61:70] = 2.0
    refined = refine_boundary(sig, boundary_sample=58, boundary_key="qrs_on", fs=250, search_ms=80)
    assert 49 <= refined <= 53


def test_refine_qrs_off_moves_to_derivative_offset():
    sig = np.zeros(140, dtype=np.float32)
    sig[50:61] = np.linspace(0.0, 2.0, 11)
    sig[61:70] = 2.0
    sig[70:81] = np.linspace(2.0, 0.0, 11)
    refined = refine_boundary(sig, boundary_sample=72, boundary_key="qrs_off", fs=250, search_ms=80)
    assert 78 <= refined <= 82


def test_refine_boundaries_defaults_to_qrs_only():
    sig = np.zeros(120, dtype=np.float32)
    sig[50:61] = np.linspace(0.0, 2.0, 11)
    boundaries = {"p_on": [20], "qrs_on": [58], "qrs_off": []}
    out = refine_boundaries(sig, boundaries, fs=250, search_ms=80)
    assert out["p_on"] == [20]
    assert 49 <= out["qrs_on"][0] <= 53
    assert out["qrs_off"] == []
