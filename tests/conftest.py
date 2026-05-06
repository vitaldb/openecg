# tests/conftest.py
import numpy as np
import pytest

from openecg.delineate import DelineateResult


def _arr(*xs):
    return np.array(xs, dtype=float)


@pytest.fixture
def empty_dr():
    return DelineateResult.empty()


@pytest.fixture
def one_beat_normal_dr():
    """One normal beat: P at 100-150, Q at 195-205, R at 195-230, S at 220-230, T at 280-400.
    Sample indices (500Hz). Single beat -> arrays length 1."""
    return DelineateResult(
        p_onsets=_arr(100),
        p_peaks=_arr(125),
        p_offsets=_arr(150),
        q_peaks=_arr(200),
        r_onsets=_arr(195),
        r_peaks=_arr(210),
        r_offsets=_arr(230),
        s_peaks=_arr(225),
        t_onsets=_arr(280),
        t_peaks=_arr(330),
        t_offsets=_arr(400),
    )


@pytest.fixture
def one_beat_wide_no_qs_dr():
    """One beat with wide R only (LBBB pattern): R 200-280 (160ms wide), no Q, no S."""
    return DelineateResult(
        p_onsets=_arr(100),
        p_peaks=_arr(125),
        p_offsets=_arr(150),
        q_peaks=_arr(np.nan),
        r_onsets=_arr(200),
        r_peaks=_arr(240),
        r_offsets=_arr(280),
        s_peaks=_arr(np.nan),
        t_onsets=_arr(330),
        t_peaks=_arr(380),
        t_offsets=_arr(450),
    )


@pytest.fixture
def one_beat_narrow_no_qs_dr():
    """One beat with narrow R only (V1 lead): R 200-230 (60ms wide), no Q, no S."""
    return DelineateResult(
        p_onsets=_arr(100),
        p_peaks=_arr(125),
        p_offsets=_arr(150),
        q_peaks=_arr(np.nan),
        r_onsets=_arr(200),
        r_peaks=_arr(215),
        r_offsets=_arr(230),
        s_peaks=_arr(np.nan),
        t_onsets=_arr(280),
        t_peaks=_arr(330),
        t_offsets=_arr(400),
    )
