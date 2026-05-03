# tests/test_labeler.py
import numpy as np

from ecgcode import labeler, vocab

FS = 500
N_SAMPLES = 500  # 1 second @ 500Hz


def _syms(events):
    return [s for s, _ in events]


def test_normal_qrs_emits_q_r_s(one_beat_normal_dr):
    events = labeler.label(one_beat_normal_dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    # Expected sequence: iso, p, iso, q, r, s, iso, t, iso
    assert syms == [
        vocab.ID_ISO, vocab.ID_P, vocab.ID_ISO,
        vocab.ID_Q, vocab.ID_R, vocab.ID_S,
        vocab.ID_ISO, vocab.ID_T, vocab.ID_ISO,
    ]


def test_wide_qrs_no_q_no_s_emits_w(one_beat_wide_no_qs_dr):
    events = labeler.label(one_beat_wide_no_qs_dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    assert vocab.ID_W in syms
    assert vocab.ID_R not in syms
    assert vocab.ID_Q not in syms
    assert vocab.ID_S not in syms


def test_narrow_qrs_no_q_no_s_emits_r_not_w(one_beat_narrow_no_qs_dr):
    events = labeler.label(one_beat_narrow_no_qs_dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    assert vocab.ID_R in syms
    assert vocab.ID_W not in syms


def test_nk_total_failure_returns_unk(empty_dr):
    """0 R peaks -> entire signal is one ? event."""
    events = labeler.label(empty_dr, [], n_samples=5000, fs=FS)
    assert events == [(vocab.ID_UNK, 10000)]   # 5000 samples * 2ms = 10000ms


def test_spike_in_normal_iso_region(one_beat_normal_dr):
    """Spike at sample 50 (in iso region before P) -> wave-*-wave split."""
    events = labeler.label(one_beat_normal_dr, [50], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    # iso (50 samples) -> spike (1 sample) -> iso (rest of pre-P) -> ...
    assert syms[0] == vocab.ID_ISO
    assert syms[1] == vocab.ID_PACER
    assert syms[2] == vocab.ID_ISO


def test_total_duration_matches_n_samples(one_beat_normal_dr):
    events = labeler.label(one_beat_normal_dr, [], n_samples=N_SAMPLES, fs=FS)
    total_ms = sum(ms for _, ms in events)
    assert total_ms == N_SAMPLES * 1000 // FS  # 1000ms


def test_boundary_clamp(one_beat_normal_dr):
    """t_offsets beyond signal length -> clamped, no crash."""
    one_beat_normal_dr.t_offsets[0] = 600  # beyond 500-sample signal
    events = labeler.label(one_beat_normal_dr, [], n_samples=N_SAMPLES, fs=FS)
    total_ms = sum(ms for _, ms in events)
    assert total_ms == 1000  # still exactly 1 second


def test_wide_qrs_threshold_at_120ms(one_beat_normal_dr):
    """QRS exactly 120ms with no Q/S -> still r (threshold is strict >120)."""
    dr = one_beat_normal_dr
    dr.q_peaks[0] = np.nan
    dr.s_peaks[0] = np.nan
    dr.r_onsets[0] = 200
    dr.r_offsets[0] = 200 + 60  # 60 samples = 120ms exactly (with on-inclusive count = 61 samples)
    events = labeler.label(dr, [], n_samples=N_SAMPLES, fs=FS)
    syms = _syms(events)
    assert vocab.ID_R in syms
    assert vocab.ID_W not in syms
