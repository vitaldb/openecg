# tests/test_pacer.py
import numpy as np

from openecg import pacer

FS = 500


def test_detects_synthetic_spike():
    n = FS * 5  # 5s
    sig = np.zeros(n)
    sig[1000] = 5.0   # spike at 2s, sample 1000
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 1
    assert abs(detected[0] - 1000) <= 1


def test_detects_negative_polarity_spike():
    n = FS * 5
    sig = np.zeros(n)
    sig[1500] = -5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 1


def test_ignores_qrs_like_wave():
    # 60ms hann window (R wave morphology) should not trigger
    sig = np.zeros(FS * 5)
    qrs = np.hanning(30) * 1.5  # 60ms wide, amp 1.5
    sig[1000:1030] = qrs
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 0


def test_refractory_dedup_bipolar():
    # bipolar artifact (over+undershoot) within 5ms -> 1 detection
    sig = np.zeros(FS * 5)
    sig[1000] = 5.0
    sig[1001] = -3.0  # 2ms apart
    detected = pacer.detect_spikes(sig, fs=FS, refractory_ms=5.0)
    assert len(detected) == 1


def test_multiple_spikes_outside_refractory():
    # Signal length must accommodate index 2500 (FS * 5 = 2500 was OOB).
    sig = np.zeros(FS * 6)
    sig[500] = 5.0
    sig[1500] = 5.0
    sig[2500] = 5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 3


def test_returns_numpy_int_array():
    sig = np.zeros(FS * 5)
    sig[1000] = 5.0
    detected = pacer.detect_spikes(sig, fs=FS)
    assert isinstance(detected, np.ndarray)
    assert np.issubdtype(detected.dtype, np.integer)


def test_no_spikes_in_pure_noise():
    rng = np.random.default_rng(seed=42)
    sig = rng.normal(0, 0.05, size=FS * 10)  # tiny noise
    detected = pacer.detect_spikes(sig, fs=FS)
    assert len(detected) == 0


# -- center-surround filter ---------------------------------------------------

def _signal_with_qrs_and_spike(fs=FS, n_secs=4, noise_mV=0.01, seed=0):
    """Synthetic ECG: 3 hann-window QRSs (~80 ms wide) plus baseline noise
    plus a 1-2 ms bipolar pacer spike just before one of them. The noise
    floor matches what real ECG amplifiers produce (~5-20 µV) and keeps
    MAD-based thresholds well-defined."""
    rng = np.random.default_rng(seed)
    n = fs * n_secs
    sig = rng.normal(0.0, noise_mV, size=n)
    qrs_w = np.hanning(int(0.080 * fs))     # 80 ms QRS
    qrs_amp = 1.0
    qrs_centers = [int(0.5 * fs), int(1.5 * fs), int(2.5 * fs)]
    for c in qrs_centers:
        lo = c - len(qrs_w) // 2
        sig[lo:lo + len(qrs_w)] += qrs_amp * qrs_w
    spike_at = int(1.45 * fs)                              # 50 ms before R2
    sig[spike_at] += 4.0                                   # +4 mV up
    sig[spike_at + 1] += -2.0                              # -2 mV down  (bipolar 2 ms)
    return sig, np.array(qrs_centers, dtype=np.int64), spike_at


def test_center_surround_score_pos_at_spike_neg_at_qrs():
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    score = pacer.pacer_center_surround_score(
        sig, fs=FS, center_ms=2.0, surround_ms=12.0, penalty=2.0,
    )
    # At the spike, score must be high.
    spike_score = score[spike_at - 1: spike_at + 3].max()
    # At the QRS center, score must be near zero or negative.
    qrs_score = score[qrs[0] - 5: qrs[0] + 5].max()
    assert spike_score > 5 * abs(qrs_score), (
        f"spike {spike_score:.4f} should dominate qrs {qrs_score:.4f}"
    )


def test_center_surround_detects_synthetic_spike_only():
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    det = pacer.detect_spikes_center_surround(
        sig, fs=FS, center_ms=2.0, surround_ms=12.0, penalty=2.0,
    )
    # Exactly the spike (+- a few samples), no QRS false-positives.
    assert det.size >= 1
    near_spike = np.abs(det - spike_at) <= 2
    assert near_spike.any(), f"spike not detected: got {det}, expected near {spike_at}"
    # No detection inside any 30 ms QRS window.
    half_qrs = int(0.020 * FS)
    for q in qrs:
        in_qrs = (det >= q - half_qrs) & (det <= q + half_qrs)
        assert not in_qrs.any(), f"false positive inside QRS at {q}: {det[in_qrs]}"


def test_center_surround_score_zero_on_flat_signal():
    sig = np.zeros(FS * 3)
    score = pacer.pacer_center_surround_score(sig, fs=FS)
    assert np.all(np.abs(score) < 1e-9)


# -- multichannel detector ----------------------------------------------------

def test_multichannel_detects_synthetic_spike_only():
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    det = pacer.detect_spikes_multichannel(
        sig, fs=FS,
        center_ms=2.0, side_ms=8.0, surround_ms=12.0, penalty=2.0,
        score_thr_mad=6.0, bipolar_thr_mad=4.0,
    )
    assert det.size >= 1
    near_spike = np.abs(det - spike_at) <= 2
    assert near_spike.any(), f"spike not near {spike_at}: got {det}"
    half_qrs = int(0.020 * FS)
    for q in qrs:
        in_qrs = (det >= q - half_qrs) & (det <= q + half_qrs)
        assert not in_qrs.any()


def test_multichannel_features_bipolar_signature():
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    feats = pacer.pacer_multichannel_features(
        sig, fs=FS, center_ms=2.0, side_ms=8.0, long_ms=12.0,
    )
    # At the spike, left × right < 0 (opposite-signed slopes).
    bp_at_spike = -feats["left"][spike_at] * feats["right"][spike_at]
    # On a QRS rising flank (a few ms before R), left × right > 0.
    rise_t = qrs[0] - int(0.008 * FS)
    bp_at_qrs = -feats["left"][rise_t] * feats["right"][rise_t]
    assert bp_at_spike > 0, f"expected bipolar > 0 at spike, got {bp_at_spike:.4f}"
    assert bp_at_qrs < 0, f"expected monotonic < 0 at QRS rise, got {bp_at_qrs:.4f}"


def test_multichannel_silent_on_pure_noise():
    rng = np.random.default_rng(seed=0)
    sig = rng.normal(0, 0.02, size=FS * 5)
    det = pacer.detect_spikes_multichannel(sig, fs=FS)
    assert det.size == 0


# -- 4-channel multi-derivative detector --------------------------------------

def test_4channel_detects_synthetic_spike_silent_on_qrs():
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    det = pacer.detect_spikes_4channel(sig, fs=FS, score_thr_mad=6.0)
    assert det.size >= 1
    assert (np.abs(det - spike_at) <= 2).any(), f"spike not detected: {det}"
    half_qrs = int(0.020 * FS)
    for q in qrs:
        in_qrs = (det >= q - half_qrs) & (det <= q + half_qrs)
        assert not in_qrs.any(), f"false positive inside QRS at {q}: {det[in_qrs]}"


def test_4channel_silent_on_pure_noise():
    rng = np.random.default_rng(seed=0)
    sig = rng.normal(0, 0.02, size=FS * 5)
    det = pacer.detect_spikes_4channel(sig, fs=FS)
    assert det.size == 0


# -- public detect_pace API -------------------------------------------------

def test_detect_pace_default_4ch_finds_synthetic_spike():
    """openecg.detect_pace wraps the 4-channel detector by default."""
    from openecg import detect_pace
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    spikes = detect_pace(sig, fs=FS)
    assert spikes.size >= 1
    assert (np.abs(spikes - spike_at) <= 2).any()


def test_detect_pace_pr_localization_filters():
    """When qrs_indices given, spikes outside [q-pre, q-gap] are dropped."""
    from openecg import detect_pace
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    spikes_all = detect_pace(sig, fs=FS)
    # The synthetic spike sits 50 ms before QRS[1] = inside [-300, -5] ms.
    spikes_loc = detect_pace(sig, fs=FS, qrs_indices=qrs)
    assert spikes_loc.size <= spikes_all.size
    assert (np.abs(spikes_loc - spike_at) <= 2).any()
    # If we shrink the PR window so the spike is excluded, no detection.
    spikes_tight = detect_pace(sig, fs=FS, qrs_indices=qrs, pre_ms=20.0, gap_ms=5.0)
    assert (np.abs(spikes_tight - spike_at) <= 2).sum() == 0


def test_detect_pace_1ch_mode_still_works():
    from openecg import detect_pace
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    spikes = detect_pace(sig, fs=FS, mode="1ch")
    assert spikes.size >= 1
    assert (np.abs(spikes - spike_at) <= 2).any()


# -- record-level is_paced_record classifier ---------------------------------

def test_is_paced_record_true_on_synthetic_paced():
    from openecg import is_paced_record
    sig, qrs, _ = _signal_with_qrs_and_spike()
    assert is_paced_record(sig, fs=FS, qrs_indices=qrs) is True


def test_is_paced_record_false_on_qrs_only_signal():
    """No spike → no high d²V/dt² peak → classifier says sinus."""
    from openecg import is_paced_record
    n = FS * 4
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 0.01, size=n)
    qrs_w = np.hanning(int(0.080 * FS))
    qrs_centers = np.array([int(0.5 * FS), int(1.5 * FS), int(2.5 * FS)],
                            dtype=np.int64)
    for c in qrs_centers:
        lo = c - len(qrs_w) // 2
        sig[lo:lo + len(qrs_w)] += qrs_w
    assert is_paced_record(sig, fs=FS, qrs_indices=qrs_centers) is False


def test_is_paced_record_threshold_tunable():
    """Lowering threshold_z fires on weaker non-linear features."""
    from openecg import is_paced_record
    sig, qrs, _ = _signal_with_qrs_and_spike(noise_mV=0.005)
    assert is_paced_record(sig, fs=FS, qrs_indices=qrs, threshold_z=10.0) is True
    # Threshold ridiculously high → no record passes.
    assert is_paced_record(sig, fs=FS, qrs_indices=qrs,
                            threshold_z=1e9) is False


def test_is_paced_record_squared_mode_still_works():
    """The older squared form (power=2) remains available."""
    from openecg import is_paced_record
    sig, qrs, _ = _signal_with_qrs_and_spike()
    assert is_paced_record(sig, fs=FS, qrs_indices=qrs,
                            power=2, center_ms=2.0, surround_ms=12.0,
                            threshold_z=1500.0) is True


# -- baseline-height (local pre-window) discriminator -------------------------

def test_baseline_height_separates_spike_from_qrs_edge():
    """Pacer spike rises from quiet PR-segment baseline → large height.
    A point on the QRS body sits where the surrounding samples are
    already off-baseline → small height."""
    sig, qrs, spike_at = _signal_with_qrs_and_spike()
    samples = np.array([spike_at, qrs[0] - int(0.005 * FS), qrs[0],
                        qrs[0] + int(0.010 * FS)], dtype=np.int64)
    h = pacer.pacer_baseline_height(sig, samples, fs=FS,
                                     baseline_window_ms=15.0, gap_ms=3.0)
    h_spike, h_qrs_pre, h_qrs_peak, h_qrs_post = h.tolist()
    assert h_spike > 5 * h_qrs_pre, (
        f"spike height {h_spike:.3f} should dominate QRS-onset {h_qrs_pre:.3f}"
    )
    assert h_spike > h_qrs_peak, (
        f"spike height {h_spike:.3f} should exceed QRS-peak {h_qrs_peak:.3f}"
    )


def test_height_filter_drops_low_amplitude_detection():
    """A high-DoB-score event with low local-baseline-height (e.g. a sharp
    QRS edge) is rejected when min_local_height_mad is set."""
    sig, qrs, spike_at = _signal_with_qrs_and_spike(noise_mV=0.005)
    # No height filter → spike is detected.
    det_open = pacer.detect_spikes_center_surround(
        sig, fs=FS, min_local_height_mad=None,
    )
    assert spike_at - 2 <= det_open[np.argmin(np.abs(det_open - spike_at))] <= spike_at + 2

    # With strict height filter (much higher than spike amplitude / σ_sig):
    # spike σ_sig is tiny (noise=0.005 mV) so spike at amplitude 4 mV is
    # ~800 σ — even a 100σ filter keeps it. Use the filter to reject a
    # synthetic "fake spike" of amplitude 0.2 mV (small).
    sig2 = sig.copy()
    sig2[spike_at] -= 4.0; sig2[spike_at + 1] -= -2.0      # remove real spike
    sig2[spike_at] += 0.05; sig2[spike_at + 1] += -0.025   # tiny replacement
    det_strict = pacer.detect_spikes_center_surround(
        sig2, fs=FS, min_local_height_mad=20.0,
    )
    near_fake = np.abs(det_strict - spike_at) <= 2 if det_strict.size else np.array([], bool)
    assert not near_fake.any(), (
        f"low-amplitude fake spike should be rejected by height filter, "
        f"got det={det_strict}"
    )
