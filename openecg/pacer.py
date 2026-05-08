# openecg/pacer.py
"""Pacemaker spike detector - highpass + adaptive threshold.

Spike characteristics: 1-3ms wide, high amplitude (>1mV), sharp slope.
ECG content (<50Hz) suppressed by 80Hz highpass; spike survives.

Spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md §8

`detect_spikes` (the original) requires absolute-mV-calibrated input and
fails on datasets whose acquisition LP filter cuts below 80 Hz (e.g.
BUT PDB has LP ~30 Hz, leaving residual spike energy at 30-50 Hz that
is invisible above 80 Hz and below the 0.1 mV absolute floor). For
those, `detect_spikes_adaptive` lowers the HP cutoff to 40 Hz and
replaces the absolute mV floor with a percentile-of-self threshold.
"""

import numpy as np
from openecg.dsp import butter, filtfilt, find_peaks


def detect_spikes(
    signal: np.ndarray,
    fs: int = 500,
    cutoff_hz: float = 80.0,
    amp_threshold_mad: float = 5.0,
    max_width_ms: float = 4.0,
    refractory_ms: float = 5.0,
    min_amplitude: float = 0.1,
) -> np.ndarray:
    """Detect pacemaker spike sample indices.

    Args:
        signal: 1D ECG samples (assumed in mV for default ``min_amplitude``).
        fs: sampling rate in Hz
        cutoff_hz: highpass cutoff (default 80Hz, suppresses ECG content)
        amp_threshold_mad: peak height threshold = N x robust sigma
            (sigma = 1.4826 * MAD).
        max_width_ms: reject peaks wider than this (likely R waves)
        refractory_ms: dedup window for bipolar artifacts
        min_amplitude: absolute floor on peak height; prevents spurious
            detections from filter ringing when MAD ~= 0.

    Returns:
        Sorted array of sample indices where spikes were detected.
    """
    nyq = fs / 2
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    hp = filtfilt(b, a, signal)
    abs_hp = np.abs(hp)

    # Robust noise estimate: 1.4826 * MAD approximates sigma of Gaussian noise.
    mad = np.median(np.abs(hp - np.median(hp)))
    sigma = 1.4826 * mad
    threshold = max(amp_threshold_mad * sigma, min_amplitude)

    refractory_samples = max(1, int(refractory_ms * fs / 1000))
    # The highpass impulse response rings for ~1/cutoff_hz seconds; use this
    # as the minimum dedup distance so ringing shoulders are suppressed.
    ringing_samples = max(1, int(fs / cutoff_hz))
    dedup_distance = max(refractory_samples, ringing_samples)
    # Filter dispersion widens narrow peaks; allow extra width for that so
    # the ``max_width_ms`` parameter refers to raw spike width, not filtered.
    filter_dispersion = max(1, int(fs / cutoff_hz / 2))
    max_width_samples = max(1, int(max_width_ms * fs / 1000)) + filter_dispersion

    peaks, _ = find_peaks(
        abs_hp,
        height=threshold,
        distance=dedup_distance,
        width=(None, max_width_samples),
    )
    return peaks.astype(np.int64)


def detect_pacer_spikes_localized(
    signal: np.ndarray,
    fs: int,
    qrs_indices: np.ndarray,
    cutoff_hz: float = 40.0,
    pre_ms: float = 80.0,
    gap_ms: float = 10.0,
    mad_scale: float = 5.0,
    bipolar_max_sep_ms: float = 2.0,
) -> np.ndarray:
    """QRS-localized bipolar pacer-spike detector.

    For each R-peak in ``qrs_indices`` (in ``fs`` sample coords), examine
    the [-pre_ms, -gap_ms] window before it. Within that window, search
    for a BIPOLAR pattern: a positive HP peak adjacent to a negative HP
    peak within ``bipolar_max_sep_ms`` ms, both > ``mad_scale`` * MAD
    sigma of the HP signal. The QRS rising edge (monopolar, slow) fails
    this test even when its narrow high-freq residuals would pass a
    simple amplitude+width detector.

    Returns the sorted sample indices of detected spike CENTERS (mid
    point between the + and - peaks). Empirically (see
    scripts/probe_pacer_bipolar.py) this distinguishes paced from
    non-paced LUDB / BUT PDB records cleanly.
    """
    sig = np.asarray(signal, dtype=np.float64)
    qrs_indices = np.asarray(qrs_indices, dtype=np.int64)
    if sig.size == 0 or qrs_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    nyq = fs / 2
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    hp = filtfilt(b, a, sig)
    sigma_robust = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
    threshold = mad_scale * sigma_robust
    if threshold <= 0:
        return np.empty(0, dtype=np.int64)

    n_pre = int(round(pre_ms * fs / 1000))
    n_gap = int(round(gap_ms * fs / 1000))
    n_sep = max(1, int(round(bipolar_max_sep_ms * fs / 1000)))
    spikes: list[int] = []
    for q in qrs_indices:
        lo = max(0, int(q) - n_pre)
        hi = max(lo, int(q) - n_gap)
        if hi - lo < 4:
            continue
        seg = hp[lo:hi]
        # Find candidate positive and negative peaks above threshold.
        pos_peaks, _ = find_peaks(seg, height=threshold)
        neg_peaks, _ = find_peaks(-seg, height=threshold)
        if pos_peaks.size == 0 or neg_peaks.size == 0:
            continue
        # Bipolar match: any (+, -) pair within n_sep samples.
        for p in pos_peaks:
            close = neg_peaks[np.abs(neg_peaks - p) <= n_sep]
            if close.size == 0:
                continue
            n = int(close[np.argmin(np.abs(close - p))])
            spikes.append(int((p + n) // 2 + lo))
            break                                           # one per QRS
    return np.asarray(sorted(set(spikes)), dtype=np.int64)


def pacer_center_surround_kernel(
    fs: int,
    center_ms: float = 3.0,
    surround_ms: float = 20.0,
    penalty: float = 3.0,
) -> np.ndarray:
    """Difference-of-boxcars kernel for the center-surround pacer score.

    Center (|i| <= Δ_c samples) gets weight +1/(2Δ_c+1).
    Surround (Δ_c < |i| <= Δ_s) gets weight -penalty/(2(Δ_s-Δ_c)).
    Convolved against (dV/dt)², it returns the average inner energy minus
    `penalty` × the average outer-ring energy.
    """
    n_c = max(1, int(round(center_ms * fs / 1000)))
    n_s = max(n_c + 1, int(round(surround_ms * fs / 1000)))
    nc_full = 2 * n_c + 1
    ns_full = 2 * (n_s - n_c)
    k = np.zeros(2 * n_s + 1, dtype=np.float64)
    inner = slice(n_s - n_c, n_s - n_c + nc_full)
    k[inner] = 1.0 / nc_full
    mask = np.ones_like(k, dtype=bool)
    mask[inner] = False
    k[mask] = -float(penalty) / ns_full
    return k


def pacer_center_surround_score(
    signal: np.ndarray,
    fs: int,
    center_ms: float = 2.0,
    surround_ms: float = 12.0,
    penalty: float = 2.0,
    diff_order: int = 1,
    power: int = 2,
) -> np.ndarray:
    """Per-sample center-vs-surround derivative score.

    Idea: a 1-2 ms pacer spike concentrates derivative energy inside a
    narrow center band. A 10-30 ms QRS rising edge spreads it across the
    center AND the surrounding band. The DoB kernel keeps the former and
    cancels the latter. With ``penalty > 1`` the QRS edge gives a
    NEGATIVE score, so a positive-side threshold rejects it cleanly.

    ``diff_order``:
      1 = first difference (DEFAULT): empirically best on LP-filtered
          clinical ECG (LUDB at 500 Hz).
      2 = second difference: zero on linear ramps, theoretically
          cleaner against QRS rising edges. In practice worse on this
          data because LP-filtered pacer spikes are smooth bumps, not
          deltas, so their d² spreads over more samples (4 vs 3).

    ``power``:
      2 = square the derivative (DEFAULT): "energy" measure; quadratically
          emphasises peaks. Best when pacer/QRS amplitude ratio is large.
      1 = absolute value: linear in the derivative magnitude. Preserves
          signal in moderate-magnitude features that ``d²`` would
          quadratically suppress; can help when the pacer/QRS amplitude
          ratio is marginal (e.g. heavily LP-filtered signals).

    Output has the same length and sample rate as the input.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.size < 3:
        return np.zeros_like(sig)
    fs_ms = fs / 1000.0                                       # samples per ms
    # Differentiate in PHYSICAL /ms units so score is fs-invariant: a fixed
    # absolute threshold (or absolute σ floor) means the same thing at fs=360,
    # 500, 1000, ... — and the σ-based threshold is unchanged either way
    # because it scales proportionally with d.
    if diff_order == 1:
        d = np.diff(sig, prepend=sig[0]) * fs_ms              # mV/ms
    elif diff_order == 2:
        d = np.diff(sig, n=2, prepend=sig[0], append=sig[-1]) * fs_ms ** 2  # mV/ms²
    else:
        raise ValueError(f"diff_order must be 1 or 2, got {diff_order}")
    if power == 2:
        e = d * d                                             # (mV/ms^k)²
    elif power == 1:
        e = np.abs(d)                                         # |mV/ms^k|
    else:
        raise ValueError(f"power must be 1 or 2, got {power}")
    k = pacer_center_surround_kernel(fs, center_ms, surround_ms, penalty)
    return np.convolve(e, k, mode="same")


def pacer_baseline_height(
    signal: np.ndarray,
    samples: np.ndarray,
    fs: int,
    baseline_window_ms: float = 15.0,
    gap_ms: float = 3.0,
) -> np.ndarray:
    """For each index in ``samples``, return |sig(t) − median(sig in pre-window)|.

    The pre-window is [t − gap_ms − baseline_window_ms, t − gap_ms]. ``gap_ms``
    excludes a small region adjacent to t so the spike itself doesn't pull the
    "baseline" toward the spike value.

    Pacer spikes occur on top of the quiet PR segment, so the spike sample
    rises far above the immediately-preceding baseline. Q/S inflections sit
    INSIDE the QRS, where the surrounding samples are already off-baseline,
    so |sig(t) − local_baseline| stays small. This is the discriminator
    suggested by user observation:
        "pacing 은 baseline 에서 발생하고 q,s는 아닐듯".
    """
    sig = np.asarray(signal, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.int64)
    if samples.size == 0:
        return np.empty(0, dtype=np.float64)
    n_g = max(1, int(round(gap_ms * fs / 1000)))
    n_w = max(2, int(round(baseline_window_ms * fs / 1000)))
    out = np.empty(samples.size, dtype=np.float64)
    for i, t in enumerate(samples):
        lo = max(0, int(t) - n_g - n_w)
        hi = max(lo + 1, int(t) - n_g)
        out[i] = abs(float(sig[t]) - float(np.median(sig[lo:hi])))
    return out


def detect_spikes_center_surround(
    signal: np.ndarray,
    fs: int,
    center_ms: float = 1.5,
    surround_ms: float = 20.0,
    penalty: float = 3.0,
    diff_order: int = 1,
    power: int = 2,
    p_high: float = 99.5,
    p_high_scale: float = 0.4,
    min_score_mad: float = 200.0,
    refractory_ms: float = 20.0,
    min_baseline_mad: float | None = None,
    min_local_height_mad: float | None = None,
    baseline_window_ms: float = 15.0,
    baseline_gap_ms: float = 3.0,
) -> np.ndarray:
    """Pacer-spike detector built on ``pacer_center_surround_score``.

    Threshold = max(``p_high_scale`` × percentile_p_high(score⁺),
                    ``min_score_mad`` × 1.4826·MAD(score)). The double
    threshold prevents firing on records where the score's noise floor
    is unusually flat (low percentile, MAD term wins) or unusually busy
    (high percentile, percentile term wins).

    Default ``min_score_mad=200`` is calibrated against LUDB sinus ctrls:
    on a paced rid8 the lowest detected event scores ~450 σ_score; on
    sinus rid33 (worst-FP ctrl) the highest spurious peak scores ~118
    σ_score. A 200-σ threshold therefore sits cleanly between the two
    populations on the LUDB cohort.

    Two optional post-filters reject candidates that pass the score
    threshold but aren't true pacer spikes:

    ``min_baseline_mad`` (global fallback): require |sig(t) − global_median|
        to exceed N × global σ_sig. Suppresses tiny noise-shaped events.

    ``min_local_height_mad`` (LOCAL baseline check, default None): require
        |sig(t) − median(sig in [t − gap − window, t − gap])| to exceed
        N × global σ_sig. Pacer spikes rise from the quiet PR-segment
        baseline; Q/S inflections sit on the QRS body where surroundings
        are already off-baseline, so their local-height stays small.
        With score-σ thresholding at 200 the local-height check is
        redundant on the LUDB cohort and disabled by default; enable it
        (e.g. 4.0) on noisier signals where the score-σ gap to ctrls is
        smaller.

    Note on ``center_ms``: at fs=500 (LUDB), the kernel's center half-
    width clamps to 1 sample (3-sample inner band, ≈6 ms), so
    ``center_ms`` values from 1.0 to 3.0 all produce the same kernel.
    Going narrower than ~3 samples is counterproductive because
    ``np.gradient`` already spreads a 1-sample bipolar pacer's d² over
    4 samples; a 1-sample center would push 75% of the spike's energy
    into the surround.
    """
    sig_arr = np.asarray(signal, dtype=np.float64)
    score = pacer_center_surround_score(
        sig_arr, fs, center_ms, surround_ms, penalty,
        diff_order=diff_order, power=power,
    )
    pos = np.maximum(score, 0.0)
    if pos.max() <= 0:
        return np.empty(0, dtype=np.int64)
    score_mad = float(np.median(np.abs(score - np.median(score))))
    score_sigma = 1.4826 * score_mad
    p_high_term = p_high_scale * float(np.percentile(pos, p_high))
    threshold = max(p_high_term, min_score_mad * score_sigma)
    if threshold <= 0:
        return np.empty(0, dtype=np.int64)
    refractory = max(1, int(round(refractory_ms * fs / 1000)))
    peaks, _ = find_peaks(pos, height=threshold, distance=refractory)
    if peaks.size:
        # ``np.diff`` shifts the d² peak by ½ sample relative to the spike's
        # signal-domain amplitude peak. Snap each detection to the nearby
        # |sig − median| maximum so the baseline-height check measures the
        # actual spike amplitude, not the post-shift sample.
        med_global = float(np.median(sig_arr))
        snap_half = max(1, int(round(2.0 * fs / 1000)))           # ±2 ms
        snapped = np.empty_like(peaks)
        for i, p in enumerate(peaks):
            lo = max(0, int(p) - snap_half)
            hi = min(sig_arr.size, int(p) + snap_half + 1)
            snapped[i] = lo + int(np.argmax(np.abs(sig_arr[lo:hi] - med_global)))
        peaks = snapped
    if peaks.size and (min_baseline_mad is not None
                        or min_local_height_mad is not None):
        med = float(np.median(sig_arr))
        sigma_sig = 1.4826 * float(np.median(np.abs(sig_arr - med)))
        if sigma_sig > 0:
            if min_baseline_mad is not None:
                keep = np.abs(sig_arr[peaks] - med) >= min_baseline_mad * sigma_sig
                peaks = peaks[keep]
            if min_local_height_mad is not None and peaks.size:
                heights = pacer_baseline_height(
                    sig_arr, peaks, fs,
                    baseline_window_ms=baseline_window_ms,
                    gap_ms=baseline_gap_ms,
                )
                peaks = peaks[heights >= min_local_height_mad * sigma_sig]
    return peaks.astype(np.int64)


def _shifted_pad(arr: np.ndarray, k: int) -> np.ndarray:
    """``arr`` shifted by ``k`` samples (positive = right) with edge padding."""
    out = np.empty_like(arr)
    if k > 0:
        out[:k] = arr[0]
        out[k:] = arr[:-k]
    elif k < 0:
        out[k:] = arr[-1]
        out[:k] = arr[-k:]
    else:
        out[:] = arr
    return out


def pacer_multichannel_features(
    signal: np.ndarray,
    fs: int,
    center_ms: float = 2.0,
    side_ms: float = 8.0,
    long_ms: float = 30.0,
) -> dict[str, np.ndarray]:
    """Four slope channels at the input sample rate.

    Each channel is a single 1D shift-and-difference (i.e. a 1D conv with
    a 2-tap kernel) on ``signal``:

        center  : sig(t+Δc) − sig(t−Δc), scaled to mV/ms.    Magnitude.
        left    : sig(t)    − sig(t−Δs), scaled to mV/ms.    Signed.
        right   : sig(t+Δs) − sig(t),    scaled to mV/ms.    Signed.
        long    : sig(t+Δl) − sig(t−Δl), scaled to mV/ms.    Magnitude.

    Pacer-spike signature (at the spike center sample):
        center  : LARGE (sharp local change)
        long    : small  (signal returns near baseline within ±long_ms/2)
        left × right : NEGATIVE (bipolar — left rises into spike, right falls out)
    QRS rising-edge signature (anywhere on a sustained ramp):
        center  : LARGE (slope still nonzero at any single sample)
        long    : LARGE (slope is sustained over long_ms)
        left × right : POSITIVE (monotonic — both sides have the same sign)

    Sample-rate-aware: half-widths in samples are floored to >=1, so on
    low-fs records (e.g. BUT PDB at 360 Hz) the effective center_ms is
    2 samples ≈ 5.5 ms, not 2 ms. For an exact 2-ms center on 360-Hz
    data, resample to 500 Hz before calling.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n_c = max(1, int(round(center_ms * fs / 2000)))      # half-width
    n_s = max(1, int(round(side_ms * fs / 1000)))         # full-width
    n_l = max(1, int(round(long_ms * fs / 2000)))         # half-width
    fs_ms = fs / 1000.0
    center = (_shifted_pad(sig, -n_c) - _shifted_pad(sig, n_c)) * fs_ms / (2 * n_c)
    left   = (sig - _shifted_pad(sig, n_s))                * fs_ms / n_s
    right  = (_shifted_pad(sig, -n_s) - sig)               * fs_ms / n_s
    longw  = (_shifted_pad(sig, -n_l) - _shifted_pad(sig, n_l)) * fs_ms / (2 * n_l)
    return {
        "center": np.abs(center),
        "left":   left,
        "right":  right,
        "long":   np.abs(longw),
    }


def detect_spikes_multichannel(
    signal: np.ndarray,
    fs: int,
    center_ms: float = 2.0,
    side_ms: float = 8.0,
    surround_ms: float = 12.0,
    penalty: float = 2.0,
    score_thr_mad: float = 6.0,
    bipolar_thr_mad: float = 4.0,
    refractory_ms: float = 30.0,
    min_local_height_mad: float | None = 8.0,
    baseline_window_ms: float = 15.0,
    baseline_gap_ms: float = 3.0,
) -> np.ndarray:
    """Joint multi-channel pacer-spike detector.

    Combines the center-vs-surround DoB-on-d² score with a bipolar
    left-vs-right slope-product gate. Both must clear their MAD-based
    thresholds in the SAME sample for a detection.

    Per-sample decision (both conditions hold):
        score    > score_thr_mad   × σ_score          (DoB: narrow energy
                                                       concentration — center
                                                       d² ≫ surround d²)
        −left·r  > bipolar_thr_mad² × σ_left·σ_right   (signed L and R slopes
                                                       have OPPOSITE signs —
                                                       bipolar transient, not
                                                       a monotonic ramp)

    The DoB rejects sustained QRS edges (center ≈ surround). The bipolar
    test rejects narrow MONOTONIC transients (e.g., a sudden DC step) that
    the DoB alone might pass. Used together, they isolate the bipolar
    1-2 ms pacer spike from both wide ramps and narrow non-bipolar artifacts.

    Returns sorted sample indices at the input sample rate.
    """
    sig = np.asarray(signal, dtype=np.float64)
    score = pacer_center_surround_score(
        sig, fs, center_ms=center_ms, surround_ms=surround_ms, penalty=penalty,
    )
    feats = pacer_multichannel_features(
        sig, fs, center_ms=center_ms, side_ms=side_ms, long_ms=surround_ms,
    )
    bipolar = -feats["left"] * feats["right"]

    def _robust_sigma(x: np.ndarray) -> float:
        med = float(np.median(x))
        return 1.4826 * float(np.median(np.abs(x - med))) + 1e-12

    sigma_score = _robust_sigma(score)
    sigma_l = _robust_sigma(feats["left"])
    sigma_r = _robust_sigma(feats["right"])

    cond = (
        (score > score_thr_mad * sigma_score)
        & (bipolar > (bipolar_thr_mad ** 2) * sigma_l * sigma_r)
    )
    if not np.any(cond):
        return np.empty(0, dtype=np.int64)

    score_pos = np.maximum(score, 0.0)
    bp_pos = np.maximum(bipolar, 0.0)
    masked = np.where(cond, score_pos * bp_pos, 0.0)
    refractory = max(1, int(round(refractory_ms * fs / 1000)))
    peaks, _ = find_peaks(masked, distance=refractory)
    if peaks.size and min_local_height_mad is not None:
        med = float(np.median(sig))
        sigma_sig = 1.4826 * float(np.median(np.abs(sig - med)))
        if sigma_sig > 0:
            heights = pacer_baseline_height(
                sig, peaks, fs,
                baseline_window_ms=baseline_window_ms,
                gap_ms=baseline_gap_ms,
            )
            peaks = peaks[heights >= min_local_height_mad * sigma_sig]
    return peaks.astype(np.int64)


def is_paced_record(
    signal: np.ndarray,
    fs: int,
    qrs_indices: np.ndarray | None = None,
    threshold_z: float = 32.0,
    pre_ms: float = 300.0,
    gap_ms: float = 5.0,
    center_ms: float = 5.0,
    surround_ms: float = 30.0,
    penalty: float = 2.0,
    diff_order: int = 2,
    power: int = 1,
) -> bool:
    """Record-level paced-vs-sinus classifier.

    Returns ``True`` if the per-sample center-surround score reaches
    ``threshold_z × σ_score`` anywhere inside the PR-segment window of
    any QRS.

    The default form is **linear absolute 2nd-derivative** (``|d²V/dt²|``,
    diff_order=2, power=1) on a **wide kernel** (center 5 ms, surround
    30 ms). On a combined PTB-XL V1 (50+50) + LUDB lead II (12+10)
    holdout this catches **48/52 paced (92%) at 100% specificity**. By
    contrast the squared form on a narrow kernel (center 2 / surround 12,
    power 2) catches only 45/52 (87%) at the same spec.

    Why this combination wins:
      * Wide kernel matches LP-filtered pacer spikes, which spread to
        5-10 ms after acquisition LP rather than staying as 1-2 ms
        deltas. A narrow center captures only the spike's centre and
        loses energy to the surround.
      * Linear absolute (|·|) preserves moderate-amplitude spikes that
        the squared form (·²) over-suppresses.

    Sweep at this configuration (``threshold_z`` chosen so no sinus
    record exceeds it):

        threshold_z =  32: 92% sens / 100% spec / 100% PPV
        threshold_z =  20: 92% sens /  98% spec
        threshold_z =  50: 88% sens / 100% spec
        threshold_z = 100: 75% sens / 100% spec

    The 4 still-missed records (PTBXL-5787, 13177, 20181, LUDB-45) have
    no detectable pacer response at the lead/derivative scale we examine
    — typically atrial-only pacing, capture failures, or signals heavily
    attenuated by the acquisition LP filter.

    Args:
        signal: 1-D ECG samples.
        fs: sampling rate (Hz).
        qrs_indices: optional R-peak indices. When None, the entire
            record is searched (no PR localisation).
        threshold_z: σ-multiplier; default 32 = max sinus + small margin
            on the validation cohort.
        pre_ms, gap_ms: PR-segment window bounds when qrs_indices given.
        center_ms, surround_ms, penalty: forwarded to
            ``pacer_center_surround_score``.
        diff_order, power: derivative-form selectors. Default
            ``(2, 1)`` = ``|d²V/dt²|``. Use ``(2, 2)`` for the older
            squared form.
    """
    sig = np.asarray(signal, dtype=np.float64)
    score = pacer_center_surround_score(
        sig, fs,
        center_ms=center_ms, surround_ms=surround_ms, penalty=penalty,
        diff_order=diff_order, power=power,
    )
    sigma = 1.4826 * float(np.median(np.abs(score - np.median(score)))) + 1e-12
    if qrs_indices is None or len(np.asarray(qrs_indices)) == 0:
        return bool(score.max() / sigma >= threshold_z)
    qrs = np.asarray(qrs_indices, dtype=np.int64)
    n_pre = int(round(pre_ms * fs / 1000))
    n_gap = int(round(gap_ms * fs / 1000))
    best = 0.0
    for q in qrs:
        lo = max(0, int(q) - n_pre)
        hi = max(lo, int(q) - n_gap)
        if hi > lo:
            best = max(best, float(score[lo:hi].max()))
    return bool(best / sigma >= threshold_z)


def detect_spikes_4channel(
    signal: np.ndarray,
    fs: int,
    center_ms: float = 1.5,
    surround_ms: float = 20.0,
    penalty: float = 3.0,
    score_thr_mad: float | tuple[float, float, float, float] = 8.0,
    refractory_ms: float = 20.0,
    min_local_height_mad: float | None = None,
    baseline_window_ms: float = 15.0,
    baseline_gap_ms: float = 3.0,
) -> np.ndarray:
    """4-channel single-lead detector AND-gating across derivative forms.

    Channels (per sample, all DoB-convolved against their respective
    derivative-magnitude operand):

        ch0 = DoB * |d|         — first-difference absolute,  linear
        ch1 = DoB * d²          — first-difference squared,   quadratic
        ch2 = DoB * |d²V/dt²|   — second-difference absolute, linear
        ch3 = DoB * (d²V/dt²)²  — second-difference squared,  quadratic

    Decision: a sample is a candidate iff EVERY channel ``i`` exceeds its
    threshold ``τᵢ × σᵢ`` simultaneously. ``score_thr_mad`` accepts:
        - a scalar: same τ for all four channels (simpler)
        - a 4-tuple: per-channel thresholds (τ₀, τ₁, τ₂, τ₃). Useful for
          loosening just the weakest-link channel; ch0 (|d|) tends to be
          smaller in σ-units than ch1/ch3 because squaring concentrates
          signal energy at peaks while the linear form is more uniform.

    Returns sorted spike sample indices.
    """
    sig = np.asarray(signal, dtype=np.float64)
    pieces = []
    for diff_order in (1, 2):
        for power in (1, 2):
            pieces.append(
                pacer_center_surround_score(
                    sig, fs,
                    center_ms=center_ms, surround_ms=surround_ms,
                    penalty=penalty, diff_order=diff_order, power=power,
                )
            )
    channels = np.stack(pieces, axis=0)                    # (4, N)

    def _sigma(x):
        m = float(np.median(x))
        return 1.4826 * float(np.median(np.abs(x - m))) + 1e-12

    sigmas = np.array([_sigma(c) for c in channels])       # (4,)
    if np.isscalar(score_thr_mad):
        thr_per_ch = np.full(4, float(score_thr_mad))
    else:
        thr_per_ch = np.asarray(score_thr_mad, dtype=np.float64)
        if thr_per_ch.shape != (4,):
            raise ValueError(f"score_thr_mad tuple must have 4 elements, got {thr_per_ch.shape}")
    cond = np.all(channels > (thr_per_ch * sigmas)[:, None], axis=0)
    if not np.any(cond):
        return np.empty(0, dtype=np.int64)
    pos = np.maximum(channels, 0.0)
    geo = np.prod(pos, axis=0) ** 0.25
    masked = np.where(cond, geo, 0.0)
    refractory = max(1, int(round(refractory_ms * fs / 1000)))
    peaks, _ = find_peaks(masked, distance=refractory)
    if peaks.size and min_local_height_mad is not None:
        med = float(np.median(sig))
        sigma_sig = 1.4826 * float(np.median(np.abs(sig - med)))
        if sigma_sig > 0:
            heights = pacer_baseline_height(
                sig, peaks, fs,
                baseline_window_ms=baseline_window_ms,
                gap_ms=baseline_gap_ms,
            )
            peaks = peaks[heights >= min_local_height_mad * sigma_sig]
    return peaks.astype(np.int64)


def detect_pace(
    signal: np.ndarray,
    fs: int,
    qrs_indices: np.ndarray | None = None,
    pre_ms: float = 300.0,
    gap_ms: float = 5.0,
    mode: str = "4ch",
    score_thr_mad: float = 6.0,
    **kwargs,
) -> np.ndarray:
    """Detect pacemaker-spike sample indices in a 1-D ECG.

    Public entry point for the center-vs-surround pacer detector. By
    default uses the 4-channel multi-derivative AND-gated detector
    (``mode='4ch'``), which AND-gates four DoB-on-derivative scores —
    |d|, d², |d²V/dt²|, (d²V/dt²)² — at the same sample. This is more
    robust than any single-channel form on the validation cohorts:

      LUDB lead II:  rid8 0.83 / 11 sinus ctrls 0% FP / BUT PDB rid22 0% FP
      PTB-XL V1:     50-record holdout, 84% sens / 100% spec / 100% PPV
                     (≥1 spike per record, gap_ms=5)

    All thresholds are expressed in physical /ms units inside the score
    so the same parameter values apply across fs=360 (BUT PDB), 500
    (LUDB / PTB-XL), 1000 (raw monitor) without rescaling.

    Pipeline:
      1. 4 DoB-on-derivative scores at every sample (each a single
         1-D convolution) — see ``detect_spikes_4channel``.
      2. Sample is a candidate iff ALL 4 channels exceed
         ``score_thr_mad × σ_channel``.
      3. (optional) PR-segment localization when ``qrs_indices`` is
         given: drop spikes outside ``[q − pre_ms, q − gap_ms]``.
         Default ``pre_ms=300, gap_ms=5`` covers atrial-only pacing
         (which sits 100-300 ms before R) and ventricular pacing
         (5-50 ms before R).

    Args:
        signal: 1-D ECG samples.
        fs: sampling rate in Hz.
        qrs_indices: optional R-peak sample indices. When provided, only
            spikes within ``[q − pre_ms, q − gap_ms]`` of any QRS are
            returned. Atrial pacing → keep ``pre_ms`` ≥ 200; ventricular
            pacing close to QRS → keep ``gap_ms`` small (≤ 10).
        pre_ms: PR-window upper bound, default 300 (covers atrial pacing).
        gap_ms: PR-window lower bound, default 5 (admits ventricular spikes
            that sit right at the QRS onset).
        mode: '4ch' (default, AND across 4 derivative forms) or '1ch'
            (single-channel d² center-surround, the older form).
        score_thr_mad: per-channel σ-threshold; only used when mode='4ch'.
            6.0 gives 100% specificity on both LUDB and PTB-XL holdouts.
        **kwargs: forwarded to the underlying detector for advanced overrides.

    Returns:
        Sorted np.int64 array of pacer-spike sample indices.
    """
    if mode == "4ch":
        spikes = detect_spikes_4channel(
            signal, fs, score_thr_mad=score_thr_mad, **kwargs,
        )
    elif mode == "1ch":
        spikes = detect_spikes_center_surround(signal, fs, **kwargs)
    else:
        raise ValueError(f"mode must be '4ch' or '1ch', got {mode!r}")
    if qrs_indices is None or spikes.size == 0:
        return spikes
    qrs = np.asarray(qrs_indices, dtype=np.int64)
    if qrs.size == 0:
        return np.empty(0, dtype=np.int64)
    qrs_sorted = np.sort(qrs)
    n_pre = int(round(pre_ms * fs / 1000))
    n_gap = int(round(gap_ms * fs / 1000))
    keep = []
    for s in spikes:
        idx = int(np.searchsorted(qrs_sorted, s))
        if idx < qrs_sorted.size:
            q = int(qrs_sorted[idx])
            if q - n_pre <= int(s) <= q - n_gap:
                keep.append(int(s))
    return np.asarray(keep, dtype=np.int64)


def detect_spikes_adaptive(
    signal: np.ndarray,
    fs: int,
    cutoff_hz: float = 40.0,
    amp_threshold_mad: float = 5.0,
    p_high: float = 99.9,
    p_high_scale: float = 0.3,
    max_width_ms: float = 4.0,
    refractory_ms: float = 5.0,
) -> np.ndarray:
    """Adaptive-threshold pacer-spike detector.

    Differences vs ``detect_spikes``:
      * ``cutoff_hz=40`` (was 80) so spike residuals on tightly-band-limited
        recordings (e.g. BUT PDB, LP ~30 Hz) survive in the 30-50 Hz band.
      * No absolute-mV floor; the threshold is purely a function of the
        signal's own HP statistics. Threshold =
        max(amp_threshold_mad * 1.4826*MAD,  p_high_scale * percentile_p_high).
        On a "no spike" record the high percentile collapses near MAD scale,
        so the second term does not over-fire.
      * Returns sample indices in the same convention as ``detect_spikes``.

    Empirically tuned on BUT PDB rid=3 / rid=22, LUDB rid=8 / rid=45 (see
    scripts/probe_pacer_numeric.py): adaptive flags spikes at >40 % of
    pre-QRS windows in paced records and ~0 % in BI control rid=22.
    """
    nyq = fs / 2
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    hp = filtfilt(b, a, signal)
    abs_hp = np.abs(hp)

    mad = float(np.median(np.abs(hp - np.median(hp))))
    sigma_robust = 1.4826 * mad
    p_top = float(np.percentile(abs_hp, p_high))
    threshold = max(amp_threshold_mad * sigma_robust, p_high_scale * p_top)
    if threshold <= 0:
        return np.empty(0, dtype=np.int64)

    refractory_samples = max(1, int(refractory_ms * fs / 1000))
    ringing_samples = max(1, int(fs / cutoff_hz))
    dedup_distance = max(refractory_samples, ringing_samples)
    filter_dispersion = max(1, int(fs / cutoff_hz / 2))
    max_width_samples = max(1, int(max_width_ms * fs / 1000)) + filter_dispersion

    peaks, _ = find_peaks(
        abs_hp,
        height=threshold,
        distance=dedup_distance,
        width=(None, max_width_samples),
    )
    return peaks.astype(np.int64)
