"""QRS R-peak detector — pure-numpy/scipy, no external deps.

Algorithm: Makowski's gradient-thresholded QRS detector, vendored from
NeuroKit2's ``_ecg_findpeaks_neurokit`` (Makowski 2020). Validated on
MIT-BIH Arrhythmia DB at micro-F1 = 0.994 (48 records, lead 0, 100 ms
tolerance per AAMI EC57); see ``scripts/validate_qrs_mitdb.py``.

Pipeline (all stages stride-1, single pass):

  1. (optional) 0.5 Hz Butterworth high-pass to remove baseline wander —
     the algorithm assumes an already-HP-filtered input, so we apply
     this by default for raw clinical signals.
  2. |∇x| — absolute first difference (numpy.gradient).
  3. Boxcar smooth with 100 ms window → ``smoothgrad``.
  4. Boxcar smooth ``smoothgrad`` with 750 ms window → ``avggrad``.
  5. Region mask: ``smoothgrad > 1.5 · avggrad``.
  6. Drop regions shorter than ``0.4 × mean(QRS_len)``.
  7. Within each surviving region, take the most-prominent local
     maximum of ``signal`` AND of ``-signal`` (so inverted leads work),
     and keep whichever has the larger prominence.
  8. Enforce 300 ms minimum delay between accepted R-peaks.

Public surface: ``detect_qrs(signal, fs)`` → int64 sample indices.

License notice (NeuroKit2 algorithm, MIT-licensed):

    Copyright (c) 2020 Dominique Makowski. Original implementation
    in ``neurokit2.ecg.ecg_findpeaks._ecg_findpeaks_neurokit``.
    Re-implemented here for openecg under Apache-2.0; the NeuroKit2
    MIT license terms apply to the algorithm-derived code (see
    https://github.com/neuropsychology/NeuroKit/blob/master/LICENSE).

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction.
"""

from __future__ import annotations

import numpy as np
from openecg.dsp import butter, filtfilt, find_peaks


def _highpass_05(signal: np.ndarray, fs: int) -> np.ndarray:
    """Zero-phase 0.5 Hz Butterworth high-pass to suppress baseline wander."""
    nyq = 0.5 * fs
    cutoff = 0.5 / nyq
    if cutoff >= 1.0:
        # Pathologically low fs — skip filtering.
        return np.asarray(signal, dtype=np.float64)
    b, a = butter(2, cutoff, btype="high")
    padlen = min(len(signal) - 1, 3 * (max(len(a), len(b)) + 1))
    if padlen <= 0:
        return filtfilt(b, a, signal)
    return filtfilt(b, a, signal, padlen=padlen)


def _boxcar_smooth(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    """Centered boxcar (uniform-kernel) smoothing — matches NeuroKit2's
    ``signal_smooth(kernel='boxcar', size=N)``."""
    if kernel_size <= 1:
        return np.asarray(signal, dtype=np.float64)
    k = np.ones(int(kernel_size), dtype=np.float64) / float(kernel_size)
    return np.convolve(signal, k, mode="same")


def measure_qrs_widths(
    signal,
    fs: int,
    peaks,
    *,
    search_window_ms: float = 100.0,
    isoelectric_window_ms: float = 250.0,
    baseline_multiplier: float = 3.0,
    smooth_ms: float = 8.0,
    physiological_min_ms: float = 40.0,
    default_width_ms: float = 80.0,
) -> np.ndarray:
    """Estimate per-beat QRS duration (ms) from |∇x| around each R peak.

    For each R peak we compute the contiguous interval around the peak
    where |∇x| (mildly smoothed, no 100 ms boxcar floor) exceeds a
    *baseline-relative* threshold derived from the local PR/TP segments.

    Algorithm
    ---------
    1. Take |∇x| of the (interp-NaN-filled) signal, then smooth with a
       short ``smooth_ms`` boxcar (default 8 ms — removes sample noise
       without blurring the QRS slope).
    2. For each peak p:
       a. ``baseline`` = 25th-percentile of |∇x| within ±isoelectric_window_ms
          of p. The 25th percentile is dominated by PR/TP segments, so it
          tracks the patient's true isoelectric gradient.
       b. ``thr = baseline × baseline_multiplier`` (default 3×).
       c. Walk outward from the peak while |∇x| > thr. The run length is
          the QRS duration.

    On clean sinus ECG this gives QRS widths ≈ 70-100 ms;
    PVCs widen to 140-180 ms. The 100 ms smoothing-kernel "floor"
    present in :func:`detect_qrs`'s internal region width is avoided
    because the smoothing here is only 8 ms.

    Args:
        signal: 1-D ECG samples (same array passed to :func:`detect_qrs`).
        fs: sampling rate in Hz.
        peaks: R-peak sample indices (as returned by ``detect_qrs``).
        search_window_ms: half-width of the per-peak |∇x| > thr walk
            (default 100). QRS won't be wider than this.
        isoelectric_window_ms: half-width of the baseline window used
            to estimate the patient's PR/TP-segment |∇x| floor
            (default 250).
        baseline_multiplier: width ends where |∇x| drops to
            baseline × baseline_multiplier (default 3.0).
        smooth_ms: short boxcar smoothing on |∇x| before threshold
            walk (default 8). 0 disables smoothing.
        physiological_min_ms: any measured width below this is treated
            as a measurement failure (default 40 — narrower than any
            clinically plausible QRS). Failed measurements are imputed.
        default_width_ms: imputation value when *all* peaks fail
            measurement (default 80 — a typical clinical narrow QRS).
            When at least one peak measures above the physiological
            floor, the failures are imputed with the in-window median
            of successful measurements instead.

    Returns:
        ``np.float64`` array of widths in ms, parallel to ``peaks``.
        Every value is ≥ physiological_min_ms; failed peaks are imputed
        with the in-window median of successful measurements (or
        ``default_width_ms`` when no peak measured successfully).
    """
    sig = np.asarray(signal, dtype=np.float64)
    peaks_arr = np.asarray(peaks, dtype=np.int64)
    if sig.size == 0 or peaks_arr.size == 0:
        return np.zeros(peaks_arr.size, dtype=np.float64)
    nan_mask = np.isnan(sig)
    if nan_mask.any():
        idx = np.arange(sig.size)
        sig = np.interp(idx, idx[~nan_mask], sig[~nan_mask])
    grad = np.abs(np.gradient(sig))
    smooth_n = max(1, int(round(smooth_ms * fs / 1000.0)))
    if smooth_n > 1:
        grad = _boxcar_smooth(grad, smooth_n)
    win_s = int(round(search_window_ms * fs / 1000.0))
    iso_s = int(round(isoelectric_window_ms * fs / 1000.0))

    # Threshold relaxation ladder: try strict first, fall back to permissive
    # multipliers if the resulting width is below the physiological floor.
    # This keeps narrow-but-real QRS at ~70-90 ms while giving low-SNR
    # / edge beats a chance to be measured rather than reported as 0.
    multiplier_ladder = (baseline_multiplier, 2.0, 1.5, 1.2, 1.0)
    min_samples = max(1, int(round(physiological_min_ms * fs / 1000.0)))
    default_samples = max(min_samples, int(round(default_width_ms * fs / 1000.0)))

    out = np.zeros(peaks_arr.size, dtype=np.float64)
    for i, p in enumerate(peaks_arr):
        lo_iso = max(0, int(p) - iso_s)
        hi_iso = min(len(grad), int(p) + iso_s)
        if hi_iso - lo_iso < 8:
            out[i] = default_width_ms
            continue
        baseline = float(np.percentile(grad[lo_iso:hi_iso], 25))
        if baseline <= 0:
            out[i] = default_width_ms
            continue
        lo = max(0, int(p) - win_s)
        hi = min(len(grad), int(p) + win_s)
        local = grad[lo:hi]
        peak_rel = int(p) - lo
        if peak_rel < 0 or peak_rel >= len(local):
            out[i] = default_width_ms
            continue
        # Walk with progressively-relaxed threshold until the measured
        # width meets the physiological minimum. Polarity-agnostic via
        # |∇x| — inverted QRS (peak below baseline) has the same gradient
        # signature so the walk works regardless of sign of the peak.
        measured = 0
        for mult in multiplier_ladder:
            thr = baseline * mult
            above = local > thr
            if not above[peak_rel]:
                continue
            l = peak_rel
            while l > 0 and above[l - 1]:
                l -= 1
            r = peak_rel
            while r < len(above) - 1 and above[r + 1]:
                r += 1
            run = r - l
            if run >= min_samples:
                measured = run
                break
            if run > measured:
                measured = run
        if measured >= min_samples:
            out[i] = measured * 1000.0 / fs
        else:
            # Even with the most permissive threshold the run is sub-floor.
            # Default to a typical clinical width rather than 0.
            out[i] = default_width_ms
    return out


def detect_qrs(
    signal,
    fs: int,
    *,
    smoothwindow_ms: float = 100.0,
    avgwindow_ms: float = 750.0,
    gradthreshweight: float = 1.5,
    minlenweight: float = 0.4,
    mindelay_ms: float = 300.0,
    highpass: bool = True,
    return_widths: bool = False,
):
    """Detect QRS R-peak sample indices in a 1-D ECG.

    Args:
        signal: 1-D ECG samples (any units).
        fs: sampling rate in Hz.
        smoothwindow_ms: short boxcar smoothing kernel for |∇x| (default
            100 — Makowski's value, robust on 250-1000 Hz).
        avgwindow_ms: long boxcar window producing the rolling baseline
            ``avggrad`` (default 750).
        gradthreshweight: threshold = ``gradthreshweight × avggrad``
            (default 1.5).
        minlenweight: drop QRS regions shorter than ``minlenweight × mean
            region length`` (default 0.4).
        mindelay_ms: minimum spacing between accepted peaks
            (default 300 — corresponds to 200 bpm).
        highpass: apply a 0.5 Hz Butterworth high-pass before detection
            (default True). The original NeuroKit algorithm assumes an
            already-HP-filtered signal; the prefilter is convenient for
            raw clinical inputs.
        return_widths: if True, also return per-beat QRS widths in ms
            (measured by :func:`measure_qrs_widths` with default
            parameters). Default False keeps the legacy signature.

    Returns:
        Sorted ``np.int64`` array of R-peak sample indices. When
        ``return_widths=True`` a ``(peaks, widths_ms)`` tuple is returned
        instead, with parallel arrays.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.size == 0:
        return np.empty(0, dtype=np.int64)

    # NaN-safe: linear interp over any NaNs.
    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return np.empty(0, dtype=np.int64)
    if nan_mask.any():
        idx = np.arange(sig.size)
        sig = np.interp(idx, idx[~nan_mask], sig[~nan_mask])

    if sig.size < max(8, int(0.05 * fs)):
        return np.empty(0, dtype=np.int64)

    proc = _highpass_05(sig, fs) if highpass else sig

    # Gradient → smoothed → long-average baseline → threshold mask.
    grad = np.gradient(proc)
    absgrad = np.abs(grad)
    smooth_n = max(1, int(round(smoothwindow_ms * fs / 1000.0)))
    avg_n = max(1, int(round(avgwindow_ms * fs / 1000.0)))
    smoothgrad = _boxcar_smooth(absgrad, smooth_n)
    avggrad = _boxcar_smooth(smoothgrad, avg_n)
    threshold = gradthreshweight * avggrad
    mindelay = max(1, int(round(mindelay_ms * fs / 1000.0)))

    qrs_mask = smoothgrad > threshold
    # Region transitions: rising = beg, falling = end.
    diff_mask = np.diff(qrs_mask.astype(np.int8))
    beg_qrs = np.where(diff_mask == 1)[0]
    end_qrs = np.where(diff_mask == -1)[0]

    if beg_qrs.size == 0 or end_qrs.size == 0:
        return np.empty(0, dtype=np.int64)

    # Discard QRS-ends that precede the first QRS-start, and clip ends
    # to the same count as begs.
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]
    num_qrs = min(beg_qrs.size, end_qrs.size)
    if num_qrs == 0:
        return np.empty(0, dtype=np.int64)

    region_lens = end_qrs[:num_qrs] - beg_qrs[:num_qrs]
    if region_lens.size == 0:
        return np.empty(0, dtype=np.int64)
    min_len = float(np.mean(region_lens)) * minlenweight

    accepted: list[int] = []
    last_peak = -mindelay - 1
    for i in range(num_qrs):
        beg = int(beg_qrs[i])
        end = int(end_qrs[i])
        if (end - beg) < min_len:
            continue
        # Find the most prominent local maximum within [beg, end] in
        # both sig and -sig (so inverted leads work). Pick the larger
        # prominence overall.
        seg = sig[beg:end]
        if seg.size < 3:
            peak = beg + int(np.argmax(np.abs(seg))) if seg.size else beg
        else:
            best_peak = -1
            best_prom = -np.inf
            for s in (seg, -seg):
                locmax, props = find_peaks(s, prominence=(None, None))
                if locmax.size == 0:
                    continue
                idx = int(np.argmax(props["prominences"]))
                prom = float(props["prominences"][idx])
                if prom > best_prom:
                    best_prom = prom
                    best_peak = beg + int(locmax[idx])
            if best_peak < 0:
                # No internal local max in either polarity — fall back.
                peak = beg + int(np.argmax(np.abs(seg)))
            else:
                peak = best_peak
        if peak - last_peak > mindelay:
            accepted.append(peak)
            last_peak = peak

    peaks = np.asarray(accepted, dtype=np.int64)
    if not return_widths:
        return peaks
    widths_ms = measure_qrs_widths(sig, fs, peaks)
    return peaks, widths_ms


__all__ = ["detect_qrs", "measure_qrs_widths"]
