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
) -> np.ndarray:
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

    Returns:
        Sorted np.int64 array of R-peak sample indices.
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

    return np.asarray(accepted, dtype=np.int64)


__all__ = ["detect_qrs"]
