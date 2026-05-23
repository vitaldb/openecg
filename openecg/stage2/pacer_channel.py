# openecg/stage2/pacer_channel.py
"""Pacer-spike channel computation for input-channel-prior models.

Three flavors:

* ``pacer_channel_from_signal`` — binary indicator at ``target_fs`` Hz
  built by running ``detect_spikes_adaptive`` (40 Hz HP + percentile
  threshold). Validated to fire on real LUDB / BUT PDB paced records.
  Keeps the original convention for backward compatibility.

* ``pacer_detect_pacings_channel`` — binary indicator like the above but
  uses ``openecg.detect_pacings`` (the public 4-channel AND-gated detector,
  100% spec on LUDB + PTB-XL holdouts). Strictly the canonical entry
  point so v22+ scripts use the same detector that ``openecg`` exposes
  as its top-level pacer detector.

* ``pacer_bandpass_channel`` — *continuous* channel: |bandpass(sig)|
  in a narrow band (default 30-50 Hz), per-record normalized by 5×
  robust sigma and clipped to [0, 1]. Avoids the binary-threshold
  brittleness on wider-bandwidth records (LUDB sinus QRS edges trigger
  any single threshold). The model's first conv layer learns the
  amplitude / temporal pattern that distinguishes paced rhythms.

Empirical motivation: ``scripts/probe_pacer_*`` show that no fixed
binary detector cleanly separates LUDB sinus from paced cohorts, while
the band-pass amplitude profile DOES (paced records have narrow bursts
at every QRS, sinus records have small wide bumps). See
``probe_pacer_continuous_channel.py`` and ``probe_pacer_dsq.py`` for the
data behind the choice.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as scipy_signal

from openecg.pacer import detect_pacings, detect_spikes_adaptive

DEFAULT_DETECT_FS = 500
DEFAULT_BROADEN_MS = 10.0
DEFAULT_BP_LO_HZ = 30.0
DEFAULT_BP_HI_HZ = 50.0
DEFAULT_NORM_MAD = 5.0
DEFAULT_SLOPE_CLIP_UV_PER_MS = 200.0


def pacer_channel_from_signal(
    sig: np.ndarray,
    fs_in: int,
    target_fs: int = 250,
    detect_fs: int = DEFAULT_DETECT_FS,
    broaden_ms: float = DEFAULT_BROADEN_MS,
) -> np.ndarray:
    """Return a binary float32 indicator at ``target_fs``.

    Output length is round(len(sig) * target_fs / fs_in). Each detected
    spike contributes a 1.0 boxcar of ``broaden_ms`` ms centered on its
    target-fs sample index (overlapping boxcars saturate at 1.0).

    When ``fs_in < detect_fs`` the signal is resampled up to ``detect_fs``
    only for detection (so 1-3 ms spike widths fit the sample period);
    the indicator is then mapped back to ``target_fs``.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n_in = len(sig)
    n_out = 0 if n_in == 0 else int(round(n_in * target_fs / fs_in))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)

    if fs_in >= detect_fs:
        sig_det = sig
        fs_det = fs_in
    else:
        n_det = max(2, int(round(n_in * detect_fs / fs_in)))
        sig_det = scipy_signal.resample(sig, n_det)
        fs_det = detect_fs

    spikes_det = detect_spikes_adaptive(sig_det, fs=fs_det)
    if spikes_det.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    spikes_target = np.round(spikes_det * (target_fs / fs_det)).astype(np.int64)

    out = np.zeros(n_out, dtype=np.float32)
    half = max(1, int(round(broaden_ms * target_fs / 1000 / 2)))
    for s in spikes_target:
        lo = max(0, int(s) - half)
        hi = min(n_out, int(s) + half + 1)
        if hi > lo:
            out[lo:hi] = 1.0
    return out


def pacer_detect_pacings_channel(
    sig: np.ndarray,
    fs_in: int,
    target_fs: int = 250,
    detect_fs: int = DEFAULT_DETECT_FS,
    broaden_ms: float = DEFAULT_BROADEN_MS,
    score_thr_mad: float = 6.0,
) -> np.ndarray:
    """Binary pacer-spike indicator using ``openecg.detect_pacings`` (4-ch AND-gated).

    Same output convention as ``pacer_channel_from_signal``: float32 boxcar
    indicator at ``target_fs``, 1.0 for ``broaden_ms`` ms centered on each
    detected spike. The only difference is the detector — ``detect_pacings`` is
    the canonical public entry point (100% spec on LUDB + PTB-XL holdouts),
    while ``pacer_channel_from_signal`` uses the older ``detect_spikes_adaptive``
    (40 Hz HP + adaptive threshold) kept for backward compatibility.

    When ``fs_in < detect_fs``, the signal is resampled up to ``detect_fs``
    so 1-3 ms spike widths are resolvable; spike indices are then mapped
    back to ``target_fs``.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n_in = len(sig)
    n_out = 0 if n_in == 0 else int(round(n_in * target_fs / fs_in))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)

    if fs_in >= detect_fs:
        sig_det = sig
        fs_det = fs_in
    else:
        n_det = max(2, int(round(n_in * detect_fs / fs_in)))
        sig_det = scipy_signal.resample(sig, n_det)
        fs_det = detect_fs

    spikes_det = detect_pacings(sig_det, fs=fs_det, score_thr_mad=score_thr_mad)
    if spikes_det.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    spikes_target = np.round(spikes_det * (target_fs / fs_det)).astype(np.int64)

    out = np.zeros(n_out, dtype=np.float32)
    half = max(1, int(round(broaden_ms * target_fs / 1000 / 2)))
    for s in spikes_target:
        lo = max(0, int(s) - half)
        hi = min(n_out, int(s) + half + 1)
        if hi > lo:
            out[lo:hi] = 1.0
    return out


def pacer_bandpass_channel(
    sig: np.ndarray,
    fs_in: int,
    target_fs: int = 250,
    bp_lo_hz: float = DEFAULT_BP_LO_HZ,
    bp_hi_hz: float = DEFAULT_BP_HI_HZ,
    norm_mad: float = DEFAULT_NORM_MAD,
) -> np.ndarray:
    """Continuous pacer channel: |bandpass(sig, 30-50 Hz)| / (norm_mad·σ),
    clipped to [0, 1] and resampled to ``target_fs``.

    This replaces brittle threshold-based binary detection with an
    amplitude-preserving signal that the model's first conv can use
    directly. Per-record normalization (sigma = 1.4826·MAD of the
    bandpassed signal) cancels gain / bandwidth differences between
    LUDB (~500 Hz wider band) and BUT PDB (~360 Hz, LP @ 30 Hz),
    so the *shape* of the response over the window is what the
    classifier consumes.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n_in = len(sig)
    n_out = 0 if n_in == 0 else int(round(n_in * target_fs / fs_in))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    nyq = fs_in / 2
    hi = min(bp_hi_hz, nyq - 1.0)
    lo = max(0.5, min(bp_lo_hz, hi - 1.0))
    b, a = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    bp = scipy_signal.filtfilt(b, a, sig)
    abs_bp = np.abs(bp)
    sigma = 1.4826 * float(np.median(np.abs(bp - np.median(bp))))
    norm = np.clip(abs_bp / max(1e-9, norm_mad * sigma), 0.0, 1.0).astype(np.float32)
    if n_out == n_in:
        return norm
    return scipy_signal.resample(norm, n_out).astype(np.float32)


def pacer_slope_channel(
    sig: np.ndarray,
    fs_in: int,
    target_fs: int = 250,
    clip_uV_per_ms: float = DEFAULT_SLOPE_CLIP_UV_PER_MS,
) -> np.ndarray:
    """Continuous slope-magnitude channel — Tereshchenko-style preprocessing.

    Computes |dV/dt| in μV/ms (assumes ``sig`` in mV) by ``np.diff(sig) * fs``,
    clips to ``clip_uV_per_ms`` so the channel is in [0, 1], and resamples
    to ``target_fs``. The classical clinical signature of pacer spikes is
    high local slope (>100 μV/ms in single lead per Haq et al., 2021) and
    this is the cleanest single-operation distillation of that prior.

    Per-record absolute scale is preserved (no MAD normalization), since
    LUDB and BUT PDB are stored in physical mV — the clip threshold is
    calibrated in physical units.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n_in = len(sig)
    n_out = 0 if n_in == 0 else int(round(n_in * target_fs / fs_in))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    slope = np.abs(np.diff(sig) * fs_in)                     # μV/ms (sig in mV)
    slope = np.concatenate([slope[:1], slope])
    norm = np.clip(slope / max(1e-9, clip_uV_per_ms), 0.0, 1.0).astype(np.float32)
    if n_out == n_in:
        return norm
    return scipy_signal.resample(norm, n_out).astype(np.float32)
