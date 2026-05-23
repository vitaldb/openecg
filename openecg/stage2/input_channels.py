# openecg/stage2/input_channels.py
"""Compose multi-channel input tensors for v22 input-channel-prior models.

A model trained with ``FrameClassifierViTRegMultiIn`` expects input of
shape [B, C, T] where C >= 1 and channel 0 is always the (z-normed) ECG
sample stream. Optional channels carry rule-based priors derived by
preprocessing on the same window. Composition order (after the signal):
    1. pacer indicator   (`with_pacer=True`)
    2. QRS-position indicator (`with_qrs=True`)
The order is fixed so checkpoints record their channel composition via
``model_config['input_channels']`` and downstream eval / inference can
reconstruct it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from openecg.stage2.pacer_channel import (
    pacer_bandpass_channel, pacer_channel_from_signal,
    pacer_detect_pacings_channel, pacer_slope_channel,
)
from openecg.stage2.qrs_channel import (
    qrs_position_channel, qrs_position_channel_from_indices,
)


CHANNEL_NAMES_BY_FLAGS: dict[tuple[bool, bool], tuple[str, ...]] = {
    (False, False): ("signal",),
    (True,  False): ("signal", "pacer"),
    (False, True):  ("signal", "qrs"),
    (True,  True):  ("signal", "pacer", "qrs"),
}


def channel_names(with_pacer: bool, with_qrs: bool) -> tuple[str, ...]:
    return CHANNEL_NAMES_BY_FLAGS[(bool(with_pacer), bool(with_qrs))]


def compute_input_channels(
    sig_target: np.ndarray,
    *,
    sig_native: np.ndarray | None = None,
    fs_native: int | None = None,
    target_fs: int = 250,
    with_pacer: bool = False,
    with_qrs: bool = False,
    qrs_indices_native: np.ndarray | None = None,
    pacer_mode: str = "slope",
    qrs_broaden_ms: float = 40.0,
) -> np.ndarray:
    """Return [C, T] np.float32 with channels stacked per the flags.

    The signal channel is always ``sig_target`` (already at ``target_fs``
    and z-normed by the caller).

    Pacer / QRS channels are detected on ``sig_native`` at ``fs_native``
    when those are provided (preferred — full bandwidth); otherwise they
    fall back to ``sig_target`` at ``target_fs`` (lossy for pacer spikes,
    fine for QRS).

    ``qrs_indices_native`` (in fs_native sample coords) bypasses
    neurokit2 detection for the QRS channel — used by training paths that
    want a deterministic GT-based channel.
    """
    sig_target = np.asarray(sig_target, dtype=np.float32)
    n_t = len(sig_target)
    chans: list[np.ndarray] = [sig_target]

    if with_pacer:
        if pacer_mode == "binary":
            pacer_fn = pacer_channel_from_signal
        elif pacer_mode == "detect_pacings":
            pacer_fn = pacer_detect_pacings_channel
        elif pacer_mode == "bandpass":
            pacer_fn = pacer_bandpass_channel
        elif pacer_mode == "slope":
            pacer_fn = pacer_slope_channel
        else:
            raise ValueError(f"unknown pacer_mode {pacer_mode!r}")
        if sig_native is not None and fs_native is not None:
            ch = pacer_fn(sig_native, fs_in=int(fs_native), target_fs=target_fs)
        else:
            ch = pacer_fn(sig_target, fs_in=target_fs, target_fs=target_fs)
        ch = _fit_length(ch, n_t)
        chans.append(ch)

    if with_qrs:
        if qrs_indices_native is not None and sig_native is not None and fs_native is not None:
            ch = qrs_position_channel_from_indices(
                qrs_indices_native, fs_in=int(fs_native),
                n_in_samples=len(sig_native), target_fs=target_fs,
                broaden_ms=qrs_broaden_ms,
            )
        elif sig_native is not None and fs_native is not None:
            ch = qrs_position_channel(sig_native, fs_in=int(fs_native),
                                       target_fs=target_fs,
                                       broaden_ms=qrs_broaden_ms)
        else:
            ch = qrs_position_channel(sig_target, fs_in=target_fs,
                                       target_fs=target_fs,
                                       broaden_ms=qrs_broaden_ms)
        ch = _fit_length(ch, n_t)
        chans.append(ch)

    return np.stack(chans, axis=0).astype(np.float32)


def _fit_length(ch: np.ndarray, n: int) -> np.ndarray:
    """Pad / truncate ``ch`` to length ``n`` so all channels align."""
    if len(ch) == n:
        return ch
    out = np.zeros(n, dtype=np.float32)
    m = min(len(ch), n)
    out[:m] = ch[:m]
    return out
