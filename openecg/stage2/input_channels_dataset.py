# openecg/stage2/input_channels_dataset.py
"""Wrap a base Stage 2 dataset so each sample's signal becomes a [C, T]
multi-channel tensor (signal + optional pacer + optional QRS).

Pacer channel uses a fast preprocessing (slope, bandpass, or binary) on
the already-decimated 250 Hz signal.

QRS channel ``qrs_source``:
  * ``"detect_qrs"`` (DEFAULT) — run ``openecg.detect_qrs`` on the
    signal at every step. Matches inference exactly so the model sees
    realistic noisy R-peak indicators during training. Use this for
    deployment-grade training.
  * ``"gt"`` — derive R-peaks from the sample's GT frame labels. Cheap
    but creates a teacher-forcing-style train/inference distribution
    mismatch (model overfits to perfect QRS positions, then collapses
    on real signals where ``detect_qrs`` is jittery / misses beats).
    Kept for the rare case the caller wants a clean upper-bound study.

Output tuple:
    (sig_multi[C, T] float32, lead_id long, labels long, *rest)
where C ∈ {1, 2, 3} depending on the (with_pacer, with_qrs) flags and
``*rest`` carries any trailing tensors emitted by the base dataset
(e.g. RegLabelDataset's reg_targets and reg_mask).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as ee
from openecg.stage2.pacer_channel import (
    pacer_bandpass_channel, pacer_channel_from_signal,
    pacer_detect_pacings_channel, pacer_slope_channel,
)
from openecg.stage2.qrs_channel import (
    qrs_position_channel, qrs_position_channel_from_indices,
)


_QRS_CLASSES = (ee.SUPER_QRS, ee.SUPER_PACED_QRS)
PACER_MODES = ("slope", "binary", "detect_pacings", "bandpass")
QRS_SOURCES = ("detect_qrs", "gt")


def _qrs_samples_from_frame_labels(labels: np.ndarray, n_samples: int) -> np.ndarray:
    """QRS_on sample indices (250 Hz grid) derived from frame labels.

    A new QRS_on event is emitted at each transition from "not QRS" to
    "QRS or PACED_QRS". IGNORE_INDEX frames break the run so a QRS that
    starts at the very edge of a masked region is ignored (consistent
    with how the model is supervised).
    """
    labels = np.asarray(labels)
    n_frames = len(labels)
    if n_frames == 0:
        return np.empty(0, dtype=np.int64)
    spf = max(1, int(round(n_samples / n_frames)))
    qrs_starts: list[int] = []
    prev_in_qrs = False
    for i, l in enumerate(labels):
        l_int = int(l)
        in_qrs = l_int in _QRS_CLASSES
        if in_qrs and not prev_in_qrs:
            qrs_starts.append(i * spf)
        prev_in_qrs = in_qrs and (l_int != ee.IGNORE_INDEX)
    return np.asarray(qrs_starts, dtype=np.int64)


def _pacer_fn_for(mode: str):
    if mode == "slope":
        return pacer_slope_channel
    if mode == "binary":
        return pacer_channel_from_signal
    if mode == "detect_pacings":
        return pacer_detect_pacings_channel
    if mode == "bandpass":
        return pacer_bandpass_channel
    raise ValueError(f"unknown pacer_mode {mode!r}; expected one of {PACER_MODES}")


class InputChannelsDataset(Dataset):
    """See module docstring."""

    def __init__(
        self,
        base: Dataset,
        with_pacer: bool = True,
        with_qrs: bool = False,
        fs: int = 250,
        pacer_mode: str = "slope",
        qrs_broaden_ms: float = 40.0,
        qrs_source: str = "detect_qrs",
    ):
        if qrs_source not in QRS_SOURCES:
            raise ValueError(
                f"qrs_source must be one of {QRS_SOURCES}, got {qrs_source!r}"
            )
        self.base = base
        self.with_pacer = bool(with_pacer)
        self.with_qrs = bool(with_qrs)
        self.fs = int(fs)
        self.pacer_mode = pacer_mode
        self.qrs_broaden_ms = float(qrs_broaden_ms)
        self.qrs_source = qrs_source
        self._pacer_fn = _pacer_fn_for(pacer_mode) if self.with_pacer else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if len(item) < 3:
            raise ValueError(
                f"base dataset must return >= 3 elements (sig, lead_id, labels); "
                f"got {len(item)} elements"
            )
        sig = item[0]
        lead_id = item[1]
        labels = item[2]
        rest = item[3:]

        sig_np = sig.detach().cpu().numpy() if isinstance(sig, torch.Tensor) else np.asarray(sig, dtype=np.float32)
        labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        n = len(sig_np)

        chans: list[np.ndarray] = [sig_np.astype(np.float32, copy=False)]
        if self.with_pacer:
            ch = self._pacer_fn(sig_np, fs_in=self.fs, target_fs=self.fs)
            chans.append(_fit_length(ch, n))
        if self.with_qrs:
            if self.qrs_source == "gt":
                qrs_samples = _qrs_samples_from_frame_labels(labels_np, n_samples=n)
                ch = qrs_position_channel_from_indices(
                    qrs_samples, fs_in=self.fs, n_in_samples=n,
                    target_fs=self.fs, broaden_ms=self.qrs_broaden_ms,
                )
            else:                                                # "detect_qrs"
                ch = qrs_position_channel(
                    sig_np, fs_in=self.fs, target_fs=self.fs,
                    broaden_ms=self.qrs_broaden_ms,
                )
            chans.append(_fit_length(ch, n))

        sig_multi = np.stack(chans, axis=0).astype(np.float32)
        out_tail = tuple(rest)
        return (torch.from_numpy(sig_multi),
                lead_id if isinstance(lead_id, torch.Tensor)
                       else torch.tensor(int(lead_id), dtype=torch.long),
                labels if isinstance(labels, torch.Tensor)
                       else torch.from_numpy(np.asarray(labels)),
                *out_tail)

    def label_counts(self):
        return self.base.label_counts()

    @property
    def items(self):
        if hasattr(self.base, "items"):
            return self.base.items
        return None


def _fit_length(ch: np.ndarray, n: int) -> np.ndarray:
    if len(ch) == n:
        return ch.astype(np.float32, copy=False)
    out = np.zeros(n, dtype=np.float32)
    m = min(len(ch), n)
    out[:m] = ch[:m]
    return out
