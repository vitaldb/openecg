# openecg/stage2/synth_dataset.py
"""Torch Dataset wrapper around openecg.synth.generate_avb_window.

Yields (signal[2500] float32, lead_id long, frame_labels[500] int64) tuples
in the same shape as LUDBFrameDataset / QTDBSlidingDataset, so it composes
trivially into ConcatDataset for training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as ee
from openecg import synth, ludb

LEAD_NAME_TO_ID = {name: i for i, name in enumerate(ludb.LEADS_12)}
WINDOW_SAMPLES = 2500
WINDOW_FRAMES = 500
SAMPLES_PER_FRAME = WINDOW_SAMPLES // WINDOW_FRAMES   # 5
N_CLASSES = 4


def _labels_to_frame_array(labels: dict[str, list[int]]) -> np.ndarray:
    """Convert the synth label dict (sample-level on/off lists) into a
    frame-level 4-class array via majority-of-5-samples voting.
    """
    sample_labels = np.full(WINDOW_SAMPLES, ee.SUPER_OTHER, dtype=np.uint8)
    # Order matters: later assignments overwrite earlier ones at overlapping
    # samples. In 3°AVB, P often lands on T or near a QRS; we want the model
    # to learn P even when buried in T amplitude, so P is written LAST and
    # wins over T at overlap. QRS still wins over P (a P that falls inside a
    # QRS is essentially invisible and not what we want to teach).
    for cls_id, on_key, off_key in (
        (ee.SUPER_T,   "t_on",   "t_off"),
        (ee.SUPER_P,   "p_on",   "p_off"),
        (ee.SUPER_QRS, "qrs_on", "qrs_off"),
    ):
        for on, off in zip(labels.get(on_key, []), labels.get(off_key, [])):
            lo = max(0, int(on))
            hi = min(WINDOW_SAMPLES, int(off) + 1)
            if hi > lo:
                sample_labels[lo:hi] = cls_id

    out = np.zeros(WINDOW_FRAMES, dtype=np.int64)
    for f in range(WINDOW_FRAMES):
        seg = sample_labels[f * SAMPLES_PER_FRAME:(f + 1) * SAMPLES_PER_FRAME]
        vals, counts = np.unique(seg, return_counts=True)
        out[f] = int(vals[np.argmax(counts)])
    return out


class SyntheticAVBDataset(Dataset):
    """Synthetic AV-block dataset with on-the-fly generation.

    Args:
        bank: pre-built TemplateBank.
        leads: tuple of LUDB lead names whose templates exist in `bank`
            (e.g. ('ii', 'v1', 'i', 'v5')).
        scenarios: tuple of scenario names; default = all three.
        n_windows: dataset length; one fresh window is generated per index.
        base_seed: deterministic per-index seeding so the same idx yields the
            same window across epochs (helpful for debugging). Set to None to
            re-seed every call.
    """

    def __init__(self,
                 bank: synth.TemplateBank,
                 leads: tuple[str, ...] = ("ii", "v1", "i", "v5", "v2"),
                 scenarios: tuple[str, ...] = ("mobitz1", "mobitz2", "complete"),
                 n_windows: int = 2000,
                 base_seed: int | None = 12345):
        self.bank = bank
        self.leads = tuple(leads)
        self.scenarios = tuple(scenarios)
        self.n_windows = int(n_windows)
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        if self.base_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.base_seed + idx)
        scenario = self.scenarios[idx % len(self.scenarios)]
        lead = self.leads[(idx // len(self.scenarios)) % len(self.leads)]
        sig, labels = synth.generate_avb_window(
            self.bank, lead, scenario, rng,
            fs=250, duration_s=10.0,
        )
        frame_labels = _labels_to_frame_array(labels)
        lead_id = LEAD_NAME_TO_ID[lead]
        return (
            torch.from_numpy(sig),
            torch.tensor(lead_id, dtype=torch.long),
            torch.from_numpy(frame_labels),
        )

    def label_counts(self) -> np.ndarray:
        """Estimate per-class frame counts from a 100-window sample so the
        downstream class-weight computation works."""
        rng = np.random.default_rng(0)
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        n_sample = 100
        for k in range(n_sample):
            scenario = self.scenarios[k % len(self.scenarios)]
            lead = self.leads[(k // len(self.scenarios)) % len(self.leads)]
            _, labels = synth.generate_avb_window(
                self.bank, lead, scenario, rng, fs=250, duration_s=10.0,
            )
            frames = _labels_to_frame_array(labels)
            for c in range(N_CLASSES):
                counts[c] += int((frames == c).sum())
        # Scale up to the dataset size.
        scale = self.n_windows / max(1, n_sample)
        return np.maximum((counts * scale).astype(np.int64), 1)
