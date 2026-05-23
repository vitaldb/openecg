# openecg/stage2/multitask_dataset.py
"""Dataset adapters for v17 multi-task training.

Two pieces:

1. `WindowLabelWrapper` — wraps an existing frame-labelled dataset (LUDB,
   ISP, QTDB, synth) and adds a (window_labels, window_mask) trailing pair
   derived from the QRS run-length structure of the frame labels.

2. `LydusMultiTaskDataset` — standalone dataset over the SNUH Lydus npz +
   duckdb. Emits a fully-IGNORE frame label (so frame loss is silently
   masked) plus zero reg targets / mask, and the real window labels read
   from lydus metadata.

Both datasets yield the same 7-tuple shape:
    (sig[2500] float32, lead_id long, frame_labels[500] long,
     reg_t[500, 6] float, reg_m[500, 6] bool,
     window_labels[K] long, window_mask[K] bool)
where K = `window_labels.N_WINDOW_TASKS`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as _ee
from openecg import lydus
from openecg.stage2 import window_labels as _wl

WINDOW_SAMPLES = 2500
WINDOW_FRAMES = 500
N_REG = 6


class WindowLabelWrapper(Dataset):
    """Wrap a 3-tuple-or-5-tuple frame dataset, adding window labels.

    `base[idx]` may return either:
      (sig, lead, frame_labels)                          ← LUDBFrameDataset, ...
      (sig, lead, frame_labels, reg_t, reg_m)            ← RegLabelDataset(...)

    The wrapper emits the canonical 7-tuple in either case. When the base
    has no reg targets, zeros are substituted with mask=False so the reg
    loss naturally drops out.

    Window labels are computed from the (un-IGNORE) frame structure once
    per __getitem__. The cost is ~10 µs per window — negligible vs the
    rest of the loader.
    """

    def __init__(self, base: Dataset, *,
                 qrs_wide_ms: float = _wl.QRS_WIDE_MS_DEFAULT,
                 rr_irreg_sd_ms: float = _wl.RR_IRREG_SD_MS_DEFAULT,
                 force_window_labels: tuple[int | None, int | None] | None = None):
        """force_window_labels=(rr, wide) overrides the derived labels with
        the provided scalars (use None for "mask out"). Used by the synth
        wrapper where labels are deterministic from scenario, not from
        derived QRS run-lengths.
        """
        self.base = base
        self.qrs_wide_ms = qrs_wide_ms
        self.rr_irreg_sd_ms = rr_irreg_sd_ms
        self.force = force_window_labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) == 3:
            sig, lead, frame = item
            reg_t = torch.zeros(WINDOW_FRAMES, N_REG, dtype=torch.float32)
            reg_m = torch.zeros(WINDOW_FRAMES, N_REG, dtype=torch.bool)
        elif len(item) == 5:
            sig, lead, frame, reg_t, reg_m = item
        else:
            raise ValueError(
                f"WindowLabelWrapper: base must yield 3- or 5-tuple, "
                f"got {len(item)}"
            )
        if self.force is not None:
            win_l, win_m = _wl.from_components(
                rr_regular=self.force[0], qrs_wide=self.force[1],
            )
        else:
            win_l, win_m = _wl.derive_from_frame_labels(
                frame.numpy() if isinstance(frame, torch.Tensor) else frame,
                qrs_wide_ms=self.qrs_wide_ms,
                rr_irreg_sd_ms=self.rr_irreg_sd_ms,
            )
        return (sig, lead, frame, reg_t, reg_m,
                torch.from_numpy(win_l), torch.from_numpy(win_m))

    def label_counts(self):
        if hasattr(self.base, "label_counts"):
            return self.base.label_counts()
        return None


class _SynthMultiTaskAdapter(Dataset):
    """Specialized wrapper for SyntheticAVBDataset that knows scenario→label
    mapping. We can't use WindowLabelWrapper directly because the synth's
    frame labels are correctly placed but the rr_regular bit for Mobitz/
    complete is rhythmic ground truth, not derivable from frames alone.
    """

    def __init__(self, synth_reg_dataset, scenarios: Sequence[str]):
        self.base = synth_reg_dataset
        self.scenarios = tuple(scenarios)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        # SyntheticAVBDataset.__getitem__ uses `idx % n_scenarios` to pick
        # scenario. RegLabelDataset preserves index, so we recover scenario
        # the same way here. is_ventricular_escape is per-call random in
        # synth, but we can't observe it from the wrapped output → assume
        # the typical case (vent escape on complete) which gives wide QRS;
        # this is a deliberate simplification, the synth's frame labels are
        # the source of truth for the qrs_wide bit so we re-derive it.
        item = self.base[idx]
        if len(item) == 5:
            sig, lead, frame, reg_t, reg_m = item
        else:
            raise ValueError("expected 5-tuple from RegLabelDataset(synth)")
        scenario = self.scenarios[idx % len(self.scenarios)]
        # Always derive qrs_wide from the actual frames (handles both
        # narrow junctional and wide ventricular escape under "complete").
        derived_l, derived_m = _wl.derive_from_frame_labels(
            frame.numpy() if isinstance(frame, torch.Tensor) else frame,
        )
        if scenario in ("mobitz1", "mobitz2"):
            # RR not truly regular; mask. qrs_wide narrow by construction.
            win_l = np.array([0, 0], dtype=np.int64)
            win_m = np.array([False, True], dtype=bool)
        elif scenario in ("complete", "paced"):
            # RR regular for both (independent ventricular schedule).
            # qrs_wide trust the frame-derived bit (ventricular vs junctional
            # escape under "complete" yields different widths).
            win_l = np.array([1, derived_l[1]], dtype=np.int64)
            win_m = np.array([True, derived_m[1]], dtype=bool)
        else:
            raise ValueError(f"unknown scenario: {scenario!r}")
        return (sig, lead, frame, reg_t, reg_m,
                torch.from_numpy(win_l), torch.from_numpy(win_m))

    def label_counts(self):
        if hasattr(self.base, "label_counts"):
            return self.base.label_counts()
        return None


class LydusMultiTaskDataset(Dataset):
    """SNUH Lydus dataset for window-only supervision.

    Each __getitem__ returns one (window, lead) pair. The window is loaded
    lazily from the memory-mapped npz, resampled 100→250 Hz, z-normalized,
    and emitted with frame labels = IGNORE_INDEX everywhere (so frame and
    aux losses naturally drop out).

    Args:
        windows: list of `lydus.LydusWindow` (filtered subset, e.g. only
            train rids).
        leads: subset of channel names from `lydus.LEADS_8` to use. Default
            uses `SAFE_LEADS` (i, ii, v1, v2, v5) — same set as our synth
            and frame datasets.
    """

    def __init__(self, windows, leads: Sequence[str] | None = None):
        self.windows = list(windows)
        if leads is None:
            leads = lydus.SAFE_LEADS
        self.lead_names = tuple(leads)
        self.lead_channels: list[int] = []
        for name in self.lead_names:
            if name not in lydus.LEADS_8:
                raise ValueError(f"lead {name!r} not in LEADS_8")
            self.lead_channels.append(lydus.LEADS_8.index(name))
        # Flat index of (window, lead) pairs.
        self._index: list[tuple[int, int]] = []
        for wi in range(len(self.windows)):
            for ci in self.lead_channels:
                self._index.append((wi, ci))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        wi, ci = self._index[idx]
        win = self.windows[wi]
        sig = lydus.load_signal(win.npz_idx, ci, fs_target=250)
        sig = (sig - sig.mean()) / (sig.std() + 1e-6)
        sig = sig.astype(np.float32)
        lead_id = lydus.lead_id_for(ci)
        if lead_id < 0:
            lead_id = 0
        frame = torch.full((WINDOW_FRAMES,), _ee.IGNORE_INDEX, dtype=torch.long)
        reg_t = torch.zeros(WINDOW_FRAMES, N_REG, dtype=torch.float32)
        reg_m = torch.zeros(WINDOW_FRAMES, N_REG, dtype=torch.bool)
        win_l, win_m = _wl.from_components(
            rr_regular=win.rr_regular, qrs_wide=win.qrs_wide,
        )
        return (
            torch.from_numpy(sig),
            torch.tensor(lead_id, dtype=torch.long),
            frame, reg_t, reg_m,
            torch.from_numpy(win_l), torch.from_numpy(win_m),
        )

    def label_counts(self) -> np.ndarray:
        # Frame labels are entirely IGNORE → contribute no class counts.
        # Return zeros so ConcatDataset.label_counts() won't bias weights.
        return np.zeros(4, dtype=np.int64)
