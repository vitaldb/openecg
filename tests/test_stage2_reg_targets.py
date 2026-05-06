import numpy as np
import pytest
import torch

from openecg import eval as ee
from openecg.stage2.reg_targets import (
    REG_CHANNELS, boundary_regression_targets,
)


def test_reg_channels_order():
    assert REG_CHANNELS == ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")


def test_no_transitions_yields_no_active_mask():
    labels = np.full(10, ee.SUPER_OTHER, dtype=np.int64)
    targets, mask = boundary_regression_targets(
        labels, samples_per_frame=5, window_frames=5,
    )
    assert targets.shape == (10, 6)
    assert mask.shape == (10, 6)
    assert not mask.any()


def test_single_p_wave_marks_two_channels():
    # P-wave from frame 5 to frame 10 (samples 25..49 inclusive given spf=5)
    labels = np.array([0]*5 + [1]*6 + [0]*5, dtype=np.int64)
    targets, mask = boundary_regression_targets(
        labels, samples_per_frame=5, window_frames=5,
    )
    p_on_active = mask[:, 0]
    assert p_on_active[5]
    assert p_on_active[0]
    assert not p_on_active[15]
    assert targets[5, 0] == 0.0
    assert targets[4, 0] == 5.0
    assert mask[10, 1]
    assert targets[10, 1] == (11 * 5 - 1) - (10 * 5)


def test_ignore_index_zeros_target_and_mask():
    labels = np.array([0, 0, ee.IGNORE_INDEX, 1, 1], dtype=np.int64)
    targets, mask = boundary_regression_targets(
        labels, samples_per_frame=5, window_frames=5,
    )
    assert not mask[2].any()
    np.testing.assert_array_equal(targets[2], 0.0)


import torch

from openecg.stage2.reg_targets import RegLabelDataset


class _HardDS:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sig = torch.zeros(2500)
        lead = torch.tensor(idx, dtype=torch.long)
        labels = torch.tensor([0]*5 + [1]*6 + [0]*489, dtype=torch.long)
        return sig, lead, labels

    def label_counts(self):
        return np.array([100, 5, 5, 5], dtype=np.int64)


def test_reg_label_dataset_yields_targets_and_mask():
    base = _HardDS()
    ds = RegLabelDataset(base, samples_per_frame=5, window_frames=5)
    sig, lead, labels, target, mask = ds[0]
    assert sig.shape == (2500,)
    assert labels.dtype == torch.long
    assert target.shape == (500, 6)
    assert mask.shape == (500, 6)
    assert mask.dtype == torch.bool
    assert mask.any()
