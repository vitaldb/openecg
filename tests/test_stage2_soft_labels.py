import numpy as np
import pytest
import torch

from openecg import eval as ee
from openecg.stage2.soft_labels import SoftLabelDataset, soft_boundary_labels


def test_no_transitions_produces_one_hot():
    labels = np.array([1, 1, 1, 1], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    assert soft.shape == (4, 4)
    assert soft.dtype == np.float32
    expected = np.array([[0, 1, 0, 0]] * 4, dtype=np.float32)
    np.testing.assert_array_almost_equal(soft, expected)


def test_single_transition_softens_two_frames():
    # other(0) -> P(1) at frames 1->2
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    np.testing.assert_almost_equal(soft[1, 0], 0.7)
    np.testing.assert_almost_equal(soft[1, 1], 0.3)
    np.testing.assert_almost_equal(soft[2, 0], 0.3)
    np.testing.assert_almost_equal(soft[2, 1], 0.7)
    np.testing.assert_almost_equal(soft[0, 0], 1.0)
    np.testing.assert_almost_equal(soft[3, 1], 1.0)


def test_ignore_index_produces_zero_row():
    labels = np.array([1, ee.IGNORE_INDEX, 2, 2], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    np.testing.assert_array_almost_equal(soft[1], np.zeros(4, dtype=np.float32))
    np.testing.assert_almost_equal(soft[0, 1], 1.0)
    np.testing.assert_almost_equal(soft[2, 2], 1.0)


def test_mass_conservation_on_valid_rows():
    labels = np.array([0, 1, 2, 3, 0, ee.IGNORE_INDEX], dtype=np.int64)
    soft = soft_boundary_labels(labels, alpha=0.7, n_classes=4)
    sums = soft.sum(axis=-1)
    np.testing.assert_array_almost_equal(sums[:-1], np.ones(5, dtype=np.float32))
    np.testing.assert_almost_equal(sums[-1], 0.0)


class _HardDS:
    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        self.calls += 1
        sig = torch.zeros(2500)
        lead = torch.tensor(idx, dtype=torch.long)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        return sig, lead, labels

    def label_counts(self):
        return np.array([10, 10, 10, 10], dtype=np.int64)


def test_soft_label_dataset_returns_soft_tensor():
    base = _HardDS()
    ds = SoftLabelDataset(base, alpha=0.7, n_classes=4)
    assert len(ds) == 2
    sig, lead, soft = ds[0]
    assert sig.shape == (2500,)
    assert lead.dtype == torch.long
    assert soft.shape == (8, 4)
    assert soft.dtype == torch.float32
    assert torch.isclose(soft[0, 0], torch.tensor(1.0))
    assert torch.isclose(soft[1, 0], torch.tensor(0.7))
    assert torch.isclose(soft[1, 1], torch.tensor(0.3))


def test_soft_label_dataset_passes_through_label_counts():
    ds = SoftLabelDataset(_HardDS())
    counts = ds.label_counts()
    np.testing.assert_array_equal(counts, [10, 10, 10, 10])
