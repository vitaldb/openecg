"""Unit tests for InputChannelsDataset wrapper."""
import numpy as np
import torch
from torch.utils.data import Dataset

from openecg import eval as ee
from openecg.stage2.input_channels_dataset import (
    InputChannelsDataset, _qrs_samples_from_frame_labels,
)


class _FakeBase(Dataset):
    """Minimal base dataset emitting (sig, lead_id, labels) for testing."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.items_data = []
        for k in range(3):
            sig = rng.normal(0, 0.05, size=2500).astype(np.float32)
            labels = np.full(500, ee.SUPER_OTHER, dtype=np.int64)
            # Plant a QRS span at frame 100..115.
            labels[100:115] = ee.SUPER_QRS
            self.items_data.append((sig, k % 12, labels))

    def __len__(self):
        return len(self.items_data)

    def __getitem__(self, idx):
        sig, lead_id, labels = self.items_data[idx]
        return (torch.from_numpy(sig.copy()),
                torch.tensor(lead_id, dtype=torch.long),
                torch.from_numpy(labels.copy()))

    def label_counts(self):
        return np.array([1000, 0, 100, 0], dtype=np.int64)


def test_qrs_samples_from_frame_labels():
    labels = np.full(500, ee.SUPER_OTHER, dtype=np.int64)
    labels[100:115] = ee.SUPER_QRS
    labels[200:215] = ee.SUPER_QRS
    qrs = _qrs_samples_from_frame_labels(labels, n_samples=2500)
    # 5 samples per frame; QRS_on at frame 100 -> sample 500, frame 200 -> 1000.
    assert qrs.tolist() == [500, 1000]


def test_signal_only_one_channel():
    ds = InputChannelsDataset(_FakeBase(), with_pacer=False, with_qrs=False)
    sig, lead_id, labels = ds[0][:3]
    assert sig.shape == (1, 2500)
    assert sig.dtype == torch.float32


def test_pacer_only_two_channels():
    ds = InputChannelsDataset(_FakeBase(), with_pacer=True, with_qrs=False,
                                pacer_mode="slope")
    sig, _, _ = ds[0][:3]
    assert sig.shape == (2, 2500)


def test_pacer_and_qrs_three_channels_with_qrs_at_planted_position():
    """qrs_source='gt' uses frame-label-derived QRS positions, so the
    planted QRS span at frame 100 lights up the QRS channel at sample 500."""
    ds = InputChannelsDataset(_FakeBase(), with_pacer=True, with_qrs=True,
                                pacer_mode="slope", qrs_source="gt")
    sig, _, _ = ds[0][:3]
    assert sig.shape == (3, 2500)
    qrs_ch = sig[2]
    assert qrs_ch[500].item() == 1.0
    for off in (-3, -1, 1, 3):
        assert qrs_ch[500 + off].item() == 1.0


def test_qrs_source_detect_qrs_default_does_not_use_labels():
    """Default qrs_source='detect_qrs' runs detect_qrs on the (random-noise)
    signal — it should NOT light up the QRS channel at the GT-planted
    position because there's no R-peak there in the noise. This verifies
    the train/inference channel matches deployment behavior."""
    ds = InputChannelsDataset(_FakeBase(), with_pacer=False, with_qrs=True)
    sig, _, _ = ds[0][:3]
    assert sig.shape == (2, 2500)
    # The fake base dataset emits Gaussian noise (no real QRS), so
    # detect_qrs has no reliable peaks at the GT frame-100 position.
    # We can't assert no spikes (detect_qrs may fire on noise), but we
    # can confirm the channel is NOT the deterministic GT mask:
    qrs_ch = sig[1].numpy()
    # GT path would saturate samples [495, 505]; detect_qrs on noise
    # wouldn't (with overwhelming probability).
    assert not (qrs_ch[495:506] == 1.0).all()


def test_label_counts_passthrough():
    ds = InputChannelsDataset(_FakeBase(), with_pacer=True, with_qrs=False)
    assert ds.label_counts().tolist() == [1000, 0, 100, 0]
