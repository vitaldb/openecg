import os
import numpy as np
import pytest
import torch

LUDB_AVAILABLE = bool(os.environ.get("ECGCODE_LUDB_ZIP"))
pytestmark = pytest.mark.skipif(not LUDB_AVAILABLE, reason="ECGCODE_LUDB_ZIP not set")


def test_dataset_basic_shapes():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1, 2])
    assert len(ds) > 0
    sig, lead_id, labels = ds[0]
    assert sig.dtype == torch.float32
    assert sig.shape == (2500,)
    assert lead_id.dtype == torch.long
    assert lead_id.shape == ()
    assert 0 <= int(lead_id) < 12
    assert labels.dtype == torch.long
    assert labels.shape == (500,)


def test_dataset_labels_in_supercategory_range():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1, 2])
    sig, lead_id, labels = ds[0]
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 3


def test_dataset_signal_normalized():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1, 2])
    sig, _, _ = ds[0]
    assert abs(float(sig.mean())) < 0.1
    assert abs(float(sig.std()) - 1.0) < 0.1


def test_dataset_covers_all_leads():
    from ecgcode.stage2.dataset import LUDBFrameDataset
    ds = LUDBFrameDataset(record_ids=[1])
    leads_seen = set()
    for i in range(len(ds)):
        _, lead_id, _ = ds[i]
        leads_seen.add(int(lead_id))
    assert leads_seen == set(range(12))


def test_compute_class_weights_inverse_sqrt():
    from ecgcode.stage2.dataset import compute_class_weights
    counts = np.array([600, 100, 100, 200], dtype=np.float64)
    weights = compute_class_weights(counts)
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]
    assert abs(weights.sum() - 4.0) < 1e-6
