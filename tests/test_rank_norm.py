"""Tests for v45m1 input preprocessing: rank_normalize + _RankNormalizedDataset
+ _TWaveFlipDataset wrappers.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.retrain_v40_common import (
    CLASS_T, _RankNormalizedDataset, _TWaveFlipDataset, rank_normalize,
)


# ---------------------------------------------------- rank_normalize ----

def test_rank_normalize_range_and_endpoints():
    """Output is in [-1, 1], extremes hit endpoints."""
    sig = np.array([5.0, -3.0, 0.0, 10.0, 7.5], dtype=np.float64)
    out = rank_normalize(sig)
    assert out.dtype == np.float32
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)
    assert -1.0 <= out.min() and out.max() <= 1.0


def test_rank_normalize_monotonic_ordering_preserved():
    """rank(x_i) > rank(x_j) iff x_i > x_j."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(2500).astype(np.float64)
    out = rank_normalize(sig)
    # Sort both; rank-of-sorted should also be sorted
    sort_idx = np.argsort(sig)
    sorted_ranks = out[sort_idx]
    assert np.all(np.diff(sorted_ranks) >= 0), \
        "monotonic ordering not preserved by rank transform"


def test_rank_normalize_amplitude_invariant():
    """rank(x) == rank(c * x) for any positive c (monotonic transform)."""
    rng = np.random.default_rng(7)
    sig = rng.standard_normal(500)
    out_a = rank_normalize(sig)
    out_b = rank_normalize(sig * 100.0)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6,
        err_msg="rank not amplitude-invariant")


def test_rank_normalize_uniform_distribution():
    """Output values approximately uniform on [-1, 1] (KS-style spot check)."""
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(2500)
    out = rank_normalize(sig)
    # 4 equal-width bins should each get roughly 25% mass
    counts, _ = np.histogram(out, bins=4, range=(-1, 1))
    expected = len(sig) / 4
    assert np.all(np.abs(counts - expected) < expected * 0.05), \
        f"rank output not uniform: bin counts={counts.tolist()}"


def test_rank_normalize_empty():
    """Empty input returns empty array (no exception)."""
    out = rank_normalize(np.array([], dtype=np.float64))
    assert out.shape == (0,)
    assert out.dtype == np.float32


# ---------------------------------------------------- _RankNormalizedDataset ----

class _DummyDataset(torch.utils.data.Dataset):
    """Yields (sig[2500], lead_id, labels[500]) tuples like _SynarrdbDataset1Ch."""
    def __init__(self, n: int = 4, multi_channel: bool = False, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.items = []
        for _ in range(n):
            if multi_channel:
                sig = rng.standard_normal((2, 2500)).astype(np.float32)
            else:
                sig = rng.standard_normal(2500).astype(np.float32)
            labels = rng.integers(0, 4, size=500).astype(np.int64)
            self.items.append((
                torch.from_numpy(sig),
                torch.tensor(0, dtype=torch.long),
                torch.from_numpy(labels),
            ))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def test_rank_dataset_1d():
    base = _DummyDataset(n=2, multi_channel=False)
    ds = _RankNormalizedDataset(base)
    sig, lead, labels = ds[0]
    assert isinstance(sig, torch.Tensor)
    assert sig.shape == (2500,)
    sig_np = sig.numpy()
    assert sig_np.min() == pytest.approx(-1.0)
    assert sig_np.max() == pytest.approx(1.0)
    # labels untouched
    assert torch.equal(labels, base[0][2])
    assert lead.item() == 0


def test_rank_dataset_multi_channel():
    """Only channel 0 (signal) rank-normed; channel 1 (e.g. qrs_box) passes through."""
    base = _DummyDataset(n=2, multi_channel=True)
    ds = _RankNormalizedDataset(base)
    sig, _, _ = ds[0]
    assert sig.shape == (2, 2500)
    # Channel 0 should be in [-1, 1] (rank-normed)
    assert sig[0].min() == pytest.approx(-1.0)
    assert sig[0].max() == pytest.approx(1.0)
    # Channel 1 should be unchanged from base
    np.testing.assert_allclose(sig[1].numpy(), base[0][0][1].numpy())


def test_rank_dataset_preserves_extras():
    """Tuples of length > 3 (e.g., RegLabelDataset 5-tuple) pass through extras."""
    class _ExtraDataset(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx):
            return (
                torch.randn(2500),
                torch.tensor(0),
                torch.randint(0, 4, (500,)),
                "extra_a", "extra_b",
            )
    base = _ExtraDataset()
    ds = _RankNormalizedDataset(base)
    out = ds[0]
    assert len(out) == 5
    assert out[3] == "extra_a"
    assert out[4] == "extra_b"


# ---------------------------------------------------- _TWaveFlipDataset ----

def test_tflip_dataset_polarity():
    """When rng < prob: T-frame signal samples are negated; other frames unchanged."""
    # Construct a deterministic base: sig is monotonically increasing,
    # labels: first 100 frames=BG, next 100 frames=T, rest=BG (clear regions).
    sig = np.arange(2500, dtype=np.float32) - 1250
    labels = np.zeros(500, dtype=np.int64)
    labels[100:200] = CLASS_T  # T region: sample idx 500..999

    class _OneItemDataset(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return (
                torch.from_numpy(sig.copy()),
                torch.tensor(0),
                torch.from_numpy(labels.copy()),
            )

    # prob=1.0 → always flip
    ds = _TWaveFlipDataset(_OneItemDataset(), prob=1.0, seed=43)
    out_sig, _, out_labels = ds[0]
    # T region samples should be negated
    np.testing.assert_allclose(
        out_sig.numpy()[500:1000], -sig[500:1000], atol=1e-6,
        err_msg="T region not negated when prob=1.0",
    )
    # Non-T regions untouched
    np.testing.assert_allclose(out_sig.numpy()[:500], sig[:500])
    np.testing.assert_allclose(out_sig.numpy()[1000:], sig[1000:])
    # Labels untouched
    np.testing.assert_array_equal(out_labels.numpy(), labels)


def test_tflip_dataset_prob_zero():
    """prob=0.0 never flips — sig unchanged."""
    sig = np.arange(2500, dtype=np.float32)
    labels = np.full(500, CLASS_T, dtype=np.int64)

    class _OneItemDataset(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return (
                torch.from_numpy(sig.copy()),
                torch.tensor(0),
                torch.from_numpy(labels.copy()),
            )

    ds = _TWaveFlipDataset(_OneItemDataset(), prob=0.0, seed=43)
    out_sig, _, _ = ds[0]
    np.testing.assert_array_equal(out_sig.numpy(), sig)


def test_tflip_dataset_stochasticity():
    """prob=0.5 → roughly half of N draws should flip (chi^2-ish spot check)."""
    sig = np.ones(2500, dtype=np.float32)
    labels = np.full(500, CLASS_T, dtype=np.int64)

    class _ManyDataset(torch.utils.data.Dataset):
        def __len__(self): return 1000
        def __getitem__(self, idx):
            return (
                torch.from_numpy(sig.copy()),
                torch.tensor(0),
                torch.from_numpy(labels.copy()),
            )

    ds = _TWaveFlipDataset(_ManyDataset(), prob=0.5, seed=43)
    n_flipped = 0
    for i in range(1000):
        out, _, _ = ds[i]
        if out[0].item() < 0:
            n_flipped += 1
    # 500 ± 50 expected (4σ for binomial)
    assert 400 <= n_flipped <= 600, f"prob=0.5 produced {n_flipped}/1000 flips"
