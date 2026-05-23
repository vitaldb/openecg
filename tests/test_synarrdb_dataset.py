"""Smoke tests for ``openecg.stage2.synarrdb_dataset.SynarrdbDataset``.

Skipped automatically when no ``SYNARRDB_DIST_DIR`` env var is set —
running these tests on CI requires the ~1.4 GB synarrdb build, which
is built out-of-tree.

Set the env var to a directory containing ``synarrdb_500hz.npz`` and
``synarrdb.duckdb``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if os.environ.get("SYNARRDB_DIST_DIR") is None:
    pytest.skip(
        "set SYNARRDB_DIST_DIR to a synarrdb build dir to run these tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def dist_paths() -> tuple[Path, Path]:
    root = Path(os.environ["SYNARRDB_DIST_DIR"])
    npz = root / "synarrdb_500hz.npz"
    db = root / "synarrdb.duckdb"
    if not (npz.is_file() and db.is_file()):
        pytest.skip(f"synarrdb build not at {root}")
    return npz, db


def _make_ds(dist_paths, **kw):
    from openecg.stage2.synarrdb_dataset import SynarrdbDataset
    return SynarrdbDataset(dist_paths[0], dist_paths[1], **kw)


def test_val_split_loads(dist_paths):
    ds = _make_ds(dist_paths, split="val", max_windows=200)
    assert len(ds) == 200
    sig, lead_id, frames = ds[0]
    assert sig.shape == (2500,)
    assert sig.dtype == torch.float32
    # z-norm: mean ≈ 0, std ≈ 1.
    assert abs(float(sig.mean())) < 0.05
    assert 0.7 < float(sig.std()) < 1.3
    assert frames.shape == (500,)
    assert frames.dtype == torch.int64
    assert int(frames.max()) < 5
    # lead "ii" is index 1 in openecg.ludb.LEADS_12.
    assert int(lead_id) == 1


def test_scenario_filter(dist_paths):
    ds = _make_ds(dist_paths, split="val", scenarios=("nsr",), max_windows=50)
    assert len(ds) == 50
    # NSR should never produce wide-QRS frames (class 4).
    for i in range(min(10, len(ds))):
        _sig, _lead, frames = ds[i]
        assert int((frames == 4).sum()) == 0


def test_label_counts_sum_matches_total_frames(dist_paths):
    ds = _make_ds(dist_paths, split="val", max_windows=100)
    counts = ds.label_counts()
    # ``label_counts`` sums over the 5000-sample frames buffer, so the
    # total is n × 5000, not n × 500. The function also clamps any zero
    # counts to 1 so the downstream class-balanced loss doesn't divide
    # by zero, which can over-count by up to N_CLASSES = 5 across an
    # otherwise-saturated split (e.g. a 100-window prefix that happens
    # to contain only afib drops both class 1 and class 4 to zero,
    # clamped back up to 1 each).
    expected = len(ds) * 5000
    diff = int(counts.sum()) - expected
    assert 0 <= diff <= 5, (
        f"counts sum {counts.sum()} differs from expected {expected} by {diff}"
    )


def test_dataloader_batches(dist_paths):
    from torch.utils.data import DataLoader
    ds = _make_ds(dist_paths, split="val", max_windows=64)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batches = list(dl)
    assert len(batches) == 8
    sig_b, lead_b, frame_b = batches[0]
    assert sig_b.shape == (8, 2500)
    assert lead_b.shape == (8,)
    assert frame_b.shape == (8, 500)
