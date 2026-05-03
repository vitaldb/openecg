import numpy as np
import pytest

from ecgcode import eval as ee
from ecgcode import vocab


def test_supercategory_mapping():
    frames = np.array([vocab.ID_ISO, vocab.ID_P, vocab.ID_Q, vocab.ID_R,
                       vocab.ID_S, vocab.ID_W, vocab.ID_T, vocab.ID_PACER,
                       vocab.ID_UNK], dtype=np.uint8)
    super_frames = ee.to_supercategory(frames)
    assert super_frames.tolist() == [
        ee.SUPER_OTHER, ee.SUPER_P, ee.SUPER_QRS, ee.SUPER_QRS,
        ee.SUPER_QRS, ee.SUPER_QRS, ee.SUPER_T, ee.SUPER_OTHER, ee.SUPER_OTHER,
    ]


def test_frame_f1_perfect_match():
    pred = np.array([ee.SUPER_P] * 5 + [ee.SUPER_QRS] * 3 + [ee.SUPER_T] * 5)
    true = pred.copy()
    metrics = ee.frame_f1(pred, true)
    for super_class in (ee.SUPER_P, ee.SUPER_QRS, ee.SUPER_T):
        assert metrics[super_class]["f1"] == 1.0
        assert metrics[super_class]["precision"] == 1.0
        assert metrics[super_class]["recall"] == 1.0


def test_frame_f1_total_disagreement():
    pred = np.array([ee.SUPER_P] * 10)
    true = np.array([ee.SUPER_QRS] * 10)
    metrics = ee.frame_f1(pred, true)
    assert metrics[ee.SUPER_P]["f1"] == 0.0
    assert metrics[ee.SUPER_QRS]["f1"] == 0.0


def test_boundary_error_perfect_alignment():
    true_idx = [100, 200, 300]
    pred_idx = [100, 200, 300]
    result = ee.boundary_error(pred_idx, true_idx, tolerance_ms=50, fs=500)
    assert result["sensitivity"] == 1.0
    assert result["ppv"] == 1.0
    assert result["median_error_ms"] == 0.0
    assert result["n_hits"] == 3


def test_boundary_error_partial_match():
    true_idx = [100, 200, 300]
    pred_idx = [105, 199, 600]
    result = ee.boundary_error(pred_idx, true_idx, tolerance_ms=50, fs=500)
    assert result["n_hits"] == 2
    assert result["sensitivity"] == pytest.approx(2/3)
    assert result["ppv"] == pytest.approx(2/3)


def test_boundary_error_empty_inputs():
    result = ee.boundary_error([], [], tolerance_ms=50, fs=500)
    assert result["n_hits"] == 0
    assert result["sensitivity"] == 0.0
    assert result["ppv"] == 0.0
