import pytest

from ecgcode.stage2.evaluate import (
    average_boundary_f1,
    boundary_metrics_by_key,
    signed_boundary_metrics,
)


def test_signed_boundary_metrics_reports_signed_error():
    result = signed_boundary_metrics([105, 198], [100, 200], tolerance_ms=20, fs=1000)
    assert result["f1"] == 1.0
    assert result["sens"] == 1.0
    assert result["ppv"] == 1.0
    assert result["mean_signed_ms"] == pytest.approx(1.5)
    assert result["n_hits"] == 2


def test_boundary_metrics_by_key_and_average():
    metrics = boundary_metrics_by_key(
        {"qrs_on": [100], "qrs_off": [210]},
        {"qrs_on": [100], "qrs_off": [200]},
        tolerances_ms={"qrs_on": 5, "qrs_off": 5},
        fs=1000,
    )
    assert metrics["qrs_on"]["f1"] == 1.0
    assert metrics["qrs_off"]["f1"] == 0.0
    assert average_boundary_f1(metrics) == pytest.approx(0.5)
