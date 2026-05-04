"""Shared Stage 2 evaluation helpers.

This module keeps boundary-metric math in one place so training and validation
scripts do not grow slightly different Martinez-style implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


MARTINEZ_TOLERANCE_MS = {
    "p_on": 50,
    "p_off": 50,
    "qrs_on": 40,
    "qrs_off": 40,
    "t_on": 50,
    "t_off": 100,
}

BOUNDARY_KEYS = tuple(MARTINEZ_TOLERANCE_MS.keys())


def signed_boundary_metrics(
    pred_indices: Sequence[int],
    true_indices: Sequence[int],
    tolerance_ms: float,
    fs: int = 250,
) -> dict:
    """Greedy nearest-match boundary metrics with signed timing error.

    Returns F1 / sensitivity / PPV plus mean signed error and SD in ms. Each
    true boundary can match at most one prediction within the tolerance.
    """
    tolerance_samples = tolerance_ms * fs / 1000.0
    pred_arr = np.sort(np.asarray(pred_indices, dtype=int))
    true_arr = np.sort(np.asarray(true_indices, dtype=int))

    matched_pred = set()
    errors = []
    for true_idx in true_arr:
        best_idx = -1
        best_abs = float("inf")
        for pred_pos, pred_idx in enumerate(pred_arr):
            if pred_pos in matched_pred:
                continue
            abs_err = abs(int(pred_idx) - int(true_idx))
            if abs_err < best_abs:
                best_abs = abs_err
                best_idx = pred_pos
        if best_idx >= 0 and best_abs <= tolerance_samples:
            matched_pred.add(best_idx)
            errors.append(int(pred_arr[best_idx]) - int(true_idx))

    n_hits = len(errors)
    sens = n_hits / len(true_arr) if len(true_arr) > 0 else 0.0
    ppv = n_hits / len(pred_arr) if len(pred_arr) > 0 else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    if errors:
        err_ms = np.asarray(errors, dtype=float) * 1000.0 / fs
        mean_signed_ms = float(np.mean(err_ms))
        mean_abs_ms = float(np.mean(np.abs(err_ms)))
        sd_ms = float(np.std(err_ms))
        median_abs_ms = float(np.median(np.abs(err_ms)))
        p95_abs_ms = float(np.percentile(np.abs(err_ms), 95))
    else:
        mean_signed_ms = 0.0
        mean_abs_ms = 0.0
        sd_ms = 0.0
        median_abs_ms = 0.0
        p95_abs_ms = 0.0

    return {
        "f1": f1,
        "sens": sens,
        "ppv": ppv,
        "mean_signed_ms": mean_signed_ms,
        "mean_abs_ms": mean_abs_ms,
        "sd_ms": sd_ms,
        "median_abs_ms": median_abs_ms,
        "p95_abs_ms": p95_abs_ms,
        "n_true": int(len(true_arr)),
        "n_pred": int(len(pred_arr)),
        "n_hits": int(n_hits),
    }


def boundary_metrics_by_key(
    pred_by_key: Mapping[str, Sequence[int]],
    true_by_key: Mapping[str, Sequence[int]],
    tolerances_ms: Mapping[str, float] | None = None,
    fs: int = 250,
) -> dict:
    """Compute signed boundary metrics for every configured boundary key."""
    tolerances = tolerances_ms or MARTINEZ_TOLERANCE_MS
    return {
        key: signed_boundary_metrics(
            pred_by_key.get(key, ()),
            true_by_key.get(key, ()),
            tolerance_ms=tol_ms,
            fs=fs,
        )
        for key, tol_ms in tolerances.items()
    }


def average_boundary_f1(metrics_by_key: Mapping[str, Mapping[str, float]]) -> float:
    """Mean F1 across boundary metric dicts."""
    if not metrics_by_key:
        return 0.0
    return float(np.mean([float(m["f1"]) for m in metrics_by_key.values()]))
