# openecg/stage2/window_labels.py
"""Window-level multi-task supervision for the v17 hierarchical model.

Two binary tasks form the divide-and-conquer pivots of the clinical
beat/rhythm interpretation tree (R-R regular? narrow QRS?). Both can be
derived from the frame-level QRS labels available on every frame-labelled
dataset (LUDB / ISP / QTDB / synth) AND read directly from the rich
metadata of the lydus dataset, so the same loss applies across domains.

Layout: window labels are packed as `int64[N_WINDOW_TASKS]` with a parallel
`bool[N_WINDOW_TASKS]` mask. Order matches `WINDOW_TASK_NAMES`.
"""

from __future__ import annotations

import numpy as np

from openecg import eval as _ee

WINDOW_TASK_NAMES: tuple[str, ...] = ("rr_regular", "qrs_wide")
N_WINDOW_TASKS = len(WINDOW_TASK_NAMES)

# QRS ≥ 120 ms is the clinical wide-QRS threshold (BBB / aberrancy boundary).
QRS_WIDE_MS_DEFAULT = 120

# RR-interval SD threshold for "regular". Sinus arrhythmia gives SD ≈ 30-80 ms;
# AF / A.flutter typically ≥ 150 ms. 100 ms is a reasonable cut for a binary
# classifier; extreme tachy / brady stays regular as long as SD is small.
RR_IRREG_SD_MS_DEFAULT = 100

# Need at least this many QRS in the window to estimate RR regularity.
RR_MIN_QRS = 3


def derive_from_frame_labels(
    frame_labels,
    *,
    fs: int = 250,
    frame_ms: int = 20,
    qrs_wide_ms: float = QRS_WIDE_MS_DEFAULT,
    rr_irreg_sd_ms: float = RR_IRREG_SD_MS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive (rr_regular, qrs_wide) labels from a per-frame supercategory
    array. IGNORE_INDEX frames are treated as non-QRS (boundary regions
    are excluded from the run-length extraction).

    Returns:
        labels: int64[2] — (rr_regular, qrs_wide), 0 or 1.
        mask:   bool[2]  — True where the corresponding label is reliable.

    Reliability rules:
      * qrs_wide is reliable when at least 1 QRS run is detected.
      * rr_regular is reliable when at least 3 QRS runs are detected
        (need 2 RR intervals for a meaningful SD).
    """
    arr = np.asarray(frame_labels)
    is_qrs = np.isin(arr, _ee.ALL_SUPER_QRS_CLASSES).astype(np.int8)
    # Run-length boundaries via padded-diff trick.
    edges = np.diff(np.concatenate([[0], is_qrs, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    n_qrs = len(starts)
    labels = np.zeros(N_WINDOW_TASKS, dtype=np.int64)
    mask = np.zeros(N_WINDOW_TASKS, dtype=bool)

    if n_qrs >= 1:
        widths_frames = ends - starts
        median_width_ms = float(np.median(widths_frames)) * frame_ms
        labels[1] = 1 if median_width_ms >= qrs_wide_ms else 0
        mask[1] = True

    if n_qrs >= RR_MIN_QRS:
        centers = (starts + ends) / 2.0
        rr_intervals_ms = np.diff(centers) * frame_ms
        rr_sd_ms = float(np.std(rr_intervals_ms))
        labels[0] = 1 if rr_sd_ms < rr_irreg_sd_ms else 0
        mask[0] = True

    return labels, mask


def from_components(
    *,
    rr_regular: int | None = None,
    qrs_wide: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack scalar component labels (with `None` denoting unavailable)
    into the canonical (labels, mask) pair.
    """
    labels = np.zeros(N_WINDOW_TASKS, dtype=np.int64)
    mask = np.zeros(N_WINDOW_TASKS, dtype=bool)
    if rr_regular is not None:
        labels[0] = int(rr_regular)
        mask[0] = True
    if qrs_wide is not None:
        labels[1] = int(qrs_wide)
        mask[1] = True
    return labels, mask


def for_synth_scenario(
    scenario: str,
    is_ventricular_escape: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic window labels for a synthetic AVB scenario.

    * mobitz1 (Wenckebach) — RR is *not* truly regular (progressive PR
      causes systematic lengthening), so rr_regular is masked out. QRS
      narrow.
    * mobitz2 — fixed PR with periodic dropped beats; ventricular RR is
      regular when blocked beats are excluded, but the dropped beat
      creates a long pause. Masking rr_regular keeps the supervision
      conservative.
    * complete — independent atrial/ventricular schedules; ventricular
      escape RR is regular. Width depends on escape origin.
    * paced — paced ventricle is wide-QRS by definition; RR is regular.

    The QRS-wide bit is set deterministically from the scenario except
    for the rare case where complete uses a junctional (narrow) escape
    instead of a wide ventricular escape.
    """
    if scenario in ("mobitz1", "mobitz2"):
        return from_components(qrs_wide=0)         # rr masked
    if scenario == "complete":
        wide = 1 if is_ventricular_escape else 0
        return from_components(rr_regular=1, qrs_wide=wide)
    if scenario == "paced":
        return from_components(rr_regular=1, qrs_wide=1)
    raise ValueError(f"unknown scenario: {scenario!r}")
