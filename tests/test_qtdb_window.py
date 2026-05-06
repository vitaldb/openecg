"""Regression tests for qtdb.annotated_window and the cluster helper.

The original bug: q1c on records like sel114, sel116, sel213, sele0112 has two
disjoint annotation clusters separated by minutes of unannotated signal. The
midpoint of (min, max) annotation samples landed in the empty gap, producing
windows with zero annotations and downstream silent model output.
"""

from openecg import qtdb
from openecg.stage2.multi_dataset import _find_annotation_clusters


def test_annotated_window_picks_dense_cluster_not_gap_midpoint():
    """Disjoint-cluster case mirroring sel114: two short clusters far apart.

    With the old (min, max)/2 logic, the chosen window would land in the
    empty gap at sample ~100000 with zero annotations. The density-based
    rewrite must choose a window covering one of the clusters.
    """
    early_cluster = list(range(50_000, 50_300, 30))   # ~10 beats
    late_cluster = list(range(150_000, 150_300, 30))  # ~10 beats
    ann = {
        "p_on": early_cluster + late_cluster,
        "p_off": [], "qrs_on": [], "qrs_off": [],
        "t_on": [], "t_off": [],
    }
    win = qtdb.annotated_window(ann, window_samples=2500)
    assert win is not None
    start, end = win
    assert end - start == 2500
    n_in_window = sum(1 for s in (early_cluster + late_cluster)
                      if start <= s < end)
    assert n_in_window > 0, (
        f"Window [{start}, {end}] contains no annotations — density-based "
        f"selection regressed to gap-midpoint behavior."
    )
    # And specifically: the midpoint of (min, max) is at ~100k, which would
    # have been the buggy choice. Make sure we picked a real cluster.
    midpoint_buggy = (50_000 + 150_300) // 2
    assert not (midpoint_buggy - 1250 <= start <= midpoint_buggy + 1250), (
        "Window starts near the (min, max) midpoint — still in the gap."
    )


def test_annotated_window_returns_none_for_empty():
    assert qtdb.annotated_window({"p_on": [], "qrs_on": []}) is None


def test_annotated_window_handles_single_annotation():
    win = qtdb.annotated_window({"qrs_on": [10_000]}, window_samples=2500)
    assert win is not None
    start, end = win
    assert start <= 10_000 < end


def test_annotated_window_dense_record_unchanged_behavior():
    """On a dense record (sel100-style), density-based picks the same region
    that midpoint would. Just confirm we still return a window covering it."""
    beats = list(range(50_000, 80_000, 250))  # ~120 beats, 1Hz, contiguous
    win = qtdb.annotated_window({"qrs_on": beats}, window_samples=2500)
    assert win is not None
    start, end = win
    n_in = sum(1 for s in beats if start <= s < end)
    assert n_in >= 8, f"only {n_in} beats covered by densest 2500-sample window"


def test_find_annotation_clusters_splits_on_large_gap():
    samples = [100, 200, 300, 50_000, 50_100]
    clusters = _find_annotation_clusters(samples, gap=10_000)
    assert clusters == [(100, 300), (50_000, 50_100)]


def test_find_annotation_clusters_single_cluster():
    samples = [100, 200, 300, 400]
    clusters = _find_annotation_clusters(samples, gap=10_000)
    assert clusters == [(100, 400)]


def test_find_annotation_clusters_empty_and_singleton():
    assert _find_annotation_clusters([], gap=10_000) == []
    assert _find_annotation_clusters([42], gap=10_000) == [(42, 42)]
