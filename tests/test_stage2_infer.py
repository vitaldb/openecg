import numpy as np

from ecgcode.stage2.infer import post_process_frames


def test_post_process_removes_short_segments():
    # 1-frame P at index 5 (=20ms < 40ms)
    frames = np.array([0] * 5 + [1] * 1 + [0] * 10, dtype=np.uint8)
    out = post_process_frames(frames, frame_ms=20, min_duration_ms=40)
    assert (out == 0).all()


def test_post_process_keeps_long_segments():
    # 5-frame P (100ms > 40ms)
    frames = np.array([0] * 5 + [1] * 5 + [0] * 5, dtype=np.uint8)
    out = post_process_frames(frames, frame_ms=20, min_duration_ms=40)
    assert (out[5:10] == 1).all()


def test_post_process_merges_close_segments():
    # P-other-P with 5-frame gap (=100ms < 300ms merge threshold).
    frames = np.array([1] * 5 + [0] * 5 + [1] * 5, dtype=np.uint8)
    out = post_process_frames(frames, frame_ms=20, merge_gap_ms=300)
    assert (out == 1).all()


def test_post_process_does_not_merge_distant_segments():
    # P with 20-frame gap (=400ms > 300ms merge threshold).
    frames = np.array([1] * 5 + [0] * 20 + [1] * 5, dtype=np.uint8)
    out = post_process_frames(frames, frame_ms=20, merge_gap_ms=300, min_duration_ms=40)
    # Middle should still be 0
    assert (out[5:25] == 0).all()
    assert (out[25:30] == 1).all()


def test_post_process_empty():
    assert len(post_process_frames(np.array([], dtype=np.uint8))) == 0


def test_post_process_returns_uint8():
    frames = np.array([0] * 10 + [1] * 10, dtype=np.uint8)
    out = post_process_frames(frames)
    assert out.dtype == np.uint8
    assert out.shape == frames.shape
