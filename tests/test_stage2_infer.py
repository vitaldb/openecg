import numpy as np

from openecg.stage2.infer import post_process_frames


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


def test_apply_reg_to_boundaries_shifts_each_sample():
    from openecg.stage2.infer import apply_reg_to_boundaries
    import numpy as np
    boundaries = {
        "p_on":   [25],
        "p_off":  [54],
        "qrs_on": [60],
        "qrs_off":[80],
        "t_on":   [],
        "t_off":  [],
    }
    reg = np.zeros((500, 6), dtype=np.float32)
    reg[5, 0] = 2.0
    reg[10, 1] = -3.0
    reg[12, 2] = 1.0
    reg[16, 3] = 0.0
    refined = apply_reg_to_boundaries(boundaries, reg, samples_per_frame=5,
                                       max_window=2500)
    assert refined["p_on"]   == [27]
    assert refined["p_off"]  == [51]
    assert refined["qrs_on"] == [61]
    assert refined["qrs_off"]== [80]


def test_predict_frames_with_reg_shapes():
    from openecg.stage2.model import FrameClassifierViTReg
    from openecg.stage2.infer import predict_frames_with_reg
    import numpy as np
    model = FrameClassifierViTReg(
        patch_size=5, d_model=32, n_heads=2, n_layers=2, ff=64,
        use_lead_emb=False, pos_type="learnable",
    )
    sig = np.zeros(2500, dtype=np.float32)
    frames, reg = predict_frames_with_reg(model, sig, lead_id=0, device="cpu")
    assert frames.shape == (500,)
    assert frames.dtype.name == "uint8"
    assert reg.shape == (500, 6)
