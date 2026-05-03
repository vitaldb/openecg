import numpy as np
import pytest

from ecgcode import codec, vocab


def test_encode_decode_roundtrip():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 92), (vocab.ID_ISO, 200)]
    packed = codec.encode(events)
    assert packed.dtype == np.uint16
    assert codec.decode(packed) == events


def test_encode_returns_uint16():
    packed = codec.encode([(vocab.ID_P, 80)])
    assert packed.dtype == np.uint16
    assert len(packed) == 1


def test_length_4ms_snap():
    events = [(vocab.ID_P, 95)]   # 95ms -> snap to nearest 4ms = 96
    packed = codec.encode(events)
    decoded = codec.decode(packed)
    assert decoded == [(vocab.ID_P, 96)]


def test_length_under_4ms_snaps_to_4ms_minimum():
    events = [(vocab.ID_PACER, 2)]   # 2ms -> min 1 unit = 4ms
    decoded = codec.decode(codec.encode(events))
    assert decoded == [(vocab.ID_PACER, 4)]


def test_length_over_1020_splits():
    events = [(vocab.ID_ISO, 2500)]  # 2500ms -> 1020 + 1020 + 460
    packed = codec.encode(events)
    assert len(packed) == 3
    decoded = codec.decode(packed)
    assert sum(ms for _, ms in decoded) == 2500
    assert all(s == vocab.ID_ISO for s, _ in decoded)


def test_zero_length_event_rejected():
    with pytest.raises(ValueError):
        codec.encode([(vocab.ID_P, 0)])


def test_frame_expansion_basic():
    # 100ms iso + 100ms P at 20ms/frame -> 5 iso frames + 5 P frames
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 100)]
    frames = codec.to_frames(events, frame_ms=20)
    assert len(frames) == 10
    assert (frames[:5] == vocab.ID_ISO).all()
    assert (frames[5:] == vocab.ID_P).all()


def test_frame_expansion_max_overlap():
    # 16ms Q + 24ms R in same 40ms window at 20ms/frame
    # frame 0 (0-20ms): Q=16, R=4 -> Q wins (max overlap)
    # frame 1 (20-40ms): R only -> R
    events = [(vocab.ID_Q, 16), (vocab.ID_R, 24)]
    frames = codec.to_frames(events, frame_ms=20)
    assert len(frames) == 2
    assert frames[0] == vocab.ID_Q
    assert frames[1] == vocab.ID_R


def test_frame_expansion_spike_priority():
    # 20ms iso + 4ms spike + 16ms iso -> 2 frames
    # frame 0 (0-20ms): iso=20 -> iso wins
    # frame 1 (20-40ms): spike=4 + iso=16 -> spike wins (priority override)
    events = [(vocab.ID_ISO, 20), (vocab.ID_PACER, 4), (vocab.ID_ISO, 16)]
    frames = codec.to_frames(events, frame_ms=20)
    assert len(frames) == 2
    assert frames[0] == vocab.ID_ISO
    assert frames[1] == vocab.ID_PACER


def test_frame_expansion_total_ms_override():
    events = [(vocab.ID_ISO, 100)]
    frames = codec.to_frames(events, frame_ms=20, total_ms=200)
    assert len(frames) == 10  # 200/20


def test_frame_expansion_uint8_dtype():
    events = [(vocab.ID_P, 60)]
    frames = codec.to_frames(events, frame_ms=20)
    assert frames.dtype == np.uint8


def test_render_compact():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 92), (vocab.ID_ISO, 44),
              (vocab.ID_Q, 16), (vocab.ID_R, 40), (vocab.ID_S, 24),
              (vocab.ID_ISO, 84), (vocab.ID_T, 164), (vocab.ID_ISO, 208)]
    s = codec.render_compact(events)
    assert s == "_p_qrs_t_"


def test_render_timed():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 100)]
    s = codec.render_timed(events, ms_per_char=20)
    # 100ms iso = 5 underscores, 100ms P = 5 p's
    assert s == "_____ppppp"


def test_render_timed_min_one_char():
    # 8ms event at 20ms/char -> round(0.4)=0, but min 1 char
    events = [(vocab.ID_PACER, 8)]
    s = codec.render_timed(events, ms_per_char=20)
    assert s == "*"


def test_render_json():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 92)]
    j = codec.render_json(events)
    assert j == [{"sym": "iso", "ms": 100}, {"sym": "P", "ms": 92}]


def test_from_frames_basic():
    frames = np.array([vocab.ID_ISO]*5 + [vocab.ID_P]*5, dtype=np.uint8)
    events = codec.from_frames(frames, frame_ms=20)
    assert events == [(vocab.ID_ISO, 100), (vocab.ID_P, 100)]


def test_from_frames_single_frame_event():
    frames = np.array([vocab.ID_ISO, vocab.ID_PACER, vocab.ID_ISO], dtype=np.uint8)
    events = codec.from_frames(frames, frame_ms=20)
    assert events == [(vocab.ID_ISO, 20), (vocab.ID_PACER, 20), (vocab.ID_ISO, 20)]


def test_from_frames_roundtrip_with_to_frames():
    events = [(vocab.ID_ISO, 100), (vocab.ID_P, 100), (vocab.ID_ISO, 80),
              (vocab.ID_R, 60), (vocab.ID_T, 200)]
    frames = codec.to_frames(events, frame_ms=20)
    recovered = codec.from_frames(frames, frame_ms=20)
    assert [s for s, _ in recovered] == [s for s, _ in events]
    assert sum(ms for _, ms in recovered) == sum(ms for _, ms in events)


def test_from_frames_empty():
    frames = np.array([], dtype=np.uint8)
    assert codec.from_frames(frames, frame_ms=20) == []
