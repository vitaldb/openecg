"""ECGCode token format codec — uint16 pack/unpack, frame expansion, ASCII render.

Spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md §5
"""

import numpy as np

from ecgcode import vocab

MS_PER_UNIT = 4
MAX_LENGTH_MS = 255 * MS_PER_UNIT  # 1020 ms


def _split_long(sym: int, length_ms: int) -> list[tuple[int, int]]:
    chunks = []
    while length_ms > MAX_LENGTH_MS:
        chunks.append((sym, MAX_LENGTH_MS))
        length_ms -= MAX_LENGTH_MS
    if length_ms > 0:
        chunks.append((sym, length_ms))
    return chunks


def encode(events: list[tuple[int, int]]) -> np.ndarray:
    """Pack (symbol_id, length_ms) events to uint16 array.

    Length is snapped to nearest 4ms (min 4ms). Long events split at 1020ms.
    Raises ValueError if any event has length 0 or invalid symbol_id.
    """
    out = []
    for sym, ms in events:
        if ms <= 0:
            raise ValueError(f"Event length must be positive, got {ms}")
        if not (0 <= sym <= 255):
            raise ValueError(f"Invalid symbol_id {sym}")
        units = max(1, round(ms / MS_PER_UNIT))
        snapped_ms = units * MS_PER_UNIT
        for chunk_sym, chunk_ms in _split_long(sym, snapped_ms):
            chunk_units = chunk_ms // MS_PER_UNIT
            packed = (chunk_sym << 8) | chunk_units
            out.append(packed)
    return np.array(out, dtype=np.uint16)


def decode(packed: np.ndarray) -> list[tuple[int, int]]:
    """Unpack uint16 array to (symbol_id, length_ms) events.

    Consecutive runs of the same symbol are merged so that long events split
    by `encode` (>1020ms) round-trip losslessly.
    """
    types = (packed >> 8).astype(np.uint8)
    units = (packed & 0xFF).astype(np.uint8)
    out: list[tuple[int, int]] = []
    for t, u in zip(types, units):
        sym = int(t)
        ms = int(u) * MS_PER_UNIT
        if out and out[-1][0] == sym:
            out[-1] = (sym, out[-1][1] + ms)
        else:
            out.append((sym, ms))
    return out


def to_frames(
    events: list[tuple[int, int]],
    frame_ms: int = 20,
    total_ms: int | None = None,
) -> np.ndarray:
    """Expand RLE events to per-frame symbol array.

    Rule: each frame gets the symbol with maximum overlap, with `*` (pacer)
    as priority override (any frame containing a spike -> pacer label).
    """
    if total_ms is None:
        total_ms = sum(ms for _, ms in events)
    n_frames = round(total_ms / frame_ms)
    out = np.zeros(n_frames, dtype=np.uint8)

    # Build (start_ms, end_ms, sym) intervals
    intervals = []
    cum = 0
    for sym, ms in events:
        intervals.append((cum, cum + ms, sym))
        cum += ms

    for f in range(n_frames):
        f_start = f * frame_ms
        f_end = f_start + frame_ms
        overlap = {}
        spike_present = False
        for s_start, s_end, sym in intervals:
            if s_end <= f_start:
                continue
            if s_start >= f_end:
                break
            ov = min(s_end, f_end) - max(s_start, f_start)
            if ov > 0:
                if sym == vocab.ID_PACER:
                    spike_present = True
                else:
                    overlap[sym] = overlap.get(sym, 0) + ov
        if spike_present:
            out[f] = vocab.ID_PACER
        elif overlap:
            out[f] = max(overlap, key=overlap.get)
        else:
            out[f] = vocab.ID_ISO
    return out


def render_compact(events: list[tuple[int, int]]) -> str:
    """One char per event."""
    return "".join(vocab.ID_TO_CHAR[sym] for sym, _ in events)


def render_timed(events: list[tuple[int, int]], ms_per_char: int = 20) -> str:
    """Char count proportional to duration. Minimum 1 char per event."""
    chars = []
    for sym, ms in events:
        n = max(1, round(ms / ms_per_char))
        chars.append(vocab.ID_TO_CHAR[sym] * n)
    return "".join(chars)


def render_json(events: list[tuple[int, int]]) -> list[dict]:
    """Verbose JSON view: [{'sym': name, 'ms': length}, ...]."""
    return [{"sym": vocab.ID_TO_NAME[sym], "ms": ms} for sym, ms in events]
