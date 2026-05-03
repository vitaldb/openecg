# tests/test_integration.py
"""End-to-end integration test on LUDB record 1.

Skipped if ECGCODE_LUDB_ZIP env var not set or extraction fails.
"""

import os

import numpy as np
import pytest

LUDB_AVAILABLE = bool(os.environ.get("ECGCODE_LUDB_ZIP"))

pytestmark = pytest.mark.skipif(
    not LUDB_AVAILABLE,
    reason="ECGCODE_LUDB_ZIP env var not set; integration test requires LUDB",
)


def test_record_1_lead_ii_end_to_end():
    from ecgcode import codec, delineate, labeler, ludb, pacer

    record = ludb.load_record(1)
    sig = record["ii"]
    assert len(sig) == 5000

    dr = delineate.run(sig, fs=500)
    assert dr.n_beats > 0   # NK should detect beats in record 1 (sinus brady)

    spikes = pacer.detect_spikes(sig, fs=500)

    events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=500)
    total_ms = sum(ms for _, ms in events)
    assert 9900 <= total_ms <= 10100, f"expected ~10s, got {total_ms}ms"

    packed = codec.encode(events)
    assert packed.dtype == np.uint16
    assert codec.decode(packed) == events

    art = codec.render_timed(events, ms_per_char=20)
    assert 480 <= len(art) <= 520, f"expected ~500 chars, got {len(art)}"
    print(f"\nRecord 1 lead II ASCII art ({len(art)} chars):")
    print(art)


def test_all_12_leads_record_1_no_crash():
    from ecgcode import codec, delineate, labeler, ludb, pacer

    record = ludb.load_record(1)
    for lead, sig in record.items():
        dr = delineate.run(sig, fs=500)
        spikes = pacer.detect_spikes(sig, fs=500)
        events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=500)
        assert len(events) > 0, f"empty events for lead {lead}"
        # Round-trip must be lossless
        packed = codec.encode(events)
        assert codec.decode(packed) == events


def test_pacer_record_8_detects_spikes():
    """Record 8 is a pacemaker patient (per ludb.csv 'Cardiac pacing' column).
    Pacer detector should find at least one spike on lead V1 or II."""
    from ecgcode import ludb, pacer

    record = ludb.load_record(8)
    spike_counts = {lead: len(pacer.detect_spikes(sig, fs=500))
                    for lead, sig in record.items()}
    total_spikes = sum(spike_counts.values())
    assert total_spikes > 0, f"no spikes detected on pacer record 8: {spike_counts}"
