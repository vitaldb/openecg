# scripts/tokenize_ludb.py
"""Run full pipeline on every LUDB record x lead -> ludb_tokens.npz.

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/tokenize_ludb.py
"""

import json
import time
from pathlib import Path

import numpy as np

from ecgcode import codec, delineate, labeler, ludb, pacer, vocab

OUT_PATH = Path("data/ludb_tokens.npz")


def tokenize_one(sig: np.ndarray, fs: int = 500) -> np.ndarray:
    dr = delineate.run(sig, fs=fs)
    spikes = pacer.detect_spikes(sig, fs=fs)
    events = labeler.label(dr, spikes.tolist(), n_samples=len(sig), fs=fs)
    return codec.encode(events)


def main():
    record_ids = ludb.all_record_ids()
    print(f"Tokenizing {len(record_ids)} records x {len(ludb.LEADS_12)} leads "
          f"= {len(record_ids) * len(ludb.LEADS_12)} sequences...")

    arrays: dict[str, np.ndarray] = {}
    t0 = time.time()
    for n, rid in enumerate(record_ids, 1):
        record = ludb.load_record(rid)
        for lead in ludb.LEADS_12:
            sig = record[lead]
            packed = tokenize_one(sig, fs=500)
            arrays[f"{rid:04d}_{lead}"] = packed
        if n % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n}/{len(record_ids)}] {elapsed:.1f}s elapsed")

    meta = {
        "vocab_version": vocab.VOCAB_VERSION,
        "ms_unit": codec.MS_PER_UNIT,
        "fs": 500,
        "n_records": len(record_ids),
        "leads": list(ludb.LEADS_12),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, meta=json.dumps(meta), **arrays)

    elapsed = time.time() - t0
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Saved {OUT_PATH} ({size_kb:.1f} KB) in {elapsed:.1f}s")
    print(f"Total sequences: {len(arrays)}, mean events/seq: "
          f"{np.mean([len(a) for a in arrays.values()]):.1f}")


if __name__ == "__main__":
    main()
