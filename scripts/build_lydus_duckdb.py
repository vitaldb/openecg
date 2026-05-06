"""Build lydus_ecg.duckdb from the lydus_ecg dataset.

Combines metadata (ecg_matched.csv.xz) + measurements (conclusions.csv.xz)
into a single `records` table keyed by `rid`. Signals stay in the existing
`ecg_matched.npz`; the `npz_idx` column lets you fancy-index the npz array
after a SQL filter:

    con = duckdb.connect("lydus_ecg.duckdb", read_only=True)
    z = np.load("ecg_matched.npz", mmap_mode="r")
    idxs = con.execute(
        "SELECT npz_idx FROM records WHERE rhythm = 'A.fib' AND age >= 60"
    ).fetchnumpy()["npz_idx"]
    signals = z["vals"][idxs]   # (N, 40000) int16, lead-major reshape to (N, 8, 5000)

Conclusions whose rid does not appear in the matched npz set are dropped
(per design: 167K of 679K conclusions retained).
"""

import argparse
import lzma
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_LYDUS_DIR = Path(r"G:\Shared drives\Datasets\ECG\lydus_ecg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lydus-dir", type=Path, default=DEFAULT_LYDUS_DIR)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .duckdb path (default: <lydus-dir>/lydus_ecg.duckdb)",
    )
    args = parser.parse_args()
    lydus = args.lydus_dir
    out_path = args.out or (lydus / "lydus_ecg.duckdb")

    print(f"[1/5] Parsing npz keys at {lydus / 'ecg_matched.npz'} ...", flush=True)
    z = np.load(lydus / "ecg_matched.npz", mmap_mode="r")
    keys = z["keys"]
    parts = np.array([str(k).split("_") for k in keys])
    npz_df = pd.DataFrame({
        "rid": parts[:, 0].astype(np.int32),
        "hid": parts[:, 1].astype(np.int32),
        "dt_str": parts[:, 2],
        "npz_idx": np.arange(len(keys), dtype=np.int32),
    })
    print(f"      {len(npz_df)} npz records, rid range "
          f"[{npz_df['rid'].min()}, {npz_df['rid'].max()}]", flush=True)

    print("[2/5] Loading ecg_matched.csv.xz ...", flush=True)
    with lzma.open(lydus / "ecg_matched.csv.xz", "rt", encoding="utf-8-sig") as f:
        ecg_df = pd.read_csv(f)
    ecg_df.columns = [c.replace(" ", "_") for c in ecg_df.columns]
    ecg_df["dt"] = pd.to_datetime(ecg_df["dt"])
    ecg_df["dt_str"] = ecg_df["dt"].dt.strftime("%Y%m%d")
    ecg_df["hid"] = ecg_df["hid"].astype(np.int32)
    print(f"      {len(ecg_df)} rows", flush=True)

    print("[3/5] Loading conclusions.csv.xz ...", flush=True)
    with lzma.open(lydus / "conclusions.csv.xz", "rt", encoding="utf-8-sig") as f:
        con_df = pd.read_csv(f)
    con_df["rid"] = con_df["rid"].astype(np.int32)
    print(f"      {len(con_df)} rows ({con_df['rid'].nunique()} unique rids)",
          flush=True)

    print("[4/5] Joining ...", flush=True)
    merged = npz_df.merge(
        ecg_df, on=["hid", "dt_str"], how="inner",
        validate="one_to_one",
    )
    print(f"      after join with ecg_matched.csv: {len(merged)}", flush=True)
    merged = merged.merge(con_df, on="rid", how="inner", validate="one_to_one")
    print(f"      after join with conclusions.csv:  {len(merged)}", flush=True)
    assert len(merged) == len(npz_df), (
        f"Lost {len(npz_df) - len(merged)} records during join — investigate"
    )

    final_cols = [
        "rid", "hid", "dt", "npz_idx",
        "rhythm", "premature_beat", "bbb", "avb", "pacing",
        "conclusion", "file",
        "dx", "age", "sex",
        "vrate", "arate", "pri", "qrsd", "qti", "qtc",
        "paxis", "raxis", "taxis",
    ]
    final = merged[final_cols].copy()
    # Cast tiny numerics to compact dtypes; keep as Pandas first, DuckDB will
    # respect the column types we declare in CREATE TABLE.
    for c in ("rid", "hid", "npz_idx"):
        final[c] = final[c].astype(np.int32)

    print(f"[5/5] Writing {out_path} ...", flush=True)
    if out_path.exists():
        out_path.unlink()
    con = duckdb.connect(str(out_path))
    con.execute(
        """
        CREATE TABLE records (
            rid             INTEGER PRIMARY KEY,
            hid             INTEGER NOT NULL,
            dt              TIMESTAMP,
            npz_idx         INTEGER NOT NULL,
            rhythm          VARCHAR,
            premature_beat  VARCHAR,
            bbb             VARCHAR,
            avb             VARCHAR,
            pacing          VARCHAR,
            conclusion      TEXT,
            file            VARCHAR,
            dx              TEXT,
            age             SMALLINT,
            sex             VARCHAR,
            vrate           INTEGER,
            arate           INTEGER,
            pri             INTEGER,
            qrsd            INTEGER,
            qti             INTEGER,
            qtc             INTEGER,
            paxis           SMALLINT,
            raxis           SMALLINT,
            taxis           SMALLINT
        )
        """
    )
    con.register("final_df", final)
    con.execute("INSERT INTO records SELECT * FROM final_df")
    con.execute("CREATE INDEX idx_records_hid    ON records(hid)")
    con.execute("CREATE INDEX idx_records_dt     ON records(dt)")
    con.execute("CREATE INDEX idx_records_rhythm ON records(rhythm)")
    con.execute("CHECKPOINT;")
    n = con.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    con.close()
    sz_mb = out_path.stat().st_size / 1e6
    print(f"      wrote {n} rows, {sz_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
