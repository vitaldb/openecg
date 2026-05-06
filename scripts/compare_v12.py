"""Aggregate all v12_*.json result files into one comparison table.

Reads out/train_v9_q1c_pu_merge_*.json and out/train_v12_*_*.json,
emits Markdown table to stdout and also writes
out/v12_comparison_<ts>.md + out/v12_comparison_<ts>.json.

Spec: docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md sec 6.

Plan-bug fix: the original `_row(d)` returned the first dict-valued
entry, which silently picked v9_q1c_pu_merge_ref (the reference block
included in every result file) instead of the actual experiment row.
This version looks up by exact experiment name.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "out"
ROW_KEYS = ("ludb_edge_filtered", "isp_test", "qtdb_pu0_random")


def _latest(prefix: str) -> dict | None:
    files = sorted(glob.glob(str(OUT_DIR / f"train_{prefix}_*.json")))
    if not files:
        return None
    return json.loads(Path(files[-1]).read_text())


def _row(d: dict | None, experiment_key: str,
         candidates: list[str] | None = None) -> dict | None:
    """Look up the row dict in d by exact experiment_key, with optional fallback.

    Each train_v12_*_<ts>.json has shape {"v9_q1c_pu_merge_ref": {...},
    "<experiment_key>": {...}}. The bare-iteration form would silently return
    the v9 reference row. v12_reg.json keys its winning entry under
    "v12_reg_best" rather than "v12_reg" so we accept a fallback list.
    """
    if d is None:
        return None
    keys = [experiment_key] + list(candidates or [])
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict) and any(rk in v for rk in ROW_KEYS):
            return {rk: float(v.get(rk, float("nan"))) for rk in ROW_KEYS}
    return None


def main():
    rows: list[tuple[str, dict | None]] = [
        ("v9_q1c_pu_merge", _row(_latest("v9_q1c_pu_merge"), "v9_q1c_pu_merge")),
        ("v12_soft",        _row(_latest("v12_soft"), "v12_soft")),
        ("v12_reg",         _row(_latest("v12_reg"), "v12_reg",
                                   candidates=["v12_reg_best"])),
        ("v12_hubert_lp",   _row(_latest("v12_hubert_lp"), "v12_hubert_lp")),
        ("v12_hubert_ft",   _row(_latest("v12_hubert_ft"), "v12_hubert_ft")),
        ("v12_stmem_lp",    _row(_latest("v12_stmem_lp"), "v12_stmem_lp")),
        ("v12_stmem_ft",    _row(_latest("v12_stmem_ft"), "v12_stmem_ft")),
        ("v12_best",        _row(_latest("v12_best"), "v12_best")),
    ]

    md_lines = ["# v12 comparison\n",
                f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
                "| run | LUDB val | ISP test | QTDB pu0 |",
                "|---|---|---|---|"]
    for name, row in rows:
        if row is None:
            md_lines.append(f"| {name} | (no run) | (no run) | (no run) |")
            continue
        md_lines.append(
            f"| {name} | "
            f"{row['ludb_edge_filtered']:.3f} | "
            f"{row['isp_test']:.3f} | "
            f"{row['qtdb_pu0_random']:.3f} |"
        )
    md = "\n".join(md_lines) + "\n"
    print(md)

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = OUT_DIR / f"v12_comparison_{ts}.md"
    md_path.write_text(md)
    json_path = OUT_DIR / f"v12_comparison_{ts}.json"
    json_path.write_text(json.dumps(
        {name: row for name, row in rows}, indent=2,
    ))
    print(f"\nSaved {md_path} / {json_path}")


if __name__ == "__main__":
    main()
