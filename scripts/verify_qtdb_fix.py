"""Compare the pre-fix v12_reg checkpoint vs the after-fix retrain on the
records that were previously silent (sel114, sel116, sel213, sel221, sel232,
sel49, sele0112) and on the full QTDB q1c set.

Outputs a markdown table to out/verify_qtdb_fix_<ts>.md and the same data as
JSON.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import qtdb
from openecg.stage2.infer import predict_frames_with_reg
from openecg.stage2.model import FrameClassifierViTReg
from openecg.stage2.train import load_checkpoint
from openecg.stage2.multi_dataset import QTDB_LEAD_TO_LUDB_ID
from scripts.train_v9_q1c_pu_merge import KWARGS

CKPT_DIR = REPO / "data" / "checkpoints"
OUT_DIR = REPO / "out"
WINDOW_SAMPLES = 2500

PRIORITY_RIDS = ("sel114", "sel116", "sel213", "sel221", "sel232", "sel49",
                 "sele0112", "sel102", "sel104")


def _load(name: str):
    ckpt = CKPT_DIR / name
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model = FrameClassifierViTReg(**KWARGS, n_reg=6)
    load_checkpoint(str(ckpt), model)
    model.train(False)
    return model


def _scan(model, device, all_records: bool):
    rids = qtdb.records_with_q1c() if all_records else PRIORITY_RIDS
    out = []
    for rid in rids:
        try:
            record = qtdb.load_record(rid)
            ann = qtdb.load_q1c(rid)
        except Exception:
            continue
        win = qtdb.annotated_window(ann)
        if win is None:
            continue
        win_lo, win_hi = win
        n = len(list(record.values())[0])
        if win_hi > n:
            win_hi = n
            win_lo = max(0, win_hi - WINDOW_SAMPLES)
        lead_name = next(
            (ln for ln in record if ln in QTDB_LEAD_TO_LUDB_ID),
            list(record.keys())[0],
        )
        lid = QTDB_LEAD_TO_LUDB_ID.get(lead_name, 1)
        sig = record[lead_name][win_lo:win_hi].astype(np.float32)
        if len(sig) < WINDOW_SAMPLES:
            sig = np.concatenate(
                [sig, np.zeros(WINDOW_SAMPLES - len(sig), dtype=np.float32)]
            )
        sig = sig[:WINDOW_SAMPLES]
        sig_n = ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)
        frames, _ = predict_frames_with_reg(model, sig_n, lid, device=device)
        out.append({
            "rid": rid, "lead": lead_name, "lead_id": int(lid),
            "win": [int(win_lo), int(win_hi)],
            "nz_frac": float((frames != 0).mean()),
            "n_p": int((frames == 1).sum()),
            "n_qrs": int((frames == 2).sum()),
            "n_t": int((frames == 3).sum()),
        })
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old = _load("stage2_v12_reg.pt").to(device)
    new = _load("stage2_v12_reg_after_qtdb_fix.pt").to(device)

    payload = {"priority": [], "all_q1c": []}
    for tag, all_records in (("priority", False), ("all_q1c", True)):
        old_scan = {r["rid"]: r for r in _scan(old, device, all_records)}
        new_scan = {r["rid"]: r for r in _scan(new, device, all_records)}
        rows = []
        for rid in old_scan:
            o = old_scan[rid]
            n_row = new_scan.get(rid, {"nz_frac": 0.0, "n_p": 0, "n_qrs": 0, "n_t": 0})
            rows.append({"rid": rid, "old": o, "new": n_row})
        payload[tag] = rows

    silent_old = sum(1 for r in payload["all_q1c"] if r["old"]["nz_frac"] == 0.0)
    silent_new = sum(1 for r in payload["all_q1c"] if r["new"]["nz_frac"] == 0.0)
    sparse_old = sum(1 for r in payload["all_q1c"] if r["old"]["nz_frac"] < 0.10)
    sparse_new = sum(1 for r in payload["all_q1c"] if r["new"]["nz_frac"] < 0.10)
    payload["summary"] = {
        "n_records": len(payload["all_q1c"]),
        "silent_old": silent_old, "silent_new": silent_new,
        "sparse_old": sparse_old, "sparse_new": sparse_new,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"verify_qtdb_fix_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md = ["# Silent-record verification (qtdb fix)\n",
          f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
          "## Summary\n",
          f"- N records (q1c set): **{payload['summary']['n_records']}**",
          f"- Silent (0% non-zero frames): **{silent_old} -> {silent_new}**",
          f"- Sparse (<10% non-zero frames): **{sparse_old} -> {sparse_new}**",
          "\n## Priority records (formerly silent or sparse)\n",
          "| record | lead | nz_frac (old -> new) | (P / QRS / T) old | (P / QRS / T) new |",
          "|---|---|---|---|---|"]
    for r in payload["priority"]:
        o, n_row = r["old"], r["new"]
        md.append(
            f"| {r['rid']} | {o.get('lead','?')} | "
            f"{o['nz_frac']:.2%} -> {n_row['nz_frac']:.2%} | "
            f"{o['n_p']} / {o['n_qrs']} / {o['n_t']} | "
            f"{n_row['n_p']} / {n_row['n_qrs']} / {n_row['n_t']} |"
        )
    md_path = OUT_DIR / f"verify_qtdb_fix_{ts}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Saved {json_path} and {md_path}")
    print(f"Silent: {silent_old} -> {silent_new}   Sparse(<10%): {sparse_old} -> {sparse_new}")


if __name__ == "__main__":
    main()
