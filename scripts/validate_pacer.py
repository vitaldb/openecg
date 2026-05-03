# scripts/validate_pacer.py
"""Validate pacer spike detector: TPR on 10 pacemaker records, FPR on non-pacer.

Usage:
    $env:ECGCODE_LUDB_ZIP = "..."
    uv run python scripts/validate_pacer.py
"""

from collections import defaultdict

from ecgcode import ludb, pacer

FS = 500


def main():
    meta = ludb.load_metadata()
    pacer_ids = sorted(int(m["ID"]) for m in meta if m["pacemaker"])
    non_pacer_ids = sorted(int(m["ID"]) for m in meta if not m["pacemaker"])

    print(f"Pacer records: {len(pacer_ids)} ({pacer_ids})")
    print(f"Non-pacer records: {len(non_pacer_ids)}")
    print()

    print("== Positive set (pacemaker patients) ==")
    pacer_results = []
    for rid in pacer_ids:
        record = ludb.load_record(rid)
        per_lead = {lead: len(pacer.detect_spikes(sig, fs=FS))
                    for lead, sig in record.items()}
        total = sum(per_lead.values())
        n_leads_with_spikes = sum(1 for v in per_lead.values() if v > 0)
        pacer_results.append({
            "id": rid, "total_spikes": total, "leads_with_spikes": n_leads_with_spikes,
            "per_lead": per_lead,
        })
        print(f"  Record {rid:3d}: {total:3d} spikes total, "
              f"{n_leads_with_spikes:2d}/12 leads detected")

    n_records_with_any = sum(1 for r in pacer_results if r["total_spikes"] > 0)
    mean_inter_lead = (sum(r["leads_with_spikes"] for r in pacer_results)
                       / len(pacer_results))
    print(f"\nPacer record detection rate: {n_records_with_any}/{len(pacer_ids)}")
    print(f"Mean inter-lead consistency: {mean_inter_lead:.1f}/12 leads (target >= 6)")

    print("\n== Negative set (non-pacemaker, sample 30 records) ==")
    sampled = non_pacer_ids[:30]
    fp_per_record = []
    suspicious = []
    for rid in sampled:
        record = ludb.load_record(rid)
        total = sum(len(pacer.detect_spikes(sig, fs=FS))
                    for sig in record.values())
        fp_per_record.append(total)
        if total > 5:
            suspicious.append((rid, total))

    fpr_per_10s = sum(fp_per_record) / len(sampled)
    print(f"Total false positives over {len(sampled)} non-pacer records: "
          f"{sum(fp_per_record)}")
    print(f"Mean FPR: {fpr_per_10s:.2f} spikes / 10s record (target < 2)")
    if suspicious:
        print(f"Suspicious records (>5 false spikes): {suspicious}")

    print("\n== Acceptance ==")
    tpr_ok = n_records_with_any >= 8
    inter_ok = mean_inter_lead >= 6
    fpr_ok = fpr_per_10s < 2
    print(f"  TPR (>=8/10 records): {'PASS' if tpr_ok else 'FAIL'}")
    print(f"  Inter-lead (>=6/12):  {'PASS' if inter_ok else 'FAIL'}")
    print(f"  FPR (<2 per 10s):    {'PASS' if fpr_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
