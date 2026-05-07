"""Smoke test for openecg.synth: build a TemplateBank from a few LUDB sinus
records and visualize a couple of generated AV-block windows per scenario."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import ludb, synth

OUT_DIR = REPO / "out" / "viz_synth_avb"
FS = 250


def plot_one(sig, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 4.0))
    t = np.arange(len(sig)) / FS
    ax.plot(t, sig, color="k", lw=0.6)
    cmap = {"p": "red", "qrs": "blue", "t": "green"}
    y_lo = sig.min() - 0.5
    y_hi = sig.min() - 0.1
    band_h = (y_hi - y_lo) / 3.0
    for cls, ymin in (("p", 0.02), ("qrs", 0.10), ("t", 0.18)):
        for on, off in zip(labels[f"{cls}_on"], labels[f"{cls}_off"]):
            ax.axvspan(on / FS, off / FS, ymin=ymin, ymax=ymin + 0.07,
                       color=cmap[cls], alpha=0.55)
    ax.plot([], [], "s", color="red",   label=f"P  ({len(labels['p_on'])})")
    ax.plot([], [], "s", color="blue",  label=f"QRS ({len(labels['qrs_on'])})")
    ax.plot([], [], "s", color="green", label=f"T  ({len(labels['t_on'])})")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (z-norm)")
    ax.set_xlim(0, len(sig) / FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Build a small bank from a subset of sinus records (faster smoke test).
    meta = ludb.load_metadata()
    sinus_ids = [r["id_int"] for r in meta if r["rhythm"].lower() == "sinus rhythm"]
    sample_ids = sinus_ids[:30]
    print(f"Building TemplateBank from {len(sample_ids)} LUDB sinus records "
          f"(leads ii, v1)...", flush=True)
    bank = synth.TemplateBank.from_ludb(
        record_ids=sample_ids, leads=("ii", "v1"), max_per_lead=200,
    )
    for lead in ("ii", "v1"):
        print(f"  {lead}: P templates={len(bank.p[lead])}  "
              f"QRS-T sinus={len(bank.qrst[lead])}  "
              f"QRS-T paced={len(bank.qrst_paced[lead])}", flush=True)

    rng = np.random.default_rng(0)
    for lead in ("ii", "v1"):
        for scenario in ("mobitz1", "mobitz2", "complete", "paced"):
            for k in range(3):
                sig, labels = synth.generate_avb_window(
                    bank, lead, scenario, rng, fs=FS, duration_s=10.0,
                )
                title = (f"synthetic {scenario}  lead={lead}  sample {k}  "
                         f"(P={len(labels['p_on'])} QRS={len(labels['qrs_on'])} "
                         f"T={len(labels['t_on'])})")
                out_path = OUT_DIR / f"{scenario}_{lead}_{k}.png"
                plot_one(sig, labels, title, out_path)
                print(f"  {out_path.name}  P={len(labels['p_on'])} "
                      f"QRS={len(labels['qrs_on'])} T={len(labels['t_on'])}",
                      flush=True)
    print(f"\nWrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
