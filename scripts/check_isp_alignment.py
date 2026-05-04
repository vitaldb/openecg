"""Check ISP signal/annotation time alignment.

Hypothesis: gt_to_super_frames uses samples_per_frame = n_samples // n_frames
which gives 19 instead of 20 if n_samples is e.g. 9998 (not exact multiple of 20),
causing accumulating time drift between signal time-axis and frame labels.
"""

import numpy as np
from ecgcode import eval as ee
from ecgcode import isp


def main():
    # Load a few ISP test records and check signal length
    print("=== ISP signal lengths (test split) ===", flush=True)
    test_ids = isp.load_split()["test"][:10]
    for rid in test_ids:
        try:
            record = isp.load_record(rid, split="test")
            sig_ii = record["ii"]
            print(f"  rid={rid}: lead_ii length = {len(sig_ii)} samples "
                  f"({len(sig_ii)/1000:.4f}s @ 1000Hz)", flush=True)
        except Exception as ex:
            print(f"  rid={rid}: ERROR {ex}", flush=True)

    # Pick one and trace label generation
    rid = test_ids[0]
    print(f"\n=== Annotation trace for rid={rid} ===", flush=True)
    record = isp.load_record(rid, split="test")
    sig_1000 = record["ii"]
    n_samples = len(sig_1000)
    print(f"  signal length: {n_samples} samples", flush=True)

    ann_super = isp.load_annotations_as_super(rid, split="test")
    print(f"  raw annotations (first 5 of each):", flush=True)
    for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
        v = ann_super[k][:5]
        print(f"    {k:8s}: {v} samples = {[f'{x:.0f}ms' for x in v]} (at 1000Hz)", flush=True)

    print(f"\n=== gt_to_super_frames internals ===", flush=True)
    fs, frame_ms = 1000, 20
    n_frames = round(n_samples * 1000.0 / fs / frame_ms)
    samples_per_frame = n_samples // n_frames if n_frames > 0 else 1
    print(f"  fs={fs}, frame_ms={frame_ms}", flush=True)
    print(f"  n_frames = round({n_samples}*1000/{fs}/{frame_ms}) = {n_frames}", flush=True)
    print(f"  samples_per_frame = {n_samples} // {n_frames} = {samples_per_frame}", flush=True)
    print(f"  EXPECTED samples_per_frame = {fs*frame_ms/1000} = {int(fs*frame_ms/1000)}", flush=True)
    if samples_per_frame != int(fs * frame_ms / 1000):
        drift_ms_at_frame_499 = (samples_per_frame - int(fs*frame_ms/1000)) * 499 / fs * 1000
        print(f"  >>> MISMATCH: drift at frame 499 = {drift_ms_at_frame_499:+.1f}ms <<<",
              flush=True)

    # Build labels and check: where does first QRS GT actually land vs where
    # we'd plot it on 50Hz frame axis?
    labels = ee.gt_to_super_frames(ann_super, n_samples=n_samples, fs=fs, frame_ms=frame_ms)
    qrs_frames = np.where(labels == ee.SUPER_QRS)[0]
    print(f"\n=== First QRS region in frame labels ===", flush=True)
    if len(qrs_frames) > 0:
        first_qrs_start = int(qrs_frames[0])
        first_qrs_end = int(qrs_frames[0])
        for f in qrs_frames:
            if f - first_qrs_end > 1:
                break
            first_qrs_end = int(f)
        plot_start_ms = first_qrs_start * 20  # what my plot uses
        plot_end_ms = (first_qrs_end + 1) * 20
        true_start_ms = first_qrs_start * samples_per_frame  # what the label actually represents
        true_end_ms = (first_qrs_end + 1) * samples_per_frame
        print(f"  first_qrs_frames = [{first_qrs_start}, {first_qrs_end}]", flush=True)
        print(f"  plot would show:  [{plot_start_ms}ms, {plot_end_ms}ms] (assumes 20ms/frame)",
              flush=True)
        print(f"  label represents: [{true_start_ms}ms, {true_end_ms}ms] (using {samples_per_frame}samples/frame)",
              flush=True)
        # Compare to raw annotation
        if ann_super["qrs_on"]:
            print(f"  raw qrs_on[0] = {ann_super['qrs_on'][0]}ms (from CSV, at 1000Hz)", flush=True)


if __name__ == "__main__":
    main()
