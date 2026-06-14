"""Headline metrics (LUDB boundary-F1 + MIT-BIH DS2 beat-F1) for a UNIFIED
8-class model, by re-exposing its merged head as frame(4)/beat(6) and reusing
the real eval code. 2-head models pass through unchanged for comparison.

    python -m scripts.eval_unified_headline \
        pod_stage/pod_pull/H_l1.0_s1.pt openecg/models/codec_v5.pt \
        --ludb data/cache/v57_500hz/ludb_val.npz --root data/rare_arrhythmia
"""
import argparse
import numpy as np
import torch

from scripts.eval_boundary_sweep import eval_one
from scripts.eval_multihead_real_500 import eval_beat
from openecg.stage2.model_variants import FrameClassifierTransformerSampleResConvTokMH1Ch

_POP = ("arch", "aux_target", "use_aux", "n_input_channels", "use_reg", "n_reg",
        "sample_res_frame", "tokenizer", "use_lead_emb", "beat_n_classes",
        "rhythm_n_classes", "use_logvar")


class UniWrap(torch.nn.Module):
    """Wrap a unified 8-class model so forward() returns (frame4, beat6, rhythm):
    frame QRS logit = logsumexp(beat-type logits); beat 'none' = logsumexp(wave
    logits). Marginals are exact, so argmax matches the unified argmax."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, lead):
        out = self.m(x, lead)
        z = out[0]                                              # (B,T,8)
        qrs = torch.logsumexp(z[..., 3:8], dim=-1, keepdim=True)
        none = torch.logsumexp(z[..., 0:3], dim=-1, keepdim=True)
        frame4 = torch.cat([z[..., 0:2], qrs, z[..., 2:3]], dim=-1)   # other,P,QRS,T
        beat6 = torch.cat([none, z[..., 3:8]], dim=-1)               # none,sinus,vpc,paced,fusion,unknown
        return frame4, beat6, out[2]


def load(ck, device):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = dict(blob["model_config"])
    for k in _POP:
        cfg.pop(k, None)
    m = FrameClassifierTransformerSampleResConvTokMH1Ch(
        use_lead_emb=False, beat_n_classes=6, rhythm_n_classes=6, **cfg)
    st = {(k[2:] if k.startswith("m.") else k): v for k, v in blob["model_state"].items()}
    m.load_state_dict(st, strict=True)
    m = m.to(device).eval()
    is_uni = m.head_sample.out_features == 8
    return (UniWrap(m).to(device).eval() if is_uni else m), is_uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--ludb", default="data/cache/v57_500hz/ludb_val.npz")
    ap.add_argument("--root", default="data/rare_arrhythmia")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = np.load(args.ludb)
    sigs, lbls = b["signals"], b["labels"]

    print(f"{'ckpt':26s} {'type':8s} {'frame_bF1':>10s} {'med_ms':>7s} "
          f"{'vpc_F1':>7s} {'sinus_F1':>9s} {'fusion':>7s}")
    for ck in args.ckpts:
        model, is_uni = load(ck, device)
        fr = eval_one(model, sigs, lbls, device)
        bt = eval_beat(model, args.root, device)
        def f1(k):
            return bt[k]["f1"] if k in bt and bt[k].get("support") else float("nan")
        name = ck.split("/")[-1]
        print(f"{name:26s} {'unified' if is_uni else '2-head':8s} "
              f"{fr['boundary_f1']:10.4f} {fr['median_ms']:7.2f} "
              f"{f1('vpc'):7.3f} {f1('sinus'):9.3f} {f1('fusion'):7.3f}", flush=True)


if __name__ == "__main__":
    main()
