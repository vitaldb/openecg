"""Self-contained pod trainer for codec_v3 (masking-fix retrain).

Standalone copy of scripts/train_multihead_sweep.py with the openecg package
dependency removed so it runs on a bare kgpu-pytorch pod (torch only):
  * rank_normalize vendored (pure numpy, bit-identical to openecg.dsp).
  * model/model_variants uploaded as FLAT modules (no openecg package import,
    so opendsp/mamba_ssm/etc are never pulled).
  * load_model_from_ckpt(...) replaced by torch.load(...) — load_checkpoint_blob
    is literally torch.load(path, map_location="cpu", weights_only=False).

Usage (one per GPU):
    CUDA_VISIBLE_DEVICES=0 python kgpu_train.py --real-frac 1.0 --synth none \
        --seed 0 --ckpt-out codec_v3_s0.pt
"""
from __future__ import annotations

import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from model_variants import FrameClassifierTransformerSampleResConvTokMH1Ch

E0 = "e0.pt"
V56C = "v56c.pt"
IGN, WS, PATCH = 255, 5000, 10
N_BEAT = N_RHYTHM = 6


def rank_normalize(sig, lo=-1.0, hi=1.0):
    """Per-window rank normalization — vendored from openecg.dsp (pure numpy)."""
    arr = np.asarray(sig, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr.astype(np.float32)
    sorter = np.argsort(arr, kind="stable")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    sorted_arr = arr[sorter]
    obs = np.concatenate([[True], sorted_arr[1:] != sorted_arr[:-1]])
    dense = obs.cumsum()[inv]
    count = np.concatenate([np.nonzero(obs)[0], [n]])
    one_based_avg = 0.5 * (count[dense] + count[dense - 1] + 1)
    denom = max(n - 1, 1)
    zero_based = one_based_avg - 1.0
    return ((zero_based / denom) * (hi - lo) + lo).astype(np.float32)


class _MH(Dataset):
    """Multi-head npz: signals + frame/rhythm/beat (IGN where absent).

    `npz` may be a comma-separated list of paths — they are concatenated
    (e.g. real_ml_v2_train_afib.npz,lydus_rhythm_train.npz). Each source keeps
    its own IGN masks, so layers a source doesn't annotate stay masked.
    """
    def __init__(self, npz, pre_norm=False, augment=False, aug_p=0.5):
        self.pre_norm = pre_norm
        self.augment = augment      # train-only: noise/baseline-wander corruption
        self.aug_p = aug_p
        paths = npz.split(",") if isinstance(npz, str) else list(npz)
        S, F, R, B, L = [], [], [], [], []
        for p in paths:
            b = np.load(p)
            S.append(b["signals"]); F.append(b["frame"]); R.append(b["rhythm"])
            B.append(b["beat"]); L.append(b["lead_ids"])
        self.source_lens = [len(s) for s in S]
        self.sig = np.concatenate(S); self.frame = np.concatenate(F)
        self.rhythm = np.concatenate(R); self.beat = np.concatenate(B)
        self.lead = np.concatenate(L)
        if len(paths) > 1:
            print(f"  _MH concat {len(paths)} sources {self.source_lens} -> {len(self.sig)} windows", flush=True)
    def __len__(self): return len(self.sig)

    @staticmethod
    def _augment(sig, rng):
        """Add baseline wander + EMG noise (labels unchanged). Teaches the model
        that noisy/wandering baseline is NOT a wave -> fixes the over-segmentation
        precision collapse seen on noisy LUDB windows. Re-rank_normalized after."""
        n = sig.size; t = np.arange(n); s = sig.std() + 1e-6; out = sig.astype(np.float64)
        for _ in range(int(rng.integers(1, 3))):              # 1-2 baseline-wander sines
            f = rng.uniform(0.15, 0.8) / 500.0
            out = out + rng.uniform(0.2, 0.9) * s * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.283))
        out = out + rng.normal(0, rng.uniform(0.03, 0.20) * s, n)   # EMG/high-freq
        if rng.random() < 0.3:
            out = out * rng.uniform(0.5, 1.5)                 # amplitude scale
        return out

    def __getitem__(self, i):
        fr = self.frame[i].astype(np.int64).copy()
        fr[fr == 4] = 2                                  # fold paced_QRS->QRS
        fr[self.frame[i] == IGN] = IGN                   # keep IGN
        raw = self.sig[i].astype(np.float32)
        if self.augment and np.random.random() < self.aug_p:
            sig = rank_normalize(self._augment(raw, np.random.default_rng()))
        else:
            sig = raw if self.pre_norm else rank_normalize(raw)   # pre-normalized caches skip the per-window cost
        return (torch.from_numpy(sig),
                torch.tensor(int(self.lead[i])),
                torch.from_numpy(fr),
                torch.from_numpy(self.rhythm[i].astype(np.int64)),
                torch.from_numpy(self.beat[i].astype(np.int64)))


TRAINABLE_WHEN_FROZEN = ("rhythm_sample.", "beat_sample.")  # heads that don't feed frame


def build_model(device, init_ckpt="", freeze_backbone=False,
                d_model=0, n_layers=0, n_heads=0, ff=0, trainable=None, use_logvar=False):
    blob = torch.load(V56C, map_location="cpu", weights_only=False)
    cfg = dict(blob.get("model_config", {})); cfg["patch_size"] = PATCH
    for k in ("arch", "aux_target", "use_aux", "n_input_channels", "use_reg", "n_reg",
              "sample_res_frame", "tokenizer", "use_lead_emb", "beat_n_classes", "rhythm_n_classes",
              "use_logvar"):
        cfg.pop(k, None)
    # ---- depth/width sweep overrides ----------------------------------------
    # Override the v56c arch dims. Width changes (d_model) rescale the
    # feed-forward at the SAME ff/d_model ratio as the baseline (codec_v3 uses
    # ff=256,d=128 -> ratio 2.0) so width is isolated from FF-ratio; pass --ff
    # to override. The warm-start below is strict=False, so width changes
    # transfer almost nothing (~from-scratch) while depth changes keep the
    # matching layers + conv-tokenizer + heads (partial warm-start). n_heads
    # must divide d_model.
    base_ratio = cfg.get("ff", 256) / max(1, cfg.get("d_model", 128))
    if d_model > 0:
        cfg["d_model"] = d_model
        cfg["ff"] = ff if ff > 0 else int(round(base_ratio * d_model))
    elif ff > 0:
        cfg["ff"] = ff
    if n_layers > 0:
        cfg["n_layers"] = n_layers
    if n_heads > 0:
        cfg["n_heads"] = n_heads
    print(f"[arch] d_model={cfg.get('d_model')} n_layers={cfg.get('n_layers')} "
          f"n_heads={cfg.get('n_heads')} ff={cfg.get('ff')} mid_split={cfg.get('mid_split')}", flush=True)
    model = FrameClassifierTransformerSampleResConvTokMH1Ch(
        use_lead_emb=False, beat_n_classes=N_BEAT, rhythm_n_classes=N_RHYTHM,
        use_logvar=use_logvar, **cfg)

    def _warm_start(blob, tag):
        # strict=False ignores missing/unexpected KEYS but still RAISES on a
        # shape mismatch of a present key, so depth/width overrides must drop
        # shape-incompatible params first. Depth change -> matching transformer
        # layers + conv-tokenizer + heads transfer; width change -> ~nothing
        # transfers (effectively from-scratch, which is the intent).
        raw = {(k[2:] if k.startswith("m.") else k): v for k, v in blob["model_state"].items()}
        tgt = model.state_dict()
        ok = {k: v for k, v in raw.items() if k in tgt and tgt[k].shape == v.shape}
        skipped = len(raw) - len(ok)
        miss, unexp = model.load_state_dict(ok, strict=False)
        print(f"[init from {tag}] transferred {len(ok)}/{len(raw)} params "
              f"(shape-skipped {skipped}, still-missing {len(miss)})", flush=True)

    if init_ckpt:
        _warm_start(torch.load(init_ckpt, map_location="cpu", weights_only=False), init_ckpt)
    else:
        _warm_start(torch.load(E0, map_location="cpu", weights_only=False), "E0")
    if freeze_backbone:
        # FREEZE backbone + frame head; train only the heads in `trainable`. The
        # frozen heads read the same frozen features -> their output == init ckpt
        # EXACTLY (LayerNorm only, no BN running stats; dropout is a train-time
        # no-op at eval). `trainable=("beat_sample.",)` => beat-only upgrade with
        # frame AND rhythm provably identical to the init ckpt.
        tr = tuple(trainable) if trainable else TRAINABLE_WHEN_FROZEN
        n_tr = 0
        for n, p in model.named_parameters():
            p.requires_grad = any(n.startswith(t) for t in tr)
            n_tr += p.numel() if p.requires_grad else 0
        print(f"[freeze-backbone] trainable params: {n_tr}  heads={tr}", flush=True)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="none")
    ap.add_argument("--real", default="real_ml_v2_train.npz")
    ap.add_argument("--val", default="real_ml_v2_val.npz")
    ap.add_argument("--real-frac", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-beat", type=float, default=0.5)
    ap.add_argument("--lambda-rhythm", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rhythm-boost", type=float, default=1.0)
    ap.add_argument("--rhythm-prior", default="",
                    help="comma list of 6 target rhythm-class sampling fractions "
                         "(e.g. test prior '0.705,0.073,0.039,0.101,0.081,0'); reweights "
                         "the sampler to that class balance and DROPS rhythm-IGN windows")
    ap.add_argument("--frame-boost", type=float, default=1.0,
                    help="oversample windows carrying frame GT (LUDB/BUT-PDB) so "
                         "the shared backbone keeps its delineation when rhythm "
                         "sources like lydus/synth (frame=IGN) dominate")
    ap.add_argument("--source-balance", action="store_true",
                    help="equal sampling mass per concatenated --real source")
    ap.add_argument("--num-samples", type=int, default=0,
                    help="sampler draws per epoch (equal-compute ablation); 0=auto (nr*2 / max(nr,ns)*2)")
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-ckpt", default="",
                    help="full-init from this ckpt (e.g. codec_v2) instead of E0 warm-start")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze backbone + frame head; train only rhythm/beat heads "
                         "(frame output stays identical to --init-ckpt)")
    ap.add_argument("--beat-only", action="store_true",
                    help="with --freeze-backbone, train ONLY the beat head — frame "
                         "AND rhythm stay byte-identical to --init-ckpt (pure beat upgrade)")
    ap.add_argument("--frame-only", action="store_true",
                    help="with --freeze-backbone, train ONLY the frame head — rhythm "
                         "AND beat stay byte-identical to --init-ckpt (pure frame upgrade)")
    ap.add_argument("--rhythm-only", action="store_true",
                    help="with --freeze-backbone, train ONLY the rhythm head — frame "
                         "AND beat stay byte-identical to --init-ckpt (pure rhythm upgrade)")
    ap.add_argument("--d-model", type=int, default=0,
                    help="override model width (0=keep v56c d128); rescales ff=4*d_model")
    ap.add_argument("--n-layers", type=int, default=0,
                    help="override transformer depth (0=keep v56c L8)")
    ap.add_argument("--n-heads", type=int, default=0,
                    help="override attention heads (0=keep v56c 4); must divide d_model")
    ap.add_argument("--ff", type=int, default=0,
                    help="override feed-forward dim (0=auto 4*d_model)")
    ap.add_argument("--pre-normalized", action="store_true",
                    help="signals in the cache are already rank-normalized; skip the "
                         "per-window rank_normalize (huge speedup on CPU-limited pods)")
    ap.add_argument("--augment", action="store_true",
                    help="train-only baseline-wander + EMG noise augmentation (labels "
                         "unchanged) -> fixes frame over-segmentation on noisy windows")
    ap.add_argument("--aug-p", type=float, default=0.5, help="augmentation probability")
    ap.add_argument("--frame-focal-gamma", type=float, default=0.0,
                    help="focal-gamma for the FRAME loss (penalize over-confident "
                         "false-positive waves; default 0 = plain CE)")
    ap.add_argument("--frame-smooth", type=float, default=0.0,
                    help="label smoothing for frame CE (soft boundaries / calibration)")
    ap.add_argument("--soft-boundary", action="store_true",
                    help="v7: train frame with per-class gaussian-smoothed SOFT targets "
                         "(calibrated probabilistic boundaries; T edges fuzzy, QRS sharp). "
                         "Models annotation uncertainty; cross-dataset conventions widen, not conflict.")
    ap.add_argument("--soft-sigma-p", type=float, default=6.0, help="P-edge soft width (samples, 500Hz)")
    ap.add_argument("--soft-sigma-qrs", type=float, default=2.5, help="QRS-edge soft width (sharp)")
    ap.add_argument("--soft-sigma-t", type=float, default=12.0, help="T-edge soft width (widest — clinically fuzzy)")
    ap.add_argument("--logvar", action="store_true",
                    help="v7: add per-sample aleatoric uncertainty head + heteroscedastic "
                         "loss (Kendall-Gal). Model learns input-dependent sigma -> high at "
                         "fuzzy/noisy boundaries (per-instance calibration).")
    ap.add_argument("--logvar-mc", type=int, default=8, help="MC samples for heteroscedastic loss")
    ap.add_argument("--ckpt-out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rf = float(args.real_frac)
    print(f"Device {device}  500Hz sample-res MH  real_frac={rf}  seed={args.seed}", flush=True)

    real = _MH(args.real, args.pre_normalized, augment=args.augment, aug_p=args.aug_p)
    valds = _MH(args.val, args.pre_normalized)        # never augment val
    if args.augment:
        print(f"augment ON (baseline-wander+EMG, p={args.aug_p}) — train only", flush=True)
    pure_real = (rf >= 1.0) or (str(args.synth).lower() == "none")
    if pure_real:
        synth = None; nr, ns = len(real), 0
        print(f"real {nr}  synth (skipped, pure-real)  val {len(valds)}", flush=True)
        train = real
        wv = np.ones(nr, dtype=np.float64)
        if args.source_balance and len(real.source_lens) > 1:
            # give each concatenated source EQUAL total sampling mass (else the
            # bigger source dominates — lydus 49k vs wfdb 25k was ~2:1).
            nsrc = len(real.source_lens); off = 0
            for sl in real.source_lens:
                wv[off:off + sl] *= 1.0 / (nsrc * sl)
                off += sl
            print(f"source-balance: {nsrc} sources {real.source_lens} -> equal mass", flush=True)
        if args.rhythm_boost > 1.0:
            rare = np.isin(real.rhythm, [1, 2, 4]).any(axis=1)
            wv[rare] *= args.rhythm_boost
            print(f"rhythm-boost {args.rhythm_boost}x on {int(rare.sum())}/{nr} rare-rhythm windows", flush=True)
        if args.rhythm_prior:
            target = np.array([float(x) for x in args.rhythm_prior.split(",")], dtype=np.float64)
            rlab = real.rhythm[:, 0].astype(np.int64)
            valid = rlab != IGN
            cnt = np.array([(rlab[valid] == c).sum() for c in range(6)], dtype=np.float64)
            wcls = np.where(cnt > 0, target / np.maximum(cnt, 1.0), 0.0)
            sc = np.zeros(nr, dtype=np.float64); sc[valid] = wcls[rlab[valid]]
            wv *= sc
            if wv.sum() <= 0:
                raise SystemExit("rhythm-prior zeroed all windows")
            eff = np.array([wv[valid & (rlab == c)].sum() for c in range(6)]); eff /= eff.sum()
            print(f"rhythm-prior target {target.tolist()} -> effective {np.round(eff, 3).tolist()} "
                  f"(dropped {int((~valid).sum())} IGN-rhythm windows)", flush=True)
        has_frame = (real.frame != IGN).any(axis=1)
        if args.frame_boost > 1.0:
            wv[has_frame] *= args.frame_boost
            print(f"frame-boost {args.frame_boost}x on {int(has_frame.sum())}/{nr} frame-GT windows", flush=True)
        frame_mass = float(wv[has_frame].sum() / wv.sum())
        print(f"effective frame-GT sampling fraction: {frame_mass:.3f}", flush=True)
        ns_draw = args.num_samples if args.num_samples > 0 else nr * 2
        print(f"sampler draws/epoch: {ns_draw}", flush=True)
        sampler = WeightedRandomSampler(torch.tensor(wv, dtype=torch.double),
                                        num_samples=ns_draw, replacement=True)
    else:
        synth = _MH(args.synth, args.pre_normalized); nr, ns = len(real), len(synth)
        print(f"real {nr}  synth {ns}  val {len(valds)}", flush=True)
        train = ConcatDataset([real, synth])
        wr = (rf / nr) if (nr and rf > 0) else 0.0
        wsy = ((1 - rf) / ns) if (ns and rf < 1) else 0.0
        w = np.r_[np.full(nr, wr), np.full(ns, wsy)]
        if w.sum() == 0:
            raise SystemExit("empty sampler (check real_frac / dataset sizes)")
        ns_draw = args.num_samples if args.num_samples > 0 else max(nr, ns) * 2
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=ns_draw, replacement=True)
    tl = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(valds, batch_size=args.batch_size, shuffle=False)

    def _cw(arr, n, cap=20.0):
        vis = arr[arr != IGN]
        c = np.bincount(vis.reshape(-1), minlength=n).astype(np.float64); c = np.maximum(c, 1)
        wv = 1 / np.sqrt(c); wv = wv / wv.mean()
        return torch.tensor(np.minimum(wv, cap), dtype=torch.float32, device=device)
    _wsrc = real if pure_real else synth
    rcw, bcw = _cw(_wsrc.rhythm, N_RHYTHM), _cw(_wsrc.beat, N_BEAT)
    print(f"rhythm cw {rcw.cpu().numpy().round(2)}  beat cw {bcw.cpu().numpy().round(2)}", flush=True)

    _trainable = None
    if args.beat_only:
        _trainable = ("beat_sample.",)
    elif args.frame_only:
        _trainable = ("head_sample.",)
    elif args.rhythm_only:
        _trainable = ("rhythm_sample.",)
    model = build_model(device, args.init_ckpt, args.freeze_backbone,
                        d_model=args.d_model, n_layers=args.n_layers,
                        n_heads=args.n_heads, ff=args.ff, trainable=_trainable,
                        use_logvar=args.logvar)
    print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)

    gamma = float(args.focal_gamma)

    def _ce(logits, target, weight):
        if gamma <= 0:
            return F.cross_entropy(logits, target, weight=weight, ignore_index=IGN)
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=weight, ignore_index=IGN, reduction="none")
        m = (target != IGN)
        tgt = torch.where(m, target, torch.zeros_like(target))
        pt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1).exp()
        loss = ((1 - pt) ** gamma) * ce
        return loss[m].mean() if m.any() else (logits.sum() * 0.0)

    fgamma = float(args.frame_focal_gamma); fsmooth = float(args.frame_smooth)

    # --- v7 soft-boundary: per-class gaussian kernels (P/QRS/T edge widths) ----
    def _gk(sigma):
        r = max(1, int(round(3 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return (k / k.sum()).view(1, 1, -1), r
    _soft_k = {1: _gk(args.soft_sigma_p), 2: _gk(args.soft_sigma_qrs), 3: _gk(args.soft_sigma_t)}

    def _soft_frame_loss(logits, target):     # logits (B,4,T), target (B,T)
        m = (target != IGN)
        th = torch.where(m, target, torch.zeros_like(target)).clamp(0, 3)
        oh = F.one_hot(th, 4).permute(0, 2, 1).float()        # (B,4,T) hard one-hot
        soft = oh.clone()
        for c, (k, r) in _soft_k.items():                     # smear P/QRS/T edges by class width
            soft[:, c:c + 1] = F.conv1d(oh[:, c:c + 1], k, padding=r)
        soft = soft / soft.sum(1, keepdim=True).clamp_min(1e-6)
        loss = -(soft * F.log_softmax(logits, dim=1)).sum(1)  # soft cross-entropy (B,T)
        return loss[m].mean() if m.any() else (logits.sum() * 0.0)

    def _frame_ce(logits, target):
        if args.soft_boundary:
            return _soft_frame_loss(logits, target)
        # frame loss with optional label-smoothing (soft boundaries / calibration)
        # + focal (penalize over-confident false-positive waves -> over-segmentation).
        if fgamma <= 0:
            return F.cross_entropy(logits, target, ignore_index=IGN, label_smoothing=fsmooth)
        ce = F.cross_entropy(logits, target, ignore_index=IGN, label_smoothing=fsmooth, reduction="none")
        m = (target != IGN)
        tgt = torch.where(m, target, torch.zeros_like(target))
        pt = F.log_softmax(logits, dim=1).gather(1, tgt.unsqueeze(1)).squeeze(1).exp()
        loss = ((1 - pt) ** fgamma) * ce
        return loss[m].mean() if m.any() else (logits.sum() * 0.0)

    def _hetero_frame_loss(mu, logvar, target):    # mu (B,T,4), logvar (B,T,1)
        # Kendall-Gal aleatoric: corrupt logits by learned sigma, expected softmax-CE.
        # Ambiguous samples (fuzzy T-off, noise) minimize loss via large sigma.
        m = (target != IGN)
        th = torch.where(m, target, torch.zeros_like(target))
        sigma = torch.exp(0.5 * logvar)
        probs = 0.0
        for _ in range(args.logvar_mc):
            probs = probs + F.softmax(mu + sigma * torch.randn_like(mu), dim=-1)
        p_true = (probs / args.logvar_mc).gather(-1, th.unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(p_true.clamp_min(1e-6))
        return loss[m].mean() if m.any() else (mu.sum() * 0.0)

    def losses(batch):
        sig, lead, fr, rh, bt = [x.to(device) for x in batch]
        out = model(sig, lead)
        frame, beat, rhythm = out[0], out[1], out[2]
        if args.logvar:
            lf = _hetero_frame_loss(frame, out[3], fr)
        else:
            lf = _frame_ce(frame.transpose(1, 2), fr)
        lb = _ce(beat.transpose(1, 2), bt, bcw)
        lr = _ce(rhythm.transpose(1, 2), rh, rcw)
        lf = torch.nan_to_num(lf); lb = torch.nan_to_num(lb); lr = torch.nan_to_num(lr)
        return lf, lb, lr

    if args.dry_run:
        model.eval()
        with torch.no_grad():
            lf, lb, lr = losses(next(iter(tl)))
        print(f"[dry-run] frame={lf.item():.4f} beat={lb.item():.4f} rhythm={lr.item():.4f}", flush=True)
        return 0

    def _macro_f1(pred, true, n_classes, *, drop_class=None):
        pred = np.asarray(pred, dtype=np.int64); true = np.asarray(true, dtype=np.int64)
        vals = []
        for cls in range(n_classes):
            if drop_class is not None and cls == drop_class:
                continue
            sup = int((true == cls).sum())
            if sup == 0:
                continue
            tp = int(((pred == cls) & (true == cls)).sum())
            fp = int(((pred == cls) & (true != cls)).sum())
            fn = int(((pred != cls) & (true == cls)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            vals.append(2 * p * r / (p + r) if p + r else 0.0)
        return float(np.mean(vals)) if vals else 0.0

    @torch.no_grad()
    def real_val():
        model.eval()
        pred_all = {"frame": [], "rhythm": [], "beat": []}
        true_all = {"frame": [], "rhythm": [], "beat": []}
        for sig, lead, fr, rh, bt in vl:
            _o = model(sig.to(device), lead.to(device))
            frame, beat, rhythm = _o[0], _o[1], _o[2]
            for name, pred, tgt in (("frame", frame, fr), ("rhythm", rhythm, rh), ("beat", beat, bt)):
                p = pred.argmax(-1).cpu(); m = (tgt != IGN)
                if name == "beat":
                    m = m & (tgt != 0)
                if m.any():
                    pred_all[name].append(p[m].numpy()); true_all[name].append(tgt[m].numpy())
        scores = {}
        for name, n_cls, drop in (("frame", 4, None), ("rhythm", 6, None), ("beat", 6, 0)):
            if true_all[name]:
                scores[name] = _macro_f1(np.concatenate(pred_all[name]),
                                         np.concatenate(true_all[name]), n_cls, drop_class=drop)
            else:
                scores[name] = 0.0
        return scores

    best = -1.0; Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        t0 = time.time(); model.train(); tot = 0.0; nb = 0
        for batch in tl:
            lf, lb, lr = losses(batch); loss = lf + args.lambda_beat * lb + args.lambda_rhythm * lr
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.item()); nb += 1
        f1 = real_val(); score = f1["frame"] + f1["rhythm"] + f1["beat"]; imp = score > best
        print(f"[ep {ep+1:3d}] {'+' if imp else ' '} loss={tot/max(1,nb):.4f} "
              f"frame_f1={f1['frame']:.4f} rhythm_f1={f1['rhythm']:.4f} beat_f1={f1['beat']:.4f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if imp:
            best = score
            torch.save({"model_state": model.state_dict(), "model_config": dict(model.model_config),
                        "epoch": ep + 1, "real_frac": rf, "seed": args.seed,
                        "val_macro_f1": f1}, args.ckpt_out)
    print(f"\nDone. best real-val macro-F1 sum={best:.4f}  ckpt={args.ckpt_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
