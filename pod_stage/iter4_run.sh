#!/bin/bash
# Iter 4a: rhythm-only linear probe re-fit at the NATURAL test prior (zero regression).
# Unlike iter2 (bias = threshold shift only), a linear re-fit can also re-weight features.
# Drops rhythm-IGN windows; reweights sampler to test class balance (70% sinus).
cd /root/loop
PRIOR="0.705,0.073,0.039,0.101,0.081,0"
echo "##### iter4a: rhythm-only @ natural prior"
python3 kgpu_train.py \
  --real real_ml_v2_train.npz,lydus_trainsub.npz --val lydus_dev_nat.npz \
  --real-frac 1.0 --synth none --init-ckpt codec_v4.pt \
  --freeze-backbone --rhythm-only --rhythm-prior "$PRIOR" \
  --epochs 12 --batch-size 32 --lr 3e-4 --ckpt-out iter4a_rp.pt 2>&1 \
  | grep -vi "warning\|nested" | tail -7

echo ""
echo "##### iter4a RAW (no bias) — nat-dev + TEST"
python3 eval_lydus_rhythm.py iter4a_rp.pt --test lydus_dev_nat.npz 2>&1 | grep -E "macro"
python3 eval_lydus_rhythm.py iter4a_rp.pt --test lydus_rhythm_test.npz 2>&1 | grep -E "sinus|avb|paced|afib|bbb|macro|acc"

echo ""
echo "##### iter4a + logit-bias (stack), then canonical gate"
python3 iter2_logit_bias.py --ckpt iter4a_rp.pt --dev lydus_dev_nat.npz \
  --test lydus_rhythm_test.npz --out iter4a_biasadj.pt 2>&1 | grep -iE "TEST macro|final bias"
echo "-- iter4a_biasadj canonical gate --"
python3 eval_lydus_rhythm.py iter4a_biasadj.pt --test lydus_rhythm_test.npz 2>&1 | grep -E "sinus|avb|paced|afib|bbb|macro|acc"
echo "ITER4_DONE"
