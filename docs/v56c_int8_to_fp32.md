# v56c: from `boundary_int8.tflite` back to fp32

`openecg/models/boundary_int8.tflite` stores the v56c boundary detector with int8
weights (per-channel scales) and float activations. Before the fp32 checkpoint was
released (`v56c.pt`, release `boundary-v56c`), an fp32 model could only be had by
rebuilding the torch graph from the TFLite file. This page records how that rebuild
is done and how close it lands to the released weights.

The released `v56c.pt` / `v56c_fp32.onnx` are the reference; nothing here replaces
them. `scripts/reconstruct_boundary_from_tflite.py` is a cross-check of the int8 file.

## Conversion steps

`vit_transformer_noaux_1ch` (`FrameClassifierTransformerNoAux1Ch`): 4 lower conv
blocks + 4 upper transformer layers (d = 128, 4 heads, ff = 256), patch 5,
992,266 parameters. Input: one lead, 2500 samples at 250 Hz, rank-normalized
(`openecg.deploy.preprocess_window`). Outputs: `cls_logits [500, 4]`,
`reg_offsets [500, 6]`.

1. **Dequantize** every weight tensor of the TFLite file:
   `w = scale_c · (q − zero_point_c)` per output channel.
2. **Map** the TFLite tensors onto the torch module by index (table in the script).
   Nothing is fitted. Three folds done by the TFLite converter are undone:
   - `patch_embed.bias` was added into the positional-encoding constant, so the
     rebuild sets `patch_embed.bias = 0` and takes `pos_enc[:500]` from the folded
     constant;
   - every upper transformer layer is pre-LN; the converter multiplied the LayerNorm
     affine (`γ`, `β`) into the next linear layer (`in_proj` for `norm1`, `linear1`
     for `norm2`): `W' = W · diag(γ)`, `b' = b + W β`. The rebuild keeps `W'`, `b'`
     and sets the upper `norm1` / `norm2` affine to identity, which gives the same
     function;
   - the lower-block LayerNorm constants are stored in reverse block order
     (block 0 uses MUL 77 / ADD 76 … block 3 uses MUL 71 / ADD 70).
3. **Check** that the rebuilt torch model reproduces the TFLite outputs
   (the script asserts `max |Δ cls_logits| < 1e-3` on `data/samples/*_10s_250hz.npy`
   windows when present).
4. **Write** the rebuilt checkpoint and an fp32 ONNX export (opset 17, static
   `1 × 2500`) under `data/` (not committed).

```bash
pip install torch onnx onnxscript ai-edge-litert
python -m scripts.reconstruct_boundary_from_tflite \
    --compare-ckpt data/checkpoints/v56c.pt --ludb-val --json out/reconstruction.json
```

## How close the rebuild is to the released weights

Measured with `--compare-ckpt data/checkpoints/v56c.pt --ludb-val` (torch 2.15.0,
ai-edge-litert 2.3.0, Windows 11, Intel Core Ultra 9 285H CPU).

**Weights.** The released weights are folded the same way the converter folds them
(above), then compared tensor by tensor with the rebuild (71 tensors):

| Tensors | Count | max \|Δ\| |
|---|---:|---|
| Biases, LayerNorm params, positional encoding (stored in float in the TFLite file) | 40 | 0 (bit-identical) |
| Folded upper-layer biases (`in_proj_bias`, `linear1.bias`: `b + W β`) | 8 | ≤ 2.4e-7 (float32 rounding of the fold) |
| int8 weight matrices / conv kernels | 23 | ≤ half an int8 step of the output channel (`max|w_c| / 254`); worst ratio 1.000 (to float rounding), largest absolute 0.0137 (`upper_transformer.layers.3.linear2.weight`) |

So every int8 weight lands within rounding distance of the released value, and every
other parameter is exact. The largest difference relative to a tensor's own range is
0.39 %.

**Outputs** on the 64 LUDB val windows (`--ludb-val`: val split, 500 → 250 Hz by
`[::2]`, rank-normalized):

| Pair | max \|Δ\| `cls_logits` | max \|Δ\| `reg_offsets` | frame-argmax agreement |
|---|---:|---:|---:|
| rebuilt fp32 vs int8 TFLite | 2.07e-5 | 1.58e-4 | 100 % |
| rebuilt fp32 vs released fp32 | 0.226 | 1.53 | 99.85 % |
| int8 TFLite vs released fp32 | 0.226 | 1.53 | 99.85 % |

The rebuild reproduces the TFLite model, not the released fp32 model: the gap between
rebuilt and released is the int8 weight rounding already present in
`boundary_int8.tflite`.

**Boundary macro-F1** (Martínez tolerances; same protocol and splits as the v56c table
in [`v56c_MODEL_CARD.md`](../openecg/models/v56c_MODEL_CARD.md): lead II, first lead for
QTDB, with reg-head refinement as in `scripts/benchmark_v56c.py`). Scored with
`scripts/eval_macro_f1_openvino.py` from the OpenVINO branch (`--backends tflite,ort,torch:cpu`,
then `--ckpt` / `--onnx` pointed at the rebuild):

| Corpus (n) | int8 TFLite | released fp32 (`v56c.pt`, torch CPU) | released `v56c_fp32.onnx` (ORT CPU) | rebuilt fp32 (torch CPU / ORT CPU) |
|---|---:|---:|---:|---:|
| LUDB val (41) | 0.9633 | 0.9633 | 0.9633 | 0.9633 |
| ISP test (72) | 0.9711 | 0.9711 | 0.9711 | 0.9711 |
| QTDB T-subset (44) | 0.9076 | 0.9083 | 0.9083 | 0.9076 |
| **mean** | **0.9473** | **0.9476** | **0.9476** | **0.9473** |

Frame-argmax agreement with the TFLite model over all scored windows: released fp32
0.9987 (LUDB) / 0.9985 (ISP) / 0.9978 (QTDB); rebuilt fp32 1.000 on all three.

The rebuilt model scores exactly like the TFLite file it came from. The released
fp32 model matches the table (0.963 / 0.971 / 0.908); its QTDB score is 0.0007 higher
than the int8 file.
