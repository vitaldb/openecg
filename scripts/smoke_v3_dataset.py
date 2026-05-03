"""Smoke test for v3 CombinedFrameDataset."""
import time

from ecgcode.stage2.multi_dataset import CombinedFrameDataset

t0 = time.time()
ds_train = CombinedFrameDataset(["ludb_train", "qtdb", "isp_train"])
print(f"TRAIN: {len(ds_train)} sequences in {time.time()-t0:.1f}s")
print(f"  source_counts: {dict(ds_train.source_counts())}")
print(f"  label_counts: {ds_train.label_counts()}")
sig, lid, lab = ds_train[0]
print(f"  sample shapes: sig={sig.shape}, lead_id={lid.item()}, labels={lab.shape}")

t0 = time.time()
ds_val_ludb = CombinedFrameDataset(["ludb_val"])
print(f"LUDB VAL: {len(ds_val_ludb)} in {time.time()-t0:.1f}s")

t0 = time.time()
ds_val_isp = CombinedFrameDataset(["isp_test"])
print(f"ISP TEST: {len(ds_val_isp)} in {time.time()-t0:.1f}s")
