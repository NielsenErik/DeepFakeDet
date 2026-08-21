"""Sanity check: is the 'periphery' of a pristine-background blend really
identical to the real image?  If it is, the leak AUC must be exactly 0.5 and
anything else is a fault in the measurement, not a finding."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcdf.data.sbi import blend_ratio, self_blend  # noqa: E402
from scripts.shortcut_audit import (collect, compression_features,  # noqa: E402
                                    periphery_blocks)

root = Path.home() / "deepfake_data"
items = collect(root, "train", 0, 20, 2)
rng = np.random.default_rng(0)
diffs, featdiffs, nvalid = [], [], []

for path, lmk, _ in items[:40]:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    seed = int(rng.integers(1 << 30))
    b, m = self_blend(img, lmk, np.random.default_rng(seed),
                      post_compress=False, pristine_background=True)
    if not (0.02 < blend_ratio(m) < 0.9):
        continue
    valid = periphery_blocks(m, img.shape[:2])
    bh, bw = valid.shape
    # expand the block mask to pixels
    pix = np.kron(valid[:bh, :bw], np.ones((8, 8), bool))
    h, w = pix.shape
    d = np.abs(b[:h, :w].astype(int) - img[:h, :w].astype(int)).max(2)
    diffs.append(d[pix].max())
    nvalid.append(int(valid.sum()))
    fr = compression_features(img, valid)
    fb = compression_features(b, valid)
    if fr is not None and fb is not None:
        featdiffs.append(float(np.abs(fr - fb).max()))

print(f"images checked            : {len(diffs)}")
print(f"valid periphery blocks    : mean {np.mean(nvalid):.0f}")
print(f"max |blend-real| on those : max {max(diffs)}  mean {np.mean(diffs):.3f}")
print(f"fraction with ANY diff    : {np.mean([d > 0 for d in diffs]):.2f}")
print(f"max |feature difference|  : {max(featdiffs):.3e}")
