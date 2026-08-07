"""PSNR/SSIM on decoded int8 images, plus the noisy-baseline CLI.

Decode per QUANT_SPEC section 2: c = (q + 128) / 255, range [0, 1].
Pure numpy; SSIM is single-scale Wang et al. with an 11x11 Gaussian window.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _decode(q):
    return (np.asarray(q, dtype=np.float64) + 128.0) / 255.0


def psnr_int8(a, b):
    d = _decode(a) - _decode(b)
    mse = float(np.mean(d * d))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gauss_kernel(n=11, sigma=1.5):
    x = np.arange(n) - n // 2
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _filter2(img, k):
    """Separable 2-D filter with reflect padding, no scipy."""
    n = len(k)
    r = n // 2
    h, w = img.shape
    p = np.pad(img, r, mode="reflect")
    tmp = np.zeros((h + 2 * r, w), dtype=np.float64)
    for i in range(n):                       # horizontal pass
        tmp += k[i] * p[:, i:i + w]
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(n):                       # vertical pass
        out += k[i] * tmp[i:i + h, :]
    return out


def ssim_int8(a, b):
    a, b = _decode(a), _decode(b)
    assert a.shape == b.shape and a.ndim == 3, "expect (C, H, W)"
    k = _gauss_kernel()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(a.shape[0]):
        x, y = a[ch], b[ch]
        mx, my = _filter2(x, k), _filter2(y, k)
        vx = _filter2(x * x, k) - mx * mx
        vy = _filter2(y * y, k) - my * my
        cxy = _filter2(x * y, k) - mx * my
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / (
            (mx * mx + my * my + c1) * (vx + vy + c2))
        vals.append(s.mean())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description="noisy-vs-ref baseline over the val split")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-start", type=int, default=270)
    args = ap.parse_args()
    root = Path(args.data)
    meta = json.loads((root / "meta.json").read_text())
    size, count = meta["size"], meta["count"]
    psnrs, ssims = [], []
    for v in range(args.val_start, count):
        noisy = np.fromfile(root / f"frame_v{v:04d}_noisy.bin",
                            dtype=np.int8).reshape(7, size, size)[0:3].astype(np.int64)
        ref = np.fromfile(root / f"frame_v{v:04d}_ref.bin",
                          dtype=np.int8).reshape(3, size, size).astype(np.int64)
        p, s = psnr_int8(noisy, ref), ssim_int8(noisy, ref)
        psnrs.append(p)
        ssims.append(s)
        print(f"v {v:04d} psnr {p:6.2f} ssim {s:.3f}")
    print(f"mean psnr {np.mean(psnrs):6.2f} ssim {np.mean(ssims):.3f}")


if __name__ == "__main__":
    main()
