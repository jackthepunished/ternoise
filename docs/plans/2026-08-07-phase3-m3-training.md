# Phase 3 (M3) Implementation Plan — Training, Calibration & Software Denoising

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallel execution:** Tasks 1-4 are designed for four parallel agents, each in
> its own worktree, with disjoint file ownership (see the Agent Assignment table).
> Task 5 is sequential integration, run in the main session with Bahadir after
> Tasks 1-4 have merged.

**Goal:** Train `DenoiseNet` with QAT on `data/v1`, replace the M1 calibration stubs (`shift=5`, `bias_int=0`) with real per-layer values, export packed weights, and prove the C++ golden model denoises a real frame bit-exact vs `int_forward` — gated on **+6.0 dB mean PSNR over the noisy baseline on the validation split, measured on the integer path**. No RTL work starts until the gate passes.

**Architecture:** The loader feeds int8-grid float tensors (`q/128`). The FP training forward gains a per-layer power-of-two scale `2^-shift` so its topology mirrors the integer path (this is how QUANT_SPEC section 1's "gamma is absorbed by the requant shift" becomes real). Calibration picks each shift from accumulator percentile stats and derives `bias_int` from the FP bias. Export writes a golden-vector-format case directory (`type: network_packed`) whose `expected.txt` comes from `int_forward`, so the **unmodified** `./golden` binary is the C++-vs-PyTorch bit-exactness check on real frames — no C++ changes anywhere in M3.

**Tech Stack:** As M1/M2 — Python (numpy, torch, pytest), C++17 golden model. New: CUDA torch wheel for the RTX 5070 (installed only in Task 5 — the `.venv` is shared across worktrees, agents must not touch it). No new Python dependencies: SSIM is hand-rolled numpy.

## Global Constraints

- QUANT_SPEC is normative; the integer path (`ref_ops`, `int_forward`, C++) is
  untouched by this phase except where stated (Task 3 modifies only the FP
  forward and device handling in `model/bitconv.py`).
- Scale conventions, used consistently everywhere:
  - int8 activation `q` <-> training float `x = q / 128` (range `[-1, 127/128]`).
  - Metric/display decode: `c = (q + 128) / 255` in `[0, 1]` (QUANT_SPEC section 2).
- Dataset split: views `0..269` train, `270..299` val (`VAL_START = 270`, defined
  in `model/dataset.py`; other modules take it as a parameter defaulting to 270 —
  no cross-module import needed).
- **The gate:** `mean over val of PSNR(int_forward output, ref) - mean over val of
  PSNR(noisy RGB, ref) >= +6.0 dB`. PSNR on decoded RGB. FP-path PSNR is a
  training-time proxy only; the gate is measured on the integer path.
- FP <-> int correspondence (derivation, referenced by Tasks 3 and 4):
  with ternary weights `Wt`, FP forward `y = 2^-s * conv(x, Wt) + b` and
  `x = q/128`, the integer accumulator is `acc = conv(q, Wt) = 128 * conv(x, Wt)`,
  so `128*y = 2^-s * acc + 128*b`. The integer layer computes
  `(acc + bias_int) * 2^-s`, hence `bias_int = round(128 * b * 2^s)`.
- Shift selection: smallest `s in [0, 15]` such that
  `percentile(|acc * 2^-s + 128*b|, 99.9) <= 127` over the calibration frames
  (falls back to 15 if none fits). Clamping to `[0, 15]` per QUANT_SPEC section 4.
- Determinism: seed 1058 for every RNG (torch, numpy, DataLoader shuffle).
- No data augmentation in v1 (a horizontal flip must negate the Nx channel;
  not worth the subtlety for 300 frames — revisit only if the gate fails).
- `data/v1` frames are stored bottom-up (`"row0": "bottom"` in meta.json, GL
  readback order). Noisy and ref share the orientation, so training and metrics
  are unaffected; only cross-checking against PNGs needs a flip.
- No emojis anywhere. C++ flags unchanged. TDD: failing test first, one commit
  per green step. numpy int64 for all integer-path math.
- New gitignore entry: `runs/` (checkpoints). `data/` already ignored.

## Agent Assignment

| Agent | Task | Creates | Modifies |
|-------------|------|---------------------------------------------------------------|-----------------------|
| `m3-loader` | 1 | `model/dataset.py`, `tests/test_dataset.py` | — |
| `m3-metrics`| 2 | `model/metrics.py`, `tests/test_metrics.py` | — |
| `m3-calib` | 3 | `model/calibrate.py`, `tools/export_case.py`, `tests/test_calibrate.py` | `model/bitconv.py` |
| `m3-train` | 4 | `model/train.py`, `tests/test_train.py` | `.gitignore` |

Rules for agents:
- Work in your own worktree on a branch named after your agent (`m3-loader` etc.).
- Touch only the files in your row. File sets are disjoint; merges are clean.
- Before your final commit: your task's tests green AND the full `make test`
  green (M1/M2 regression).
- Merge order: Tasks 1, 2, 3 in any order; Task 4 merges last (its `evaluate`
  test imports `model.metrics` via `pytest.importorskip`, so it degrades to a
  skip when run stand-alone, but the full suite needs 1-3 in place).
- Cross-task interfaces are specified exactly in each task's Interfaces block.
  Code against those signatures; do not import another task's module except
  where a task explicitly says so.

---

### Task 1 (agent `m3-loader`): Dataset loader

**Files:**
- Create: `model/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `data/v1`-format dump dirs (M2): `frame_v%04d_noisy.bin` (7*S*S int8,
  planar `[R,G,B,Nx,Ny,Nz,D]`), `frame_v%04d_ref.bin` (3*S*S int8), `meta.json`
  with keys `size`, `count`.
- Produces (later tasks rely on these exact names):
  - `VAL_START = 270` (module constant)
  - `load_meta(root) -> dict`
  - `load_pair_int8(root, v) -> tuple[np.ndarray, np.ndarray]` — noisy
    `(7,S,S)` and ref `(3,S,S)`, both `np.int64` (ready for `int_forward` /
    `ref_ops`).
  - `DenoisePairs(root, split="train", val_start=VAL_START)` — torch `Dataset`;
    `__getitem__` returns `(noisy, ref)` float32 tensors `(7,S,S)` / `(3,S,S)`
    with values `q/128`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.dataset import DenoisePairs, load_meta, load_pair_int8


def make_dump(root: Path, count: int = 3, size: int = 8) -> Path:
    """Synthetic M2-format dump with deterministic contents."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1058)
    for v in range(count):
        noisy = rng.integers(-128, 128, size=(7, size, size)).astype(np.int8)
        ref = rng.integers(-128, 128, size=(3, size, size)).astype(np.int8)
        (root / f"frame_v{v:04d}_noisy.bin").write_bytes(noisy.tobytes())
        (root / f"frame_v{v:04d}_ref.bin").write_bytes(ref.tobytes())
    (root / "meta.json").write_text(json.dumps(
        {"size": size, "ref_spp": 16, "count": count, "seed": 1058,
         "channels": "R,G,B,Nx,Ny,Nz,D", "hdr_clamp": [0, 1], "row0": "bottom"}))
    return root


def test_meta_and_int8_pair(tmp_path):
    make_dump(tmp_path)
    assert load_meta(tmp_path)["size"] == 8
    noisy, ref = load_pair_int8(tmp_path, 1)
    assert noisy.shape == (7, 8, 8) and noisy.dtype == np.int64
    assert ref.shape == (3, 8, 8) and ref.dtype == np.int64
    raw = np.fromfile(tmp_path / "frame_v0001_noisy.bin", dtype=np.int8)
    assert np.array_equal(noisy.ravel(), raw.astype(np.int64))


def test_split_membership(tmp_path):
    make_dump(tmp_path, count=3)
    tr = DenoisePairs(tmp_path, split="train", val_start=2)
    va = DenoisePairs(tmp_path, split="val", val_start=2)
    assert len(tr) == 2 and len(va) == 1
    with pytest.raises(ValueError):
        DenoisePairs(tmp_path, split="test", val_start=2)


def test_values_are_q_over_128(tmp_path):
    make_dump(tmp_path, count=1)
    ds = DenoisePairs(tmp_path, split="train", val_start=1)
    x, y = ds[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    raw = np.fromfile(tmp_path / "frame_v0000_noisy.bin",
                      dtype=np.int8).reshape(7, 8, 8)
    assert np.allclose(x.numpy(), raw.astype(np.float32) / 128.0)
    assert x.min() >= -1.0 and x.max() <= 127.0 / 128.0


def test_truncated_file_rejected(tmp_path):
    make_dump(tmp_path, count=1)
    p = tmp_path / "frame_v0000_ref.bin"
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):
        load_pair_int8(tmp_path, 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.dataset'`

- [ ] **Step 3: Implement `model/dataset.py`**

```python
"""Loader for M2 dataset dumps. int8 q <-> training float q/128 (QUANT_SPEC)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VAL_START = 270  # views [0, VAL_START) train, [VAL_START, count) val


def load_meta(root):
    return json.loads((Path(root) / "meta.json").read_text())


def _load_bin(path, channels, size):
    arr = np.fromfile(path, dtype=np.int8)
    want = channels * size * size
    if arr.size != want:
        raise ValueError(f"{path}: {arr.size} values, want {want}")
    return arr.reshape(channels, size, size).astype(np.int64)


def load_pair_int8(root, v):
    root = Path(root)
    size = load_meta(root)["size"]
    noisy = _load_bin(root / f"frame_v{v:04d}_noisy.bin", 7, size)
    ref = _load_bin(root / f"frame_v{v:04d}_ref.bin", 3, size)
    return noisy, ref


class DenoisePairs(Dataset):
    def __init__(self, root, split="train", val_start=VAL_START):
        if split not in ("train", "val"):
            raise ValueError(f"unknown split '{split}'")
        self.root = Path(root)
        count = load_meta(self.root)["count"]
        vs = min(val_start, count)
        self.views = list(range(vs)) if split == "train" else list(range(vs, count))

    def __len__(self):
        return len(self.views)

    def __getitem__(self, i):
        noisy, ref = load_pair_int8(self.root, self.views[i])
        return (torch.from_numpy(noisy.astype(np.float32) / 128.0),
                torch.from_numpy(ref.astype(np.float32) / 128.0))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_dataset.py -q`
Expected: 4 passed

- [ ] **Step 5: Full regression, then commit**

Run: `make test`
Expected: everything green (no existing file was touched).

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "feat(model): dataset loader for INT8 pair dumps"
```

---

### Task 2 (agent `m3-metrics`): PSNR/SSIM metrics + noisy-baseline CLI

**Files:**
- Create: `model/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from other tasks. The CLI reads dump files directly with
  `np.fromfile` (deliberately no `model.dataset` import — keeps Task 2
  independently mergeable).
- Produces (Task 4's `evaluate` relies on these exact names):
  - `psnr_int8(a, b) -> float` — int arrays, same shape (typically `(3,H,W)`),
    decoded `(q+128)/255` before MSE; returns `float("inf")` for identical inputs.
  - `ssim_int8(a, b) -> float` — mean over channels of single-scale SSIM on the
    decoded images: 11x11 Gaussian window sigma 1.5, reflect padding,
    `C1 = 0.01**2`, `C2 = 0.03**2` (standard Wang et al. constants, data range 1).
  - CLI: `python -m model.metrics --data DIR [--val-start 270]` — prints one line
    per val view `v NNN psnr X.XX ssim 0.XXX` (noisy RGB vs ref) and a final
    `mean psnr X.XX ssim 0.XXX` line. This mean PSNR is the noisy baseline the
    +6 dB gate is measured against.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import math

import numpy as np
import pytest

from model.metrics import psnr_int8, ssim_int8


def test_psnr_identical_is_inf():
    a = np.arange(-128, 127, dtype=np.int64).reshape(1, 5, 51)
    assert psnr_int8(a, a) == float("inf")


def test_psnr_constant_offset_closed_form():
    # decoded difference is exactly 16/255 everywhere
    a = np.full((3, 8, 8), -128, dtype=np.int64)
    b = np.full((3, 8, 8), -112, dtype=np.int64)
    want = 20.0 * math.log10(255.0 / 16.0)
    assert psnr_int8(a, b) == pytest.approx(want, rel=1e-9)
    assert psnr_int8(b, a) == pytest.approx(want, rel=1e-9)  # symmetric


def test_ssim_self_is_one_and_noise_degrades():
    rng = np.random.default_rng(1058)
    a = rng.integers(-128, 128, size=(3, 32, 32)).astype(np.int64)
    assert ssim_int8(a, a) == pytest.approx(1.0, abs=1e-9)
    noise = rng.integers(-40, 41, size=a.shape)
    b = np.clip(a + noise, -128, 127)
    s = ssim_int8(a, b)
    assert s < 0.99
    assert s == pytest.approx(ssim_int8(b, a), abs=1e-12)


def test_ssim_smooth_images_score_higher_than_noisy():
    # structural similarity: a small constant shift barely hurts, noise hurts a lot
    yy = np.linspace(-100, 100, 32)
    a = np.clip(np.round(np.tile(yy, (32, 1))), -128, 127).astype(np.int64)[None]
    shifted = np.clip(a + 4, -128, 127)
    rng = np.random.default_rng(7)
    noisy = np.clip(a + rng.integers(-30, 31, size=a.shape), -128, 127)
    assert ssim_int8(a, shifted) > ssim_int8(a, noisy)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.metrics'`

- [ ] **Step 3: Implement `model/metrics.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_metrics.py -q`
Expected: 4 passed

- [ ] **Step 5: Record the real baseline (needs `data/v1` present)**

Run: `.venv/bin/python -m model.metrics --data data/v1`
Expected: 30 per-frame lines and a `mean psnr ... ssim ...` line. Paste the mean
line into the task's commit message body — it is the number the gate is measured
against. (If `data/v1` is missing in the worktree, regenerate per CLAUDE.md or
run against the main checkout's `data/v1` path; do not skip this step.)

- [ ] **Step 6: Full regression, then commit**

Run: `make test`
Expected: green.

```bash
git add model/metrics.py tests/test_metrics.py
git commit -m "feat(model): PSNR/SSIM metrics and noisy-baseline CLI"
```

---

### Task 3 (agent `m3-calib`): Calibration, export, C++ bit-exact on real-shape frames

**Files:**
- Create: `model/calibrate.py`, `tools/export_case.py`
- Modify: `model/bitconv.py` (the FP `forward` and `.cpu()` device-safety only —
  the integer path and `ternary()` semantics are untouched)
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `model.ref_ops` (M1 signatures), `tools.pack_weights.pack`,
  `vectors.io_text.save_tensor`, the existing `./golden` runner (`network_packed`
  case type — verified: it takes any input H x W, fixed v1 CHANS table).
- Produces (Task 4 and Task 5 rely on these exact names):
  - `BitConv2d.forward` now computes
    `y = conv(xq, Wt_ste) * 2**-shift + bias` (bias moved out of `F.conv2d`).
    Signature unchanged; `DenoiseNet.forward` unchanged.
  - `choose_shift(acc_nobias, bias_fp, pctl=99.9) -> int` — `acc_nobias` float
    array `(N,C,H,W)`, `bias_fp` float `(C,)`; smallest fitting shift per the
    Global Constraints rule.
  - `calibrate(net, frames, pctl=99.9) -> None` — `frames` is a list of
    `(7,H,W)` int64 arrays; sets `conv.shift` (int) and `conv.bias_int`
    (np.int64 `(C,)`) on every layer, propagating each frame through the
    integer path layer by layer. Works on a CUDA-resident net (reads params
    via `.cpu()`).
  - `export_case(net, frame_int8, out_dir) -> None` — writes `input.txt`,
    `w1..w5.bin` (2-bit packed), `b1..b5.txt`, `params.json`
    (`type network_packed`, per-layer shift/relu), and `expected.txt`
    = `net.int_forward(frame_int8)`. A directory `./golden` accepts as-is.
  - CLI: `python -m tools.export_case --ckpt CKPT --data DIR --view V --out DIR`
    (uses `model.train.load_checkpoint` — import inside `main()` only, so the
    module works before Task 4 merges).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
import subprocess
from pathlib import Path

import numpy as np
import torch

from model import ref_ops
from model.bitconv import BitConv2d
from model.calibrate import calibrate, choose_shift
from model.network import DenoiseNet

ROOT = Path(__file__).resolve().parents[1]


def seeded_net():
    torch.manual_seed(1058)
    return DenoiseNet()


def test_fp_forward_applies_pow2_scale_and_bias():
    m = BitConv2d(1, 1)
    with torch.no_grad():
        m.weight.copy_(torch.ones(1, 1, 3, 3))   # ternarizes to all +1
        m.bias.copy_(torch.tensor([0.25]))
    m.shift = 3
    x = torch.full((1, 1, 4, 4), 64.0 / 128.0)   # exact int8-grid value
    y = m(x)
    # interior pixel: conv sum = 9 * 0.5 = 4.5 -> y = 4.5 / 2**3 + 0.25
    assert torch.allclose(y[0, 0, 1, 1], torch.tensor(4.5 / 8.0 + 0.25))
    # corner pixel (zero padding, 4 taps): 4 * 0.5 / 8 + 0.25
    assert torch.allclose(y[0, 0, 0, 0], torch.tensor(2.0 / 8.0 + 0.25))


def test_choose_shift_smallest_fitting():
    rng = np.random.default_rng(1058)
    acc = rng.integers(-30000, 30001, size=(2, 4, 8, 8)).astype(np.float64)
    b = np.zeros(4)
    s = choose_shift(acc, b)
    assert 0 <= s <= 15
    assert np.percentile(np.abs(acc / 2.0 ** s), 99.9) <= 127.0
    if s > 0:
        assert np.percentile(np.abs(acc / 2.0 ** (s - 1)), 99.9) > 127.0
    assert choose_shift(np.zeros((1, 1, 2, 2)), np.zeros(1)) == 0


def test_calibrate_sets_shift_and_bias_int():
    net = seeded_net()
    rng = np.random.default_rng(1058)
    frames = [rng.integers(-128, 128, size=(7, 16, 16)).astype(np.int64)
              for _ in range(2)]
    calibrate(net, frames)
    for conv in net.convs:
        assert 0 <= conv.shift <= 15
        b_fp = conv.bias.detach().cpu().numpy().astype(np.float64)
        want = np.round(128.0 * (2.0 ** conv.shift) * b_fp).astype(np.int64)
        assert np.array_equal(conv.bias_int, want)
    # deterministic
    net2 = seeded_net()
    calibrate(net2, frames)
    assert [c.shift for c in net2.convs] == [c.shift for c in net.convs]


def test_export_case_is_bit_exact_in_cpp(tmp_path):
    from tools.export_case import export_case
    net = seeded_net()
    rng = np.random.default_rng(1058)
    frame = rng.integers(-128, 128, size=(7, 64, 64)).astype(np.int64)
    calibrate(net, [frame])
    export_case(net, frame, tmp_path)
    for f in ["input.txt", "expected.txt", "params.json",
              "w1.bin", "w5.bin", "b1.txt", "b5.txt"]:
        assert (tmp_path / f).exists()
    subprocess.run(["make", "golden"], cwd=ROOT, check=True, capture_output=True)
    r = subprocess.run([str(ROOT / "golden"), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_calibrate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.calibrate'`

- [ ] **Step 3: Modify `model/bitconv.py`**

Replace `ternary()`'s first line and `forward()` (everything else unchanged):

```python
    def ternary(self):
        w = self.weight.detach().cpu()
        gamma = w.abs().mean()
        if gamma == 0:
            return np.zeros(tuple(w.shape), dtype=np.int64)
        # round-half-away-from-zero to match QUANT_SPEC (torch.round is half-even)
        r = w / gamma
        t = torch.sign(r) * torch.floor(r.abs() + 0.5)
        return t.clamp(-1, 1).numpy().astype(np.int64)
```

```python
    def forward(self, x):
        xq = _round_ste(x.clamp(-1, 127 / 128) * 128) / 128  # int8-grid activations
        y = F.conv2d(xq, self._fake_quant_weight(), None, padding=1)
        # mirror the integer path: requant's 2^-shift, then the bias
        # (QUANT_SPEC section 1: gamma's effect lives in the requant shift)
        return y * (2.0 ** -float(self.shift)) + self.bias.view(1, -1, 1, 1)
```

Then run the M1 regression immediately:

Run: `.venv/bin/pytest tests/test_bitconv.py -q`
Expected: 4 passed (the int path and gradient flow are unaffected; only FP
forward magnitudes changed, which no M1 test pins).

- [ ] **Step 4: Implement `model/calibrate.py`**

```python
"""Calibration: turn the M1 stubs (shift=5, bias_int=0) into real values.

Derivation (see the plan's Global Constraints): with FP forward
y = 2^-s * conv(x, Wt) + b and x = q/128, matching the integer layer
(acc + bias_int) * 2^-s requires bias_int = round(128 * b * 2^s).
The shift is the smallest s whose requantized activations (bias included)
fit int8 at the 99.9th percentile over the calibration frames.
"""
import numpy as np

from model import ref_ops


def choose_shift(acc_nobias, bias_fp, pctl=99.9):
    scaled_bias = 128.0 * bias_fp[None, :, None, None]
    for s in range(16):
        m = np.percentile(np.abs(acc_nobias * 2.0 ** -s + scaled_bias), pctl)
        if m <= 127.0:
            return s
    return 15


def calibrate(net, frames, pctl=99.9):
    xs = [np.asarray(f, dtype=np.int64) for f in frames]
    for i, conv in enumerate(net.convs):
        wt = conv.ternary()
        b_fp = conv.bias.detach().cpu().numpy().astype(np.float64)
        zero_bias = np.zeros(wt.shape[0], dtype=np.int64)
        accs = [ref_ops.conv3x3(x, wt, zero_bias) for x in xs]
        s = choose_shift(np.stack(accs).astype(np.float64), b_fp, pctl)
        conv.shift = s
        conv.bias_int = np.round(128.0 * (2.0 ** s) * b_fp).astype(np.int64)
        use_relu = i < len(net.convs) - 1
        xs = []
        for a in accs:
            y = ref_ops.requant(a + conv.bias_int[:, None, None], s)
            xs.append(ref_ops.relu(y) if use_relu else y)
```

- [ ] **Step 5: Implement `tools/export_case.py`**

```python
"""Export a DenoiseNet as a golden-vector case dir (type network_packed).

expected.txt is int_forward's own output, so `./golden <dir>` passing proves
the C++ model matches the PyTorch integer path bit-for-bit on this input.
"""
import json
from pathlib import Path

import numpy as np

from tools.pack_weights import pack
from vectors.io_text import save_tensor


def export_case(net, frame_int8, out_dir):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    frame_int8 = np.asarray(frame_int8, dtype=np.int64)
    save_tensor(d / "input.txt", frame_int8)
    layers = []
    n = len(net.convs)
    for i, conv in enumerate(net.convs, start=1):
        (d / f"w{i}.bin").write_bytes(pack(conv.ternary()))
        save_tensor(d / f"b{i}.txt", conv.bias_int)
        layers.append({"shift": int(conv.shift), "relu": int(i < n)})
    save_tensor(d / "expected.txt", net.int_forward(frame_int8))
    (d / "params.json").write_text(json.dumps(
        {"type": "network_packed", "layers": layers}) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--view", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from model.dataset import load_pair_int8   # Task 1
    from model.train import load_checkpoint    # Task 4; import here, not module top
    net = load_checkpoint(args.ckpt)
    noisy, _ = load_pair_int8(args.data, args.view)
    export_case(net, noisy, args.out)
    print(f"exported view {args.view} to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/pytest tests/test_calibrate.py -q`
Expected: 4 passed (the last one builds and runs `./golden`, printing `PASS`).

- [ ] **Step 7: Full regression, then commit**

Run: `make test`
Expected: green — the M1 vector cases still pass because the integer path is
byte-identical; only the FP training forward changed.

```bash
git add model/bitconv.py model/calibrate.py tools/export_case.py tests/test_calibrate.py
git commit -m "feat(model): real per-layer shift calibration and packed-case export, C++ bit-exact on real-shape frames"
```

---

### Task 4 (agent `m3-train`): QAT training loop + integer-path evaluation

**Files:**
- Create: `model/train.py`
- Modify: `.gitignore` (add `runs/` line)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `DenoiseNet` (M1). The train loop takes any torch `Dataset` of
    `(x float32 (7,S,S), ref float32 (3,S,S))` pairs — tests inject a
    `TensorDataset`, the CLI wires Task 1's `DenoisePairs`.
  - Optional `recalibrate` callback (the CLI wires Task 3's
    `calibrate(net, frames)`; core code never imports `model.calibrate`).
  - `model.metrics.psnr_int8` / `ssim_int8` (Task 2) — imported inside
    `evaluate` only.
- Produces:
  - `train(net, train_ds, val_ds, *, epochs=150, lr=1e-3, batch_size=8,
    device="cpu", out_dir="runs/v1", recalibrate=None, recalib_epochs=3,
    seed=1058) -> Path` — returns the best-checkpoint path (`best.pt`); also
    writes `last.pt` every epoch. Adam, L1 loss, MultiStepLR x0.1 at 80% of
    epochs, `recalibrate(net)` called at the start of epochs `0..recalib_epochs-1`
    (shifts freeze afterwards), best = highest FP-proxy val PSNR.
  - `save_checkpoint(path, net, epoch, val_psnr)` / `load_checkpoint(path) ->
    DenoiseNet` — checkpoint dict holds `state_dict`, per-layer `shifts`,
    `bias_int` lists, `epoch`, `val_psnr_fp` (shift/bias_int are plain
    attributes, not in `state_dict`, so they are saved explicitly).
  - `evaluate(net, pairs) -> dict` — `pairs` iterable of int64
    `(noisy (7,H,W), ref (3,H,W))`; runs `int_forward`; returns
    `{"frames": [...], "mean": {"psnr_noisy", "psnr_den", "ssim_noisy",
    "ssim_den", "delta_psnr"}}`. `delta_psnr` is the gate number.
  - CLI: `python -m model.train --data DIR --out DIR [--epochs N] [--lr F]
    [--batch-size N] [--device cpu|cuda] [--recalib-epochs N] [--val-start N]`
    to train, or `python -m model.train --eval CKPT --data DIR [--val-start N]`
    to print the per-frame and mean gate table.
- **Environment note:** the CUDA wheel swap happens in Task 5 only. This task's
  code must run on CPU torch. Never modify the shared `.venv` from a worktree.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from model.network import DenoiseNet
from model.train import load_checkpoint, save_checkpoint, train


def synth_ds(n=8, s=16, seed=1058):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-128, 128, (n, 7, s, s), generator=g).float() / 128.0
    ref = x[:, 0:3].clone()          # ideal answer: zero residual
    return TensorDataset(x, ref)


def test_train_smoke_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1058)
    net = DenoiseNet()
    best = train(net, synth_ds(8), synth_ds(2, seed=7), epochs=2, batch_size=4,
                 out_dir=tmp_path, recalibrate=None, recalib_epochs=0)
    assert best.exists() and (tmp_path / "last.pt").exists()
    net2 = load_checkpoint(best)
    for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())
    assert [c.shift for c in net2.convs] == [c.shift for c in net.convs]
    for c, c2 in zip(net.convs, net2.convs):
        assert np.array_equal(c.bias_int, c2.bias_int)


def test_recalibrate_called_only_in_early_epochs(tmp_path):
    calls = []
    torch.manual_seed(1058)
    net = DenoiseNet()
    train(net, synth_ds(4), synth_ds(2, seed=7), epochs=3, batch_size=4,
          out_dir=tmp_path, recalibrate=lambda n: calls.append(1),
          recalib_epochs=2)
    assert len(calls) == 2


def test_evaluate_reports_gate_delta(tmp_path):
    pytest.importorskip("model.metrics")     # Task 2; skip when run stand-alone
    from model.train import evaluate
    torch.manual_seed(1058)
    net = DenoiseNet()
    rng = np.random.default_rng(1058)
    pairs = [(rng.integers(-128, 128, size=(7, 16, 16)).astype(np.int64),
              rng.integers(-128, 128, size=(3, 16, 16)).astype(np.int64))]
    r = evaluate(net, pairs)
    m = r["mean"]
    assert set(m) == {"psnr_noisy", "psnr_den", "ssim_noisy", "ssim_den",
                      "delta_psnr"}
    assert m["delta_psnr"] == pytest.approx(m["psnr_den"] - m["psnr_noisy"])
    assert len(r["frames"]) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.train'`

- [ ] **Step 3: Implement `model/train.py`**

```python
"""QAT training for DenoiseNet on M2 dumps, plus integer-path evaluation.

FP-path PSNR here is a cheap per-epoch proxy; the +6 dB gate is measured by
evaluate() on int_forward output only.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.network import DenoiseNet


def _psnr_fp(y, ref):
    # both on the q/128 grid; decoded difference is (y - ref) * 128/255
    d = (y - ref) * (128.0 / 255.0)
    mse = float((d * d).mean())
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def save_checkpoint(path, net, epoch, val_psnr):
    torch.save({"state_dict": net.state_dict(),
                "shifts": [int(c.shift) for c in net.convs],
                "bias_int": [np.asarray(c.bias_int) for c in net.convs],
                "epoch": epoch, "val_psnr_fp": val_psnr}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = DenoiseNet()
    net.load_state_dict(ck["state_dict"])
    for c, s, b in zip(net.convs, ck["shifts"], ck["bias_int"]):
        c.shift = int(s)
        c.bias_int = np.asarray(b, dtype=np.int64)
    return net


def train(net, train_ds, val_ds, *, epochs=150, lr=1e-3, batch_size=8,
          device="cpu", out_dir="runs/v1", recalibrate=None, recalib_epochs=3,
          seed=1058):
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[max(1, int(epochs * 0.8))], gamma=0.1)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(val_ds, batch_size=batch_size)
    best_psnr, best_path = -1.0, out / "best.pt"
    for epoch in range(epochs):
        if recalibrate is not None and epoch < recalib_epochs:
            recalibrate(net)         # refresh shifts while the net is plastic
        net.train()
        loss = torch.tensor(0.0)
        for x, ref in tl:
            x, ref = x.to(device), ref.to(device)
            loss = F.l1_loss(net(x), ref)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            psnrs = [_psnr_fp(net(x.to(device)), ref.to(device))
                     for x, ref in vl]
        vp = float(np.mean(psnrs)) if psnrs else float("nan")
        save_checkpoint(out / "last.pt", net, epoch, vp)
        if vp > best_psnr:
            best_psnr = vp
            save_checkpoint(best_path, net, epoch, vp)
        print(f"epoch {epoch:3d}  loss {float(loss):.4f}  val_psnr_fp {vp:.2f}",
              flush=True)
    return best_path


def evaluate(net, pairs):
    from model.metrics import psnr_int8, ssim_int8
    net = net.cpu()
    rows = []
    for noisy, ref in pairs:
        den = net.int_forward(np.asarray(noisy, dtype=np.int64))
        rows.append({"psnr_noisy": psnr_int8(noisy[0:3], ref),
                     "psnr_den": psnr_int8(den, ref),
                     "ssim_noisy": ssim_int8(noisy[0:3], ref),
                     "ssim_den": ssim_int8(den, ref)})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    mean["delta_psnr"] = mean["psnr_den"] - mean["psnr_noisy"]
    return {"frames": rows, "mean": mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--recalib-epochs", type=int, default=3)
    ap.add_argument("--val-start", type=int, default=270)
    ap.add_argument("--eval", default=None, metavar="CKPT")
    args = ap.parse_args()
    from model.dataset import DenoisePairs, load_meta, load_pair_int8  # Task 1
    if args.eval:
        net = load_checkpoint(args.eval)
        count = load_meta(args.data)["count"]
        pairs = (load_pair_int8(args.data, v)
                 for v in range(args.val_start, count))
        r = evaluate(net, pairs)
        for v, row in zip(range(args.val_start, count), r["frames"]):
            print(f"v {v:04d}  noisy {row['psnr_noisy']:6.2f}  "
                  f"denoised {row['psnr_den']:6.2f}  ssim {row['ssim_den']:.3f}")
        m = r["mean"]
        print(f"mean  noisy {m['psnr_noisy']:6.2f}  denoised {m['psnr_den']:6.2f}  "
              f"delta {m['delta_psnr']:+.2f} dB  (gate: >= +6.00)")
        return
    from model.calibrate import calibrate  # Task 3
    torch.manual_seed(1058)
    net = DenoiseNet()
    calib_frames = [load_pair_int8(args.data, v)[0] for v in range(8)]
    train(net, DenoisePairs(args.data, "train", args.val_start),
          DenoisePairs(args.data, "val", args.val_start),
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
          device=args.device, out_dir=args.out,
          recalibrate=lambda n: calibrate(n, calib_frames),
          recalib_epochs=args.recalib_epochs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `runs/` to `.gitignore`**

Append the line `runs/` to `.gitignore`.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_train.py -q`
Expected: 2 passed + 1 pass-or-skip (`test_evaluate_reports_gate_delta` skips
until Task 2 is merged; it must pass in the integrated tree).

- [ ] **Step 6: Full regression, then commit**

Run: `make test`
Expected: green.

```bash
git add model/train.py tests/test_train.py .gitignore
git commit -m "feat(model): QAT training loop with integer-path gate evaluation"
```

---

### Task 5 (integration — main session with Bahadir, after Tasks 1-4 merge): Train, gate, close out

**Files:**
- Modify: `README.md` (tick M3), `CLAUDE.md` (status), new `devlog/` entry
- No new source files; everything below uses Task 1-4 deliverables.

- [ ] **Step 1: Merge Tasks 1-4, full suite green**

Run: `make test`
Expected: everything green in the merged tree, including the previously-skipped
`test_evaluate_reports_gate_delta`.

- [ ] **Step 2: Swap in the CUDA torch wheel (RTX 5070, CUDA 13.2)**

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True` and a name containing `5070`. If the wheel/driver combo fails,
fall back to CPU training (slow but functional) and flag it to Bahadir before
burning hours.

Run: `make test`
Expected: still green on the CUDA wheel (everything CPU-path in tests).

- [ ] **Step 3: Record the noisy baseline**

Run: `.venv/bin/python -m model.metrics --data data/v1`
Expected: 30 lines + mean. Save the mean for the devlog and the gate arithmetic.

- [ ] **Step 4: Train**

```bash
.venv/bin/python -m model.train --data data/v1 --out runs/v1 --device cuda
```

Expected: 150 epoch lines; `val_psnr_fp` trending up; `runs/v1/best.pt` written.
Watch the first few epochs: if loss is flat or NaN, stop and debug (with
`superpowers:systematic-debugging`) instead of waiting out the run.

- [ ] **Step 5: The gate**

```bash
.venv/bin/python -m model.train --eval runs/v1/best.pt --data data/v1
```

Expected: per-frame table + `mean ... delta +X.XX dB (gate: >= +6.00)`.
- **Gate passes (delta >= +6.00):** proceed.
- **Gate fails:** no RTL. Iterate hyperparameters (epochs, lr schedule,
  percentile, recalib epochs) and record every attempt in the devlog. Any change
  beyond hyperparameters (architecture, scales, augmentation) needs a new plan
  discussed with Bahadir first.

- [ ] **Step 6: C++ denoises a real frame, bit-exact**

```bash
.venv/bin/python -m tools.export_case --ckpt runs/v1/best.pt --data data/v1 --view 270 --out /tmp/m3_case
make golden && ./golden /tmp/m3_case
```

Expected: `PASS /tmp/m3_case` — the C++ golden model reproduces the PyTorch
integer path on a real rendered frame, bit for bit, through packed weights.

- [ ] **Step 7: Close out**

- Devlog entry: baseline PSNR, gate delta, training wall-clock, GPU used,
  hyperparameters, and the `./golden` PASS line.
- README: tick M3. CLAUDE.md: update Status (M3 done, M4 next).
- Run: `make test` one final time, expected green.

```bash
git add README.md CLAUDE.md devlog/
git commit -m "docs: tick M3 - trained ternary denoiser passes +6 dB gate, C++ bit-exact on real frames"
git push
```

**M3 exit criterion:** `make test` green; `runs/v1/best.pt` exists; the eval
table shows `delta >= +6.00 dB` on the val split via the integer path;
`./golden` PASSes on an exported real-frame case; docs and devlog updated.

---

## Execution order and gates

Tasks 1, 2, 3 are pairwise independent (disjoint files) — run in parallel, merge
in any order. Task 4 merges last. Task 5 is strictly sequential and interactive:
it changes the shared `.venv`, runs the real training, and makes the gate call
that decides whether M4 (RTL) gets written at all.
