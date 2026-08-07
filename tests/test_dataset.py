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
