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
