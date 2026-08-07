"""Regenerate all golden vectors. Deterministic: seed 1058."""
import argparse, json
from pathlib import Path
import numpy as np
from model import ref_ops
from vectors.io_text import save_tensor

SEED = 1058

def rand_ternary(rng, shape):
    return rng.integers(-1, 2, size=shape).astype(np.int64)

def rand_act(rng, shape):
    return rng.integers(-128, 128, size=shape).astype(np.int64)

def emit_conv(d, x, w, b, shift, use_relu):
    d.mkdir(parents=True, exist_ok=True)
    save_tensor(d / "input.txt", x); save_tensor(d / "weights.txt", w)
    save_tensor(d / "bias.txt", b)
    save_tensor(d / "expected.txt", ref_ops.layer(x, w, b, shift, use_relu))
    (d / "params.json").write_text(json.dumps(
        {"type": "conv", "shift": shift, "relu": int(use_relu)}) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent))
    out = Path(ap.parse_args().out)
    rng = np.random.default_rng(SEED)

    # case01: tiny hand-checkable conv (matches tests/test_ref_ops.py by hand)
    x = np.arange(1, 17, dtype=np.int64).reshape(1, 4, 4)
    w = np.array([[[[1, 0, -1]] * 3]], dtype=np.int64)
    emit_conv(out / "case01_conv_hand", x, w, np.array([10], dtype=np.int64), 2, True)

    # case02: random 8->8 over 16x16, hits general datapath
    emit_conv(out / "case02_conv_rand", rand_act(rng, (8, 16, 16)),
              rand_ternary(rng, (8, 8, 3, 3)),
              rng.integers(-1000, 1000, size=8).astype(np.int64), 5, True)

    # case03: extremes - all +1 weights, all 127 input, C_in=32 (accumulator max + clamp)
    emit_conv(out / "case03_conv_extreme",
              np.full((32, 8, 8), 127, dtype=np.int64),
              np.ones((4, 32, 3, 3), dtype=np.int64),
              np.zeros(4, dtype=np.int64), 5, False)

    # case04: full network v1, 7ch 16x16, shifts fixed for the vector
    d = out / "case04_network"; d.mkdir(parents=True, exist_ok=True)
    x = rand_act(rng, (7, 16, 16))
    chans = [7, 16, 16, 16, 16, 3]
    shifts = [5, 5, 5, 5, 5]
    ws, bs = [], []
    for i in range(5):
        ws.append(rand_ternary(rng, (chans[i + 1], chans[i], 3, 3)))
        bs.append(rng.integers(-500, 500, size=chans[i + 1]).astype(np.int64))
        save_tensor(d / f"w{i+1}.txt", ws[-1]); save_tensor(d / f"b{i+1}.txt", bs[-1])
    save_tensor(d / "input.txt", x)
    save_tensor(d / "expected.txt", ref_ops.network(x, ws, bs, shifts))
    (d / "params.json").write_text(json.dumps(
        {"type": "network",
         "layers": [{"shift": s, "relu": int(i < 4)} for i, s in enumerate(shifts)]}) + "\n")

if __name__ == "__main__":
    main()
