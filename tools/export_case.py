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
