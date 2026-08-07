"""Golden case dir -> $readmemh inputs for the RTL testbenches.

Emits w{i}.mem (2-bit packed weights), b{i}.mem (int32 two's complement),
and meta.txt (type/chans/shift/relu/dims) per QUANT_SPEC sections 6-7.
"""
import argparse
import json
from pathlib import Path

from tools.pack_weights import bias_to_mem, pack, to_mem, unpack
from vectors.io_text import load_tensor

CHANS_V1 = [7, 16, 16, 16, 16, 3]


def convert(case_dir, out_dir):
    case, out = Path(case_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = json.loads((case / "params.json").read_text())
    x = load_tensor(case / "input.txt")
    lines = []
    if p["type"] == "conv":
        w = load_tensor(case / "weights.txt")
        (out / "w1.mem").write_text(to_mem(pack(w)))
        (out / "b1.mem").write_text(bias_to_mem(load_tensor(case / "bias.txt")))
        lines += ["type conv", "layers 1",
                  f"chans {w.shape[1]} {w.shape[0]}",
                  f"shift {p['shift']}", f"relu {p['relu']}"]
    else:
        shifts, relus = [], []
        for i, layer in enumerate(p["layers"], start=1):
            if p["type"] == "network_packed":
                raw = (case / f"w{i}.bin").read_bytes()
                n = CHANS_V1[i] * CHANS_V1[i - 1] * 9
                unpack(raw, (n,))          # validates: raises on code 11
            else:
                raw = pack(load_tensor(case / f"w{i}.txt"))
            (out / f"w{i}.mem").write_text(to_mem(raw))
            (out / f"b{i}.mem").write_text(
                bias_to_mem(load_tensor(case / f"b{i}.txt")))
            shifts.append(str(layer["shift"]))
            relus.append(str(layer["relu"]))
        lines += ["type network", "layers 5",
                  "chans " + " ".join(str(c) for c in CHANS_V1),
                  "shift " + " ".join(shifts), "relu " + " ".join(relus)]
    lines += [f"height {x.shape[1]}", f"width {x.shape[2]}"]
    (out / "meta.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--out", required=True, dest="out_dir")
    convert(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
