"""Pack ternary weights to the QUANT_SPEC 2-bit format (.bin) and .mem files."""
import argparse
from pathlib import Path
import numpy as np

ENC = {0: 0b00, 1: 0b01, -1: 0b10}
DEC = {0b00: 0, 0b01: 1, 0b10: -1}

def pack(w: np.ndarray) -> bytes:
    flat = w.reshape(-1)
    out = bytearray((len(flat) + 3) // 4)
    for i, v in enumerate(flat):
        out[i // 4] |= ENC[int(v)] << (2 * (i % 4))
    return bytes(out)

def unpack(raw: bytes, shape) -> np.ndarray:
    n = int(np.prod(shape))
    vals = np.empty(n, dtype=np.int64)
    for i in range(n):
        code = (raw[i // 4] >> (2 * (i % 4))) & 0b11
        if code == 0b11:
            raise ValueError(f"illegal weight encoding 11 at index {i}")
        vals[i] = DEC[code]
    return vals.reshape(shape)

def to_mem(raw: bytes) -> str:
    return "".join(f"{b:02x}\n" for b in raw)

def bias_to_mem(b: np.ndarray) -> str:
    return "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in b)

def main():
    from vectors.io_text import load_tensor
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.case_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        w = load_tensor(src / f"w{i}.txt")
        raw = pack(w)
        (out / f"w{i}.bin").write_bytes(raw)
        (out / f"w{i}.mem").write_text(to_mem(raw))
        (out / f"b{i}.mem").write_text(bias_to_mem(load_tensor(src / f"b{i}.txt")))

if __name__ == "__main__":
    main()
