import json, subprocess, sys
from pathlib import Path
import numpy as np
from vectors.io_text import load_tensor
from model import ref_ops

ROOT = Path(__file__).resolve().parents[1]
CASES = ["case01_conv_hand", "case02_conv_rand", "case03_conv_extreme", "case04_network"]

def test_cases_exist_and_verify():
    for name in CASES:
        d = ROOT / "vectors" / name
        p = json.loads((d / "params.json").read_text())
        x = load_tensor(d / "input.txt")
        exp = load_tensor(d / "expected.txt")
        if p["type"] == "conv":
            got = ref_ops.layer(x, load_tensor(d / "weights.txt"),
                                load_tensor(d / "bias.txt"), p["shift"], bool(p["relu"]))
        else:
            ws = [load_tensor(d / f"w{i}.txt") for i in range(1, 6)]
            bs = [load_tensor(d / f"b{i}.txt") for i in range(1, 6)]
            got = ref_ops.network(x, ws, bs, [l["shift"] for l in p["layers"]])
        assert np.array_equal(got, exp), name

def test_regeneration_is_deterministic(tmp_path):
    subprocess.run([sys.executable, "-m", "vectors.gen_vectors", "--out", str(tmp_path)],
                   check=True, cwd=ROOT)
    for name in CASES:
        for f in sorted((ROOT / "vectors" / name).iterdir()):
            assert (tmp_path / name / f.name).read_bytes() == f.read_bytes(), f
