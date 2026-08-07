import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.pack_weights import pack, to_mem
from vectors.io_text import load_tensor

ROOT = Path(__file__).resolve().parents[1]


def run_tool(case, out):
    return subprocess.run(
        [sys.executable, "-m", "tools.case_to_mem", str(case), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True)


def test_conv_case(tmp_path):
    case = ROOT / "vectors" / "case01_conv_hand"
    r = run_tool(case, tmp_path)
    assert r.returncode == 0, r.stderr
    w = load_tensor(case / "weights.txt")
    assert (tmp_path / "w1.mem").read_text() == to_mem(pack(w))
    meta = dict(line.split(None, 1) for line in
                (tmp_path / "meta.txt").read_text().splitlines())
    assert meta["type"] == "conv"
    assert meta["chans"] == "1 1"
    assert meta["shift"] == "2" and meta["relu"] == "1"
    assert meta["height"] == "4" and meta["width"] == "4"


def test_network_case(tmp_path):
    r = run_tool(ROOT / "vectors" / "case04_network", tmp_path)
    assert r.returncode == 0, r.stderr
    for i in range(1, 6):
        assert (tmp_path / f"w{i}.mem").exists()
        assert (tmp_path / f"b{i}.mem").exists()
    meta = dict(line.split(None, 1) for line in
                (tmp_path / "meta.txt").read_text().splitlines())
    assert meta["chans"] == "7 16 16 16 16 3"
    assert meta["shift"] == "5 5 5 5 5"
    assert meta["relu"] == "1 1 1 1 0"


def test_network_packed_case_roundtrips(tmp_path):
    case = ROOT / "vectors" / "case05_network_packed"
    r = run_tool(case, tmp_path)
    assert r.returncode == 0, r.stderr
    # packed input: bin bytes must reproduce the .mem exactly
    raw = (case / "w1.bin").read_bytes()
    assert (tmp_path / "w1.mem").read_text() == to_mem(raw)


def test_bias_mem_is_twos_complement(tmp_path):
    run_tool(ROOT / "vectors" / "case04_network", tmp_path)
    b1 = load_tensor(ROOT / "vectors" / "case04_network" / "b1.txt")
    lines = (tmp_path / "b1.mem").read_text().splitlines()
    assert len(lines) == len(b1)
    assert int(lines[0], 16) == int(b1[0]) & 0xFFFFFFFF
