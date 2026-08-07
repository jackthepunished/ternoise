import subprocess
from pathlib import Path

import numpy as np
import torch

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
