import numpy as np
import torch
from pathlib import Path
from model.bitconv import BitConv2d
from model.network import DenoiseNet
from vectors.io_text import load_tensor

ROOT = Path(__file__).resolve().parents[1]

def test_ternary_absmean_example():
    m = BitConv2d(1, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.weight.view(-1)[0:3] = torch.tensor([0.4, -0.05, 0.3])
    t = m.ternary()  # gamma = mean|W| over all 9 -> small -> clamps kick in
    assert t.min() >= -1 and t.max() <= 1

def test_ternary_identity_on_ternary_master():
    # setting the FP master to a ternary tensor must reproduce it exactly
    m = BitConv2d(2, 3)
    w = np.random.default_rng(7).integers(-1, 2, size=(3, 2, 3, 3))
    with torch.no_grad():
        m.weight.copy_(torch.tensor(w, dtype=torch.float32))
    assert np.array_equal(m.ternary(), w)

def test_int_forward_matches_golden_network_case():
    d = ROOT / "vectors" / "case04_network"
    net = DenoiseNet()
    for i, conv in enumerate(net.convs, start=1):
        w = load_tensor(d / f"w{i}.txt"); b = load_tensor(d / f"b{i}.txt")
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(w, dtype=torch.float32))
        conv.bias_int = b
        conv.shift = 5
    got = net.int_forward(load_tensor(d / "input.txt"))
    assert np.array_equal(got, load_tensor(d / "expected.txt"))

def test_training_forward_backward_runs():
    net = DenoiseNet()
    x = torch.rand(2, 7, 16, 16) * 2 - 1
    y = net(x)
    assert y.shape == (2, 3, 16, 16)
    y.abs().mean().backward()
    g = net.convs[0].weight.grad
    assert g is not None and g.abs().sum() > 0  # STE lets gradients through
