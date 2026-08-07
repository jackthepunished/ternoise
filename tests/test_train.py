import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from model.network import DenoiseNet
from model.train import load_checkpoint, save_checkpoint, train


def synth_ds(n=8, s=16, seed=1058):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-128, 128, (n, 7, s, s), generator=g).float() / 128.0
    ref = x[:, 0:3].clone()          # ideal answer: zero residual
    return TensorDataset(x, ref)


def test_train_smoke_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1058)
    net = DenoiseNet()
    best = train(net, synth_ds(8), synth_ds(2, seed=7), epochs=2, batch_size=4,
                 out_dir=tmp_path, recalibrate=None, recalib_epochs=0)
    assert best.exists() and (tmp_path / "last.pt").exists()
    # last.pt is written from the net as it stands now; best.pt may hold an
    # earlier epoch (best = highest FP-proxy val PSNR, which need not be the
    # final epoch), so the roundtrip is checked against last.pt.
    net2 = load_checkpoint(tmp_path / "last.pt")
    for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())
    assert [c.shift for c in net2.convs] == [c.shift for c in net.convs]
    for c, c2 in zip(net.convs, net2.convs):
        assert np.array_equal(c.bias_int, c2.bias_int)
    # best.pt is a valid checkpoint too, and records the epoch it came from
    assert load_checkpoint(best) is not None
    ck = torch.load(best, map_location="cpu", weights_only=False)
    assert 0 <= ck["epoch"] < 2 and "val_psnr_fp" in ck


def test_recalibrate_called_only_in_early_epochs(tmp_path):
    calls = []
    torch.manual_seed(1058)
    net = DenoiseNet()
    train(net, synth_ds(4), synth_ds(2, seed=7), epochs=3, batch_size=4,
          out_dir=tmp_path, recalibrate=lambda n: calls.append(1),
          recalib_epochs=2)
    assert len(calls) == 2


def test_evaluate_reports_gate_delta(tmp_path):
    pytest.importorskip("model.metrics")     # Task 2; skip when run stand-alone
    from model.train import evaluate
    torch.manual_seed(1058)
    net = DenoiseNet()
    rng = np.random.default_rng(1058)
    pairs = [(rng.integers(-128, 128, size=(7, 16, 16)).astype(np.int64),
              rng.integers(-128, 128, size=(3, 16, 16)).astype(np.int64))]
    r = evaluate(net, pairs)
    m = r["mean"]
    assert set(m) == {"psnr_noisy", "psnr_den", "ssim_noisy", "ssim_den",
                      "delta_psnr"}
    assert m["delta_psnr"] == pytest.approx(m["psnr_den"] - m["psnr_noisy"])
    assert len(r["frames"]) == 1
