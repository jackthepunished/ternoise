"""QAT training for DenoiseNet on M2 dumps, plus integer-path evaluation.

FP-path PSNR here is a cheap per-epoch proxy; the +6 dB gate is measured by
evaluate() on int_forward output only.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.calibrate import refresh_bias_int
from model.network import DenoiseNet


def _psnr_fp(y, ref):
    # both on the q/128 grid; decoded difference is (y - ref) * 128/255
    d = (y - ref) * (128.0 / 255.0)
    mse = float((d * d).mean())
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def save_checkpoint(path, net, epoch, val_psnr):
    torch.save({"state_dict": net.state_dict(),
                "shifts": [int(c.shift) for c in net.convs],
                "bias_int": [np.asarray(c.bias_int) for c in net.convs],
                "epoch": epoch, "val_psnr_fp": val_psnr}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = DenoiseNet()
    net.load_state_dict(ck["state_dict"])
    for c, s, b in zip(net.convs, ck["shifts"], ck["bias_int"]):
        c.shift = int(s)
        c.bias_int = np.asarray(b, dtype=np.int64)
    return net


def train(net, train_ds, val_ds, *, epochs=150, lr=1e-3, batch_size=8,
          device="cpu", out_dir="runs/v1", recalibrate=None, recalib_epochs=3,
          seed=1058):
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[max(1, int(epochs * 0.8))], gamma=0.1)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(val_ds, batch_size=batch_size)
    best_psnr, best_path = -1.0, out / "best.pt"
    for epoch in range(epochs):
        if recalibrate is not None and epoch < recalib_epochs:
            recalibrate(net)         # refresh shifts while the net is plastic
        net.train()
        loss = torch.tensor(0.0)
        for x, ref in tl:
            x, ref = x.to(device), ref.to(device)
            loss = F.l1_loss(net(x), ref)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            psnrs = [_psnr_fp(net(x.to(device)), ref.to(device))
                     for x, ref in vl]
        vp = float(np.mean(psnrs)) if psnrs else float("nan")
        # bias_int goes stale as the FP biases train (M3: cost 0.95 dB on the
        # gate); checkpoints must always carry values matching current weights
        refresh_bias_int(net)
        save_checkpoint(out / "last.pt", net, epoch, vp)
        if vp > best_psnr:
            best_psnr = vp
            save_checkpoint(best_path, net, epoch, vp)
        print(f"epoch {epoch:3d}  loss {float(loss.detach()):.4f}  "
              f"val_psnr_fp {vp:.2f}",
              flush=True)
    return best_path


def evaluate(net, pairs):
    from model.metrics import psnr_int8, ssim_int8
    net = net.cpu()
    rows = []
    for noisy, ref in pairs:
        den = net.int_forward(np.asarray(noisy, dtype=np.int64))
        rows.append({"psnr_noisy": psnr_int8(noisy[0:3], ref),
                     "psnr_den": psnr_int8(den, ref),
                     "ssim_noisy": ssim_int8(noisy[0:3], ref),
                     "ssim_den": ssim_int8(den, ref)})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    mean["delta_psnr"] = mean["psnr_den"] - mean["psnr_noisy"]
    return {"frames": rows, "mean": mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--recalib-epochs", type=int, default=3)
    ap.add_argument("--val-start", type=int, default=270)
    ap.add_argument("--eval", default=None, metavar="CKPT")
    args = ap.parse_args()
    from model.dataset import DenoisePairs, load_meta, load_pair_int8  # Task 1
    if args.eval:
        net = load_checkpoint(args.eval)
        count = load_meta(args.data)["count"]
        pairs = (load_pair_int8(args.data, v)
                 for v in range(args.val_start, count))
        r = evaluate(net, pairs)
        for v, row in zip(range(args.val_start, count), r["frames"]):
            print(f"v {v:04d}  noisy {row['psnr_noisy']:6.2f}  "
                  f"denoised {row['psnr_den']:6.2f}  ssim {row['ssim_den']:.3f}")
        m = r["mean"]
        print(f"mean  noisy {m['psnr_noisy']:6.2f}  denoised {m['psnr_den']:6.2f}  "
              f"delta {m['delta_psnr']:+.2f} dB  (gate: >= +6.00)")
        return
    from model.calibrate import calibrate  # Task 3
    torch.manual_seed(1058)
    net = DenoiseNet()
    # calibrate on train views only, and never past the end of the dump
    n_calib = max(1, min(8, args.val_start, load_meta(args.data)["count"]))
    calib_frames = [load_pair_int8(args.data, v)[0] for v in range(n_calib)]
    train(net, DenoisePairs(args.data, "train", args.val_start),
          DenoisePairs(args.data, "val", args.val_start),
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
          device=args.device, out_dir=args.out,
          recalibrate=lambda n: calibrate(n, calib_frames),
          recalib_epochs=args.recalib_epochs)


if __name__ == "__main__":
    main()
