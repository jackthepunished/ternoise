"""Calibration: turn the M1 stubs (shift=5, bias_int=0) into real values.

Derivation (see the M3 plan's Global Constraints): with FP forward
y = 2^-s * conv(x, Wt) + b and x = q/128, matching the integer layer
(acc + bias_int) * 2^-s requires bias_int = round(128 * b * 2^s).
The shift is the smallest s whose requantized activations (bias included)
fit int8 at the 99.9th percentile over the calibration frames.
"""
import numpy as np

from model import ref_ops


def choose_shift(acc_nobias, bias_fp, pctl=99.9):
    scaled_bias = 128.0 * bias_fp[None, :, None, None]
    for s in range(16):
        m = np.percentile(np.abs(acc_nobias * 2.0 ** -s + scaled_bias), pctl)
        if m <= 127.0:
            return s
    return 15


def calibrate(net, frames, pctl=99.9):
    xs = [np.asarray(f, dtype=np.int64) for f in frames]
    for i, conv in enumerate(net.convs):
        wt = conv.ternary()
        b_fp = conv.bias.detach().cpu().numpy().astype(np.float64)
        zero_bias = np.zeros(wt.shape[0], dtype=np.int64)
        accs = [ref_ops.conv3x3(x, wt, zero_bias) for x in xs]
        s = choose_shift(np.stack(accs).astype(np.float64), b_fp, pctl)
        conv.shift = s
        conv.bias_int = np.round(128.0 * (2.0 ** s) * b_fp).astype(np.int64)
        use_relu = i < len(net.convs) - 1
        xs = []
        for a in accs:
            y = ref_ops.requant(a + conv.bias_int[:, None, None], s)
            xs.append(ref_ops.relu(y) if use_relu else y)


def refresh_bias_int(net):
    """Recompute every layer's bias_int from its CURRENT FP bias.

    calibrate() sets bias_int once, but the FP biases keep training
    afterwards, so anything consuming bias_int later (checkpoints, export,
    the gate eval) must refresh first. Found in M3: stale biases cost
    0.95 dB on the val gate. Shifts are left untouched - the FP forward
    trained against them.
    """
    for conv in net.convs:
        b_fp = conv.bias.detach().cpu().numpy().astype(np.float64)
        conv.bias_int = np.round(128.0 * (2.0 ** conv.shift) * b_fp).astype(np.int64)
