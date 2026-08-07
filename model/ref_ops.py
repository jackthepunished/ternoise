"""Integer reference for the QUANT_SPEC contract. numpy int64 everywhere."""
import numpy as np

def requant(acc, shift):
    acc = acc.astype(np.int64)
    if shift > 0:
        h = np.int64(1) << (shift - 1)
        acc = np.where(acc >= 0, (acc + h) >> shift, -(((-acc) + h) >> shift))
    return np.clip(acc, -128, 127)

def relu(q):
    return np.maximum(q, 0)

def conv3x3(x, w, bias):
    c_in, h_, w_ = x.shape
    c_out = w.shape[0]
    assert w.shape == (c_out, c_in, 3, 3)
    xp = np.zeros((c_in, h_ + 2, w_ + 2), dtype=np.int64)
    xp[:, 1:-1, 1:-1] = x
    acc = np.zeros((c_out, h_, w_), dtype=np.int64)
    for co in range(c_out):
        for ci in range(c_in):
            for ky in range(3):
                for kx in range(3):
                    wv = int(w[co, ci, ky, kx])
                    if wv:
                        acc[co] += wv * xp[ci, ky:ky + h_, kx:kx + w_]
    acc += bias[:, None, None]
    assert np.abs(acc).max() < 2**31, "contract guarantees int32 range"
    return acc

def layer(x, w, bias, shift, use_relu):
    y = requant(conv3x3(x, w, bias), shift)
    return relu(y) if use_relu else y

def residual_add(rgb_in, res):
    return np.clip(rgb_in.astype(np.int64) + res.astype(np.int64), -128, 127)

def network(x, weights, biases, shifts):
    a = x
    for i in range(len(weights)):
        a = layer(a, weights[i], biases[i], shifts[i], use_relu=(i < len(weights) - 1))
    return residual_add(x[0:3], a)
