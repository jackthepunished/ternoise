import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _round_ste(x):
    return (x.round() - x).detach() + x

class BitConv2d(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in, 3, 3))
        nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.bias_int = np.zeros(c_out, dtype=np.int64)  # eval-path bias (M3 calibrates)
        self.shift = 5                                    # eval-path shift (M3 calibrates)

    def ternary(self):
        w = self.weight.detach().cpu()
        gamma = w.abs().mean()
        if gamma == 0:
            return np.zeros(tuple(w.shape), dtype=np.int64)
        # round-half-away-from-zero to match QUANT_SPEC (torch.round is half-even)
        r = w / gamma
        t = torch.sign(r) * torch.floor(r.abs() + 0.5)
        return t.clamp(-1, 1).numpy().astype(np.int64)

    def _fake_quant_weight(self):
        gamma = self.weight.abs().mean().clamp(min=1e-8)
        r = self.weight / gamma
        t = torch.sign(r) * torch.floor(r.abs() + 0.5)
        tq = t.clamp(-1, 1)
        return (tq - r).detach() + r  # STE

    def forward(self, x):
        xq = _round_ste(x.clamp(-1, 127 / 128) * 128) / 128  # int8-grid activations
        y = F.conv2d(xq, self._fake_quant_weight(), None, padding=1)
        # mirror the integer path: requant's 2^-shift, then the bias
        # (QUANT_SPEC section 1: gamma's effect lives in the requant shift)
        return y * (2.0 ** -float(self.shift)) + self.bias.view(1, -1, 1, 1)
