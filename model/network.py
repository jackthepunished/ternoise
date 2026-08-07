import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.bitconv import BitConv2d
from model import ref_ops

CHANS = [7, 16, 16, 16, 16, 3]

class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(BitConv2d(CHANS[i], CHANS[i + 1]) for i in range(5))

    def forward(self, x):  # FP training path, mirrors integer topology
        a = x
        for i, conv in enumerate(self.convs):
            a = conv(a)
            if i < 4:
                a = F.relu(a)
        return (x[:, 0:3] + a).clamp(-1, 127 / 128)  # residual head

    def int_forward(self, x_int8: np.ndarray) -> np.ndarray:
        ws = [c.ternary() for c in self.convs]
        bs = [c.bias_int for c in self.convs]
        shifts = [c.shift for c in self.convs]
        return ref_ops.network(x_int8, ws, bs, shifts)
