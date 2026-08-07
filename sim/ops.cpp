#include "ops.h"
#include <algorithm>
#include <cassert>

int64_t requant(int64_t acc, int shift) {
    if (shift > 0) {
        const int64_t h = 1LL << (shift - 1);
        acc = acc >= 0 ? (acc + h) >> shift : -(((-acc) + h) >> shift);
    }
    return std::clamp<int64_t>(acc, -128, 127);
}

Tensor conv3x3(const Tensor& x, const Tensor& w, const Tensor& bias) {
    const int ci_n = x.dims[0], H = x.dims[1], W = x.dims[2], co_n = w.dims[0];
    Tensor acc;
    acc.dims = {co_n, H, W};
    acc.data.assign((size_t)co_n * H * W, 0);
    for (int co = 0; co < co_n; ++co)
        for (int y = 0; y < H; ++y)
            for (int xx = 0; xx < W; ++xx) {
                int64_t a = bias.data[co];
                for (int ci = 0; ci < ci_n; ++ci)
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            const int iy = y + ky - 1, ix = xx + kx - 1;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;  // zero pad
                            const int64_t wv = w.data[((((size_t)co * ci_n + ci) * 3) + ky) * 3 + kx];
                            if (wv == 1)       a += x.data[((size_t)ci * H + iy) * W + ix];
                            else if (wv == -1) a -= x.data[((size_t)ci * H + iy) * W + ix];
                        }
                assert(a < (1LL << 31) && a > -(1LL << 31));
                acc.data[((size_t)co * H + y) * W + xx] = a;
            }
    return acc;
}

Tensor layer(const Tensor& x, const Tensor& w, const Tensor& bias, int shift, bool use_relu) {
    Tensor y = conv3x3(x, w, bias);
    for (auto& v : y.data) {
        v = requant(v, shift);
        if (use_relu && v < 0) v = 0;
    }
    return y;
}

Tensor network(const Tensor& x, const std::vector<Tensor>& ws,
               const std::vector<Tensor>& bs, const std::vector<int>& shifts) {
    Tensor a = x;
    const size_t n = ws.size();
    for (size_t i = 0; i < n; ++i)
        a = layer(a, ws[i], bs[i], shifts[i], /*use_relu=*/i < n - 1);
    // residual head: clamp(x[ch] + y[ch]) for ch 0..2, x is the network input
    const int H = x.dims[1], W = x.dims[2];
    Tensor out;
    out.dims = {3, H, W};
    out.data.resize((size_t)3 * H * W);
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = std::clamp<int64_t>(x.data[i] + a.data[i], -128, 127);
    return out;
}
