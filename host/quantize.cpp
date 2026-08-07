#include "quantize.h"
#include <cmath>

// round-half-away-from-zero, matching the contract (lround does exactly this)
static int8_t q_color(float c) {
    long v = std::lround((double)c * 255.0) - 128;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}
static int8_t q_normal(float n) {
    long v = std::lround((double)n * 127.0);
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}

void quantize_rgb(const float* rgb, int n_px, int8_t* out3) {
    for (int i = 0; i < n_px; ++i)
        for (int c = 0; c < 3; ++c)
            out3[c * n_px + i] = q_color(rgb[i * 4 + c]);
}

void quantize_frame(const float* rgb, const float* normal4, const float* depth,
                    int n_px, int8_t* out7) {
    quantize_rgb(rgb, n_px, out7);
    for (int i = 0; i < n_px; ++i) {
        for (int c = 0; c < 3; ++c)
            out7[(3 + c) * n_px + i] = q_normal(normal4[i * 4 + c]);
        out7[6 * n_px + i] = q_color(depth[i]);   // depth shares the color encoding
    }
}
