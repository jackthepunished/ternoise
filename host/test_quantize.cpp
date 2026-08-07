// QUANT_SPEC section 2 examples: endpoints and half-away rounding
#include "quantize.h"
#include <cstdio>

static int fail(int i, long got, long want) {
    std::printf("quantize FAIL ch %d: got %ld want %ld\n", i, got, want);
    return 1;
}
int main() {
    // one pixel: color (0, 0.5, 1), normal (-1, 0, 1), depth 0.25
    const float rgb[4] = {0.0f, 0.5f, 1.0f, 0.0f};
    const float norm4[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    const float depth[1] = {0.25f};
    int8_t out[7];
    quantize_frame(rgb, norm4, depth, 1, out);
    // color: 0 -> -128; 0.5*255 = 127.5 -> half-away 128 -> 0; 1 -> 127
    // normal: round(n*127); depth: 0.25*255 = 63.75 -> 64 -> -64
    const long want[7] = {-128, 0, 127, -127, 0, 127, -64};
    for (int i = 0; i < 7; ++i)
        if (out[i] != want[i]) return fail(i, out[i], want[i]);
    std::puts("test_quantize PASS");
    return 0;
}
