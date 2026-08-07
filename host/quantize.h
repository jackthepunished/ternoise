#pragma once
#include <cstdint>

// QUANT_SPEC section 2. Inputs are the GL readback layouts:
// rgb: RGBA32F (stride 4, alpha ignored), normal4: RGBA32F, depth: R32F (stride 1).
// out7 layout is channel-major planes: [R, G, B, Nx, Ny, Nz, D], each n_px long.
void quantize_frame(const float* rgb, const float* normal4, const float* depth,
                    int n_px, int8_t* out7);
// Same color encoding for the 3-channel reference, planes [R, G, B].
void quantize_rgb(const float* rgb, int n_px, int8_t* out3);
