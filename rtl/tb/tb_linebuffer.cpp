// Streams random frames through Vlinebuffer with random stalls; checks every
// window against a software reference. Bit-exact, zero tolerance.
//
// Handshake sampling: inputs are applied, the model is evaluated with clk low
// so the combinational outputs settle, the valid/ready pairs and m_win are
// sampled at that point (this is what the DUT registers see at the edge), and
// only then does the posedge happen. Sampling after the posedge would read the
// next cycle's ready/valid against this cycle's drive.
#include "Vlinebuffer.h"
#include "verilated.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int C = 16;

struct Ref {  // golden windows for one frame
    int h, w;
    std::vector<int8_t> px;  // pixel (y, x) channel c at (y*w + x)*C + c
    int8_t at(int y, int x, int c) const {
        if (y < 0 || y >= h || x < 0 || x >= w) return 0;
        return px[((size_t)y * (size_t)w + (size_t)x) * C + c];
    }
};

int main(int argc, char** argv) {
    VerilatedContext ctx;
    ctx.commandArgs(argc, argv);
    Vlinebuffer dut{&ctx};
    std::mt19937 rng(1058);

    auto lo = [&] { dut.clk = 0; dut.eval(); };   // settle combinational logic
    auto hi = [&] { dut.clk = 1; dut.eval(); };   // posedge

    auto reset = [&] {
        dut.rst = 1;
        dut.s_valid = 0;
        dut.m_ready = 0;
        lo(); hi();
        lo(); hi();
        dut.rst = 0;
    };

    // Streams one frame; returns 0 on success. Never resets, so it can be
    // called twice in a row to prove back-to-back frames work.
    auto run_frame = [&](int H, int W, const char* tag) -> int {
        Ref ref{H, W, {}};
        ref.px.resize((size_t)H * (size_t)W * C);
        for (auto& v : ref.px) v = (int8_t)(rng() & 0xff);

        dut.width = W;
        dut.height = H;

        size_t in_i = 0, out_i = 0, total = (size_t)H * (size_t)W;
        long cycles = 0;
        bool held = false;               // m_valid seen with m_ready low
        uint8_t held_win[C * 9];         // window that must stay stable
        while (out_i < total && cycles < 4000000) {
            ++cycles;
            // drive input with random stalls
            bool drive = in_i < total && (rng() & 3) != 0;
            dut.s_valid = drive;
            if (drive)
                for (int c = 0; c < C; ++c)
                    ((uint8_t*)&dut.s_data)[c] = (uint8_t)ref.px[in_i * C + c];
            dut.m_ready = (rng() & 7) != 0;   // random downstream stalls
            lo();

            const bool in_xfer = dut.s_valid && dut.s_ready;
            const bool out_xfer = dut.m_valid && dut.m_ready;
            if (out_xfer) {
                const int y = (int)(out_i / (size_t)W), x = (int)(out_i % (size_t)W);
                for (int c = 0; c < C; ++c)
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            const int t = c * 9 + ky * 3 + kx;
                            const int8_t got = (int8_t)((uint8_t*)&dut.m_win)[t];
                            const int8_t want = ref.at(y + ky - 1, x + kx - 1, c);
                            if (got != want) {
                                std::printf("FAIL %s %dx%d pix(%d,%d) tap(c=%d,ky=%d,kx=%d): got %d want %d\n",
                                            tag, H, W, y, x, c, ky, kx, got, want);
                                return 1;
                            }
                        }
            }
            // valid/data stability: once m_valid is up with m_ready low, the
            // window must not change and valid must not drop until accepted
            if (held) {
                if (!dut.m_valid) {
                    std::printf("FAIL %s %dx%d: m_valid dropped while stalled\n", tag, H, W);
                    return 1;
                }
                for (int t = 0; t < C * 9; ++t)
                    if (((uint8_t*)&dut.m_win)[t] != held_win[t]) {
                        std::printf("FAIL %s %dx%d: m_win changed while stalled (tap %d)\n", tag, H, W, t);
                        return 1;
                    }
            }
            if (dut.m_valid && !dut.m_ready) {
                held = true;
                for (int t = 0; t < C * 9; ++t)
                    held_win[t] = ((uint8_t*)&dut.m_win)[t];
            } else {
                held = false;
            }
            hi();

            if (in_xfer) ++in_i;
            if (out_xfer) ++out_i;
        }
        if (out_i != total) {
            std::printf("FAIL %s %dx%d: only %zu/%zu windows\n", tag, H, W, out_i, total);
            return 1;
        }
        std::printf("ok %s %dx%d (%ld cycles)\n", tag, H, W, cycles);
        return 0;
    };

    // {5,512} exercises width == W_MAX, where the wr1_r flush-column guard is
    // load-bearing (mutating it corrupts ram1[0]); {1,8}/{2,2} are the
    // degenerate sizes where left and right zero pads overlap.
    const int sizes[][2] = {{1, 8}, {2, 2}, {3, 3}, {4, 7}, {16, 16},
                            {31, 5}, {64, 64}, {5, 512}};
    for (auto& sz : sizes) {
        reset();
        if (run_frame(sz[0], sz[1], "frame")) return 1;
    }

    // Two frames back to back with no reset in between: the module must not
    // carry state (stale rows, half-emitted windows) across the boundary.
    reset();
    if (run_frame(16, 16, "b2b_1")) return 1;
    if (run_frame(16, 16, "b2b_2")) return 1;
    if (run_frame(7, 9, "b2b_3")) return 1;   // dimension change, still no reset

    std::puts("PASS tb_linebuffer");
    return 0;
}
