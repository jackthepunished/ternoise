// Streams a whole golden-vector frame of 3x3 windows through Vconv3x3 and
// diffs every output pixel word against the case's expected.txt, element for
// element, zero tolerance.
//
// The windows are built here in C++ with QUANT_SPEC zero padding (the same
// reference logic tb_linebuffer checks the line buffer against), so this
// testbench isolates the conv layer: weight decode, ternary adder tree, bias,
// requant, relu, channel packing.
//
// Compile-time TB_C_IN / TB_C_OUT / TB_RELU must match the -G parameters the
// module was verilated with; they are re-checked against the case's meta.txt
// at run time so a mismatched make rule fails loudly instead of silently
// testing the wrong shape.
//
// Handshake sampling follows rtl/tb/tb_linebuffer.cpp: drive, evaluate with
// clk low so the combinational logic settles, sample valid/ready/data at that
// point, only then take the posedge.
#include "Vconv3x3.h"
#include "verilated.h"
#include "../../sim/tensor.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef TB_C_IN
#error "TB_C_IN must be defined on the verilator -CFLAGS line"
#endif
#ifndef TB_C_OUT
#error "TB_C_OUT must be defined on the verilator -CFLAGS line"
#endif
#ifndef TB_RELU
#error "TB_RELU must be defined on the verilator -CFLAGS line"
#endif

static constexpr int C_IN  = TB_C_IN;
static constexpr int C_OUT = TB_C_OUT;
static constexpr int NTAP  = C_IN * 9;

// $clog2: ceil(log2(n)), the SystemVerilog definition.
static int clog2(long long n) {
    int r = 0;
    while ((1LL << r) < n) ++r;
    return r;
}

static std::string plusarg(int argc, char** argv, const char* key) {
    const std::string pre = std::string("+") + key + "=";
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], pre.c_str(), pre.size()) == 0)
            return std::string(argv[i] + pre.size());
    std::printf("FAIL: missing plusarg +%s=<value>\n", key);
    std::exit(2);
}

// meta.txt is line-oriented "key rest-of-line" (tools/case_to_mem.py).
static std::string meta_get(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) { std::printf("FAIL: cannot open %s\n", path.c_str()); std::exit(2); }
    std::string line;
    while (std::getline(f, line)) {
        const size_t sp = line.find(' ');
        if (sp != std::string::npos && line.substr(0, sp) == key)
            return line.substr(sp + 1);
    }
    std::printf("FAIL: key %s missing from %s\n", key.c_str(), path.c_str());
    std::exit(2);
}

int main(int argc, char** argv) {
    VerilatedContext ctx;
    ctx.commandArgs(argc, argv);   // must precede the first eval: the DUT's
                                   // $readmemh initial block reads +w1=/+b1=

    const std::string case_dir = plusarg(argc, argv, "case");
    const std::string meta     = plusarg(argc, argv, "meta");
    const char* tag            = case_dir.c_str();

    // ---- case parameters -------------------------------------------------
    int meta_ci = 0, meta_co = 0;
    { std::istringstream s(meta_get(meta, "chans")); s >> meta_ci >> meta_co; }
    const int shift = std::stoi(meta_get(meta, "shift"));
    const int relu  = std::stoi(meta_get(meta, "relu"));
    const int H     = std::stoi(meta_get(meta, "height"));
    const int W     = std::stoi(meta_get(meta, "width"));
    if (meta_ci != C_IN || meta_co != C_OUT || relu != TB_RELU) {
        std::printf("FAIL %s: built for %d->%d relu=%d, case is %d->%d relu=%d\n",
                    tag, C_IN, C_OUT, TB_RELU, meta_ci, meta_co, relu);
        return 1;
    }

    const Tensor x   = load_tensor(case_dir + "/input.txt");
    const Tensor exp = load_tensor(case_dir + "/expected.txt");
    const Tensor b   = load_tensor(case_dir + "/bias.txt");
    if (x.dims[0] != C_IN || x.dims[1] != H || x.dims[2] != W ||
        exp.dims[0] != C_OUT || exp.dims[1] != H || exp.dims[2] != W) {
        std::printf("FAIL %s: tensor dims disagree with meta.txt\n", tag);
        return 1;
    }

    // Global Constraints: ACC_W = clog2(9*C_IN*128) + 3 and every loaded bias
    // must leave room for a full-magnitude accumulator on top of it.
    const int     ACC_W  = clog2(9LL * C_IN * 128) + 3;
    const int64_t MAXACC = 9LL * C_IN * 128;
    const int64_t LIMIT  = (1LL << (ACC_W - 1)) - MAXACC;
    for (size_t i = 0; i < b.data.size(); ++i) {
        const int64_t v = b.data[i] < 0 ? -b.data[i] : b.data[i];
        if (v >= LIMIT) {
            std::printf("FAIL %s: bias[%zu]=%lld does not fit ACC_W=%d (limit %lld)\n",
                        tag, i, (long long)b.data[i], ACC_W, (long long)LIMIT);
            return 1;
        }
    }

    Vconv3x3 dut{&ctx};
    std::mt19937 rng(1058);

    auto lo = [&] { dut.clk = 0; dut.eval(); };   // settle combinational logic
    auto hi = [&] { dut.clk = 1; dut.eval(); };   // posedge

    const size_t total = (size_t)H * (size_t)W;

    // Window for raster pixel p, QUANT_SPEC zero padding, tap t = ci*9+ky*3+kx.
    auto fill_window = [&](size_t p) {
        uint8_t* w = (uint8_t*)&dut.s_win;
        const int y = (int)(p / (size_t)W), xx = (int)(p % (size_t)W);
        for (int ci = 0; ci < C_IN; ++ci)
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    const int iy = y + ky - 1, ix = xx + kx - 1;
                    const bool in = iy >= 0 && iy < H && ix >= 0 && ix < W;
                    const int64_t v =
                        in ? x.data[((size_t)ci * H + iy) * W + ix] : 0;
                    w[ci * 9 + ky * 3 + kx] = (uint8_t)(int8_t)v;
                }
    };
    // Garbage on the window bus whenever s_valid is low: a DUT that latches
    // without a handshake gets caught instead of silently reading stale data.
    auto scramble_window = [&] {
        uint8_t* w = (uint8_t*)&dut.s_win;
        for (int t = 0; t < NTAP; ++t) w[t] = (uint8_t)(rng() & 0xff);
    };

    // Streams the whole frame. stalls=false measures steady-state throughput.
    auto run = [&](bool stalls, const char* mode, long* out_cycles) -> int {
        dut.rst = 1;
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.shift = (uint8_t)shift;
        lo(); hi();
        lo(); hi();
        dut.rst = 0;

        size_t in_i = 0, out_i = 0;
        long cycles = 0;
        bool held = false;
        uint8_t held_d[C_OUT];
        while (out_i < total && cycles < 20000000) {
            ++cycles;
            const bool drive = in_i < total && (!stalls || (rng() & 3) != 0);
            dut.s_valid = drive;
            if (drive) fill_window(in_i); else scramble_window();
            dut.m_ready = stalls ? ((rng() & 7) != 0) : 1;
            lo();

            const bool in_xfer  = dut.s_valid && dut.s_ready;
            const bool out_xfer = dut.m_valid && dut.m_ready;
            const uint8_t* md = (const uint8_t*)&dut.m_data;

            if (out_xfer) {
                const int y = (int)(out_i / (size_t)W), xx = (int)(out_i % (size_t)W);
                for (int co = 0; co < C_OUT; ++co) {
                    const int got = (int)(int8_t)md[co];
                    const int want = (int)exp.data[((size_t)co * H + y) * W + xx];
                    if (got != want) {
                        std::printf("FAIL %s [%s] pix(%d,%d) co=%d: got %d want %d\n",
                                    tag, mode, y, xx, co, got, want);
                        return 1;
                    }
                }
            }
            // Output stability: once m_valid is up with m_ready low, m_data
            // must not move and m_valid must not drop until it is accepted.
            if (held) {
                if (!dut.m_valid) {
                    std::printf("FAIL %s [%s]: m_valid dropped while stalled\n", tag, mode);
                    return 1;
                }
                for (int co = 0; co < C_OUT; ++co)
                    if (md[co] != held_d[co]) {
                        std::printf("FAIL %s [%s]: m_data changed while stalled (co %d)\n",
                                    tag, mode, co);
                        return 1;
                    }
            }
            if (dut.m_valid && !dut.m_ready) {
                held = true;
                for (int co = 0; co < C_OUT; ++co) held_d[co] = md[co];
            } else {
                held = false;
            }
            hi();

            if (in_xfer) ++in_i;
            if (out_xfer) ++out_i;
        }
        if (out_i != total) {
            std::printf("FAIL %s [%s]: only %zu/%zu pixels out\n", tag, mode, out_i, total);
            return 1;
        }
        *out_cycles = cycles;
        return 0;
    };

    long c_stall = 0, c_free = 0;
    if (run(true, "stalls", &c_stall)) return 1;
    if (run(false, "free", &c_free)) return 1;

    // Integer math only: hundredths of a cycle per pixel.
    const long hundredths = (c_free * 100 + (long)total / 2) / (long)total;
    std::printf("PASS tb_conv3x3 %s (%d->%d relu=%d shift=%d %dx%d) "
                "cycles=%ld pixels=%zu cyc_per_px=%ld.%02ld (stalled run %ld cycles)\n",
                tag, C_IN, C_OUT, relu, shift, H, W,
                c_free, total, hundredths / 100, hundredths % 100, c_stall);
    return 0;
}
