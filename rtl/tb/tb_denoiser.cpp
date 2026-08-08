// Streams a whole golden-vector frame through Vdenoiser_top - the full v1
// network, 7->16->16->16->16->3 plus the residual head - and diffs every
// output pixel word against the case's expected.txt, element for element,
// zero tolerance.
//
// This is the end-to-end bit-exactness claim of M4: five line buffers, five
// output-channel-serial ternary conv layers, five requant/relu stages, the
// noisy-RGB delay FIFO and the clamped residual add, all driven through a
// valid/ready stream with randomized stalls on both sides.
//
// RELU=0 oracle (Task 4 review finding): case03 saturated uniformly to 127,
// so the conv testbench could not tell a relu-less layer from a relu'd one.
// Layer 5 here is the first real oracle for RELU=0, but only if the case's
// layer-5 residual actually goes negative somewhere. Wherever the denoised
// output sits below the noisy input the residual must have been negative
// (the clamp only ever pulls values up towards -128, never below the input
// unless the residual itself was negative), so counting those pixels is a
// sound lower bound on the negative-residual count. The count is printed and
// a case with none of them is rejected as a useless oracle.
//
// Handshake sampling follows rtl/tb/tb_linebuffer.cpp: drive, evaluate with
// clk low so the combinational logic settles, sample valid/ready/data at that
// point, only then take the posedge.
#include "Vdenoiser_top.h"
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

static constexpr int NLAYER = 5;
static constexpr int C_IN0  = 7;   // noisy RGB + normals + depth
static constexpr int C_OUT5 = 3;   // RGB residual
static const int CHANS[NLAYER + 1] = {7, 16, 16, 16, 16, 3};

// $clog2: ceil(log2(n)), the SystemVerilog definition.
static int clog2(long long n) {
    int r = 0;
    while ((1LL << r) < n) ++r;
    return r;
}

static bool plusarg_opt(int argc, char** argv, const char* key, std::string* out) {
    const std::string pre = std::string("+") + key + "=";
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], pre.c_str(), pre.size()) == 0) {
            *out = std::string(argv[i] + pre.size());
            return true;
        }
    return false;
}

static std::string plusarg(int argc, char** argv, const char* key) {
    std::string v;
    if (plusarg_opt(argc, argv, key, &v)) return v;
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
    ctx.commandArgs(argc, argv);   // must precede the first eval: each conv
                                   // instance's $readmemh initial block reads
                                   // its +w<i>=/+b<i>= plusarg

    const std::string case_dir = plusarg(argc, argv, "case");
    const std::string meta     = plusarg(argc, argv, "meta");
    std::string tag = case_dir;
    plusarg_opt(argc, argv, "tag", &tag);
    const char* t = tag.c_str();

    // ---- case parameters -------------------------------------------------
    if (meta_get(meta, "type") != "network") {
        std::printf("FAIL %s: meta type is '%s', want 'network'\n",
                    t, meta_get(meta, "type").c_str());
        return 1;
    }
    if (meta_get(meta, "layers") != "5") {
        std::printf("FAIL %s: meta layers is '%s', want '5'\n",
                    t, meta_get(meta, "layers").c_str());
        return 1;
    }
    {
        std::istringstream s(meta_get(meta, "chans"));
        for (int i = 0; i <= NLAYER; ++i) {
            int v = -1;
            s >> v;
            if (v != CHANS[i]) {
                std::printf("FAIL %s: chans[%d]=%d, denoiser_top is wired for %d\n",
                            t, i, v, CHANS[i]);
                return 1;
            }
        }
    }
    int shift[NLAYER] = {0, 0, 0, 0, 0};
    {
        std::istringstream s(meta_get(meta, "shift"));
        for (int i = 0; i < NLAYER; ++i) s >> shift[i];
    }
    {
        std::istringstream s(meta_get(meta, "relu"));
        for (int i = 0; i < NLAYER; ++i) {
            int v = -1;
            s >> v;
            const int want = (i < NLAYER - 1) ? 1 : 0;   // QUANT_SPEC section 5
            if (v != want) {
                std::printf("FAIL %s: relu[%d]=%d, denoiser_top is wired for %d\n",
                            t, i, v, want);
                return 1;
            }
        }
    }
    const int H = std::stoi(meta_get(meta, "height"));
    const int W = std::stoi(meta_get(meta, "width"));

    uint32_t shift_word = 0;   // 5 x 4 bits, layer i in bits [i*4 +: 4]
    for (int i = 0; i < NLAYER; ++i) {
        if (shift[i] < 0 || shift[i] > 15) {
            std::printf("FAIL %s: shift[%d]=%d outside QUANT_SPEC range [0,15]\n",
                        t, i, shift[i]);
            return 1;
        }
        shift_word |= (uint32_t)shift[i] << (i * 4);
    }

    const Tensor x   = load_tensor(case_dir + "/input.txt");
    const Tensor exp = load_tensor(case_dir + "/expected.txt");
    if (x.dims.size() != 3 || exp.dims.size() != 3 ||
        x.dims[0] != C_IN0 || x.dims[1] != H || x.dims[2] != W ||
        exp.dims[0] != C_OUT5 || exp.dims[1] != H || exp.dims[2] != W) {
        std::printf("FAIL %s: tensor dims disagree with meta.txt (%dx%d)\n", t, H, W);
        return 1;
    }

    // Global Constraints: ACC_W = clog2(9*C_IN*128) + 3 per layer, and every
    // loaded bias must leave room for a full-magnitude accumulator on top.
    for (int i = 1; i <= NLAYER; ++i) {
        const Tensor b = load_tensor(case_dir + "/b" + std::to_string(i) + ".txt");
        const int     c_in   = CHANS[i - 1];
        const int     ACC_W  = clog2(9LL * c_in * 128) + 3;
        const int64_t MAXACC = 9LL * c_in * 128;
        const int64_t LIMIT  = (1LL << (ACC_W - 1)) - MAXACC;
        if ((int)b.data.size() != CHANS[i]) {
            std::printf("FAIL %s: b%d.txt has %zu entries, want %d\n",
                        t, i, b.data.size(), CHANS[i]);
            return 1;
        }
        for (size_t k = 0; k < b.data.size(); ++k) {
            const int64_t v = b.data[k] < 0 ? -b.data[k] : b.data[k];
            if (v >= LIMIT) {
                std::printf("FAIL %s: b%d[%zu]=%lld does not fit ACC_W=%d (limit %lld)\n",
                            t, i, k, (long long)b.data[k], ACC_W, (long long)LIMIT);
                return 1;
            }
        }
    }

    // ---- RELU=0 oracle ---------------------------------------------------
    const size_t plane = (size_t)H * (size_t)W;
    long neg_res = 0;
    for (int c = 0; c < C_OUT5; ++c)
        for (size_t i = 0; i < plane; ++i)
            if (exp.data[(size_t)c * plane + i] < x.data[(size_t)c * plane + i])
                ++neg_res;
    if (neg_res == 0) {
        std::printf("FAIL %s: no output pixel below its noisy input - layer 5's "
                    "residual is nowhere negative, so this case cannot "
                    "discriminate the no-relu path\n", t);
        return 1;
    }

    Vdenoiser_top dut{&ctx};
    std::mt19937 rng(1058);

    auto lo = [&] { dut.clk = 0; dut.eval(); };   // settle combinational logic
    auto hi = [&] { dut.clk = 1; dut.eval(); };   // posedge

    auto set_input = [&](size_t p) {
        const size_t y = p / (size_t)W, xx = p % (size_t)W;
        uint64_t v = 0;
        for (int c = 0; c < C_IN0; ++c) {
            const uint8_t byte =
                (uint8_t)(int8_t)x.data[((size_t)c * (size_t)H + y) * (size_t)W + xx];
            v |= (uint64_t)byte << (c * 8);
        }
        dut.s_data = v;
    };
    // Garbage on the input bus whenever s_valid is low: a DUT that latches
    // without a handshake gets caught instead of silently reading stale data.
    auto scramble_input = [&] {
        const uint64_t v = ((uint64_t)rng() << 32) | (uint64_t)rng();
        dut.s_data = v & 0x00FFFFFFFFFFFFFFULL;   // 56 bits
    };

    // Streams the whole frame. stalls=false measures steady-state throughput;
    // reset=false proves nothing survives a frame boundary.
    auto run = [&](bool stalls, bool reset, const char* mode,
                   long* out_cycles) -> int {
        dut.width  = (uint16_t)W;
        dut.height = (uint16_t)H;
        dut.shifts = shift_word;
        if (reset) {
            dut.rst = 1;
            dut.s_valid = 0;
            dut.m_ready = 0;
            lo(); hi();
            lo(); hi();
            dut.rst = 0;
        }

        size_t in_i = 0, out_i = 0;
        long cycles = 0;
        bool held = false;
        uint32_t held_d = 0;
        while (out_i < plane && cycles < 50000000) {
            ++cycles;
            const bool drive = in_i < plane && (!stalls || (rng() & 3) != 0);
            dut.s_valid = drive;
            if (drive) set_input(in_i); else scramble_input();
            dut.m_ready = stalls ? ((rng() & 7) != 0) : 1;
            lo();

            const bool in_xfer  = dut.s_valid && dut.s_ready;
            const bool out_xfer = dut.m_valid && dut.m_ready;
            const uint32_t md   = (uint32_t)dut.m_data;

            if (out_xfer) {
                const int y = (int)(out_i / (size_t)W), xx = (int)(out_i % (size_t)W);
                for (int c = 0; c < C_OUT5; ++c) {
                    const int got  = (int)(int8_t)((md >> (c * 8)) & 0xff);
                    const int want = (int)exp.data[(size_t)c * plane + out_i];
                    if (got != want) {
                        std::printf("FAIL %s [%s] pix(%d,%d) ch=%d: got %d want %d\n",
                                    t, mode, y, xx, c, got, want);
                        return 1;
                    }
                }
            }
            // Output stability: once m_valid is up with m_ready low, m_data
            // must not move and m_valid must not drop until it is accepted.
            if (held) {
                if (!dut.m_valid) {
                    std::printf("FAIL %s [%s]: m_valid dropped while stalled\n", t, mode);
                    return 1;
                }
                if (md != held_d) {
                    std::printf("FAIL %s [%s]: m_data changed while stalled\n", t, mode);
                    return 1;
                }
            }
            if (dut.m_valid && !dut.m_ready) {
                held = true;
                held_d = md;
            } else {
                held = false;
            }
            hi();

            if (in_xfer) ++in_i;
            if (out_xfer) ++out_i;
        }
        if (out_i != plane) {
            std::printf("FAIL %s [%s]: only %zu/%zu pixels out\n", t, mode, out_i, plane);
            return 1;
        }
        *out_cycles = cycles;
        return 0;
    };

    long c_stall = 0, c_free = 0, c_b2b = 0;
    if (run(true,  true,  "stalls", &c_stall)) return 1;
    if (run(false, true,  "free",   &c_free))  return 1;
    if (run(false, false, "b2b",    &c_b2b))   return 1;   // no reset in between

    // Integer math only: hundredths of a cycle per pixel.
    const long hundredths = (c_free * 100 + (long)plane / 2) / (long)plane;
    std::printf("PASS tb_denoiser %s (7->16->16->16->16->3 shifts=%d,%d,%d,%d,%d "
                "%dx%d) cycles=%ld pixels=%zu cyc_per_px=%ld.%02ld "
                "neg_residual=%ld/%zu (stalled run %ld cycles, b2b %ld)\n",
                t, shift[0], shift[1], shift[2], shift[3], shift[4], H, W,
                c_free, plane, hundredths / 100, hundredths % 100,
                neg_res, plane * C_OUT5, c_stall, c_b2b);
    return 0;
}
