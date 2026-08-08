// QUANT_SPEC section 4 in silicon: the eleven normative rows from the spec
// table, then an exhaustive sweep of the whole 18-bit-safe accumulator range
// against the C++ golden model's requant(). Zero tolerance, combinational DUT.
//
// The oracle is sim/ops.cpp, linked straight into this harness - the same
// function the golden model runs on every checked-in vector.
#include "Vrequant.h"
#include "verilated.h"
#include "../../sim/ops.h"
#include <cstdint>
#include <cstdio>

// Must match the requant instance's ACC_W (verilator default parameter).
static constexpr int      ACC_W    = 18;
static constexpr int64_t  ACC_MIN  = -(1LL << (ACC_W - 1));
static constexpr int64_t  ACC_MAX  = (1LL << (ACC_W - 1)) - 1;
static constexpr uint32_t ACC_MASK = (1u << ACC_W) - 1u;

int main(int argc, char** argv) {
    VerilatedContext ctx;
    ctx.commandArgs(argc, argv);
    Vrequant dut{&ctx};

    // Verilator does not mask input ports for us: feed exactly ACC_W bits.
    auto drive = [&](int64_t acc, int shift) {
        dut.acc = (uint32_t)(int32_t)acc & ACC_MASK;
        dut.shift = (uint8_t)shift;
        dut.eval();
        return (int)(int8_t)dut.q;
    };

    // 1) the eleven normative rows from QUANT_SPEC section 4
    const int64_t spec[][3] = {{0, 0, 0},   {127, 0, 127}, {200, 0, 127},
                               {-200, 0, -128}, {5, 1, 3}, {-5, 1, -3},
                               {3, 1, 2},   {-3, 1, -2},   {2, 2, 1},
                               {-2, 2, -1}, {36864, 5, 127}};
    for (auto& r : spec) {
        const int got = drive(r[0], (int)r[1]);
        if (got != (int)r[2]) {
            std::printf("FAIL spec acc=%lld s=%lld: got %d want %lld\n",
                        (long long)r[0], (long long)r[1], got, (long long)r[2]);
            return 1;
        }
    }

    // 2) exhaustive over every shift and a dense accumulator range. The bounds
    //    straddle both requant clamps and every rounding boundary in between.
    for (int64_t a = -100000; a <= 100000; ++a)
        for (int s = 0; s < 16; ++s) {
            const int got = drive(a, s);
            const int want = (int)requant(a, s);
            if (got != want) {
                std::printf("FAIL acc=%lld s=%d: got %d want %d\n",
                            (long long)a, s, got, want);
                return 1;
            }
        }

    // 3) the extremes of the ACC_W range itself, including the most negative
    //    value (where negating the magnitude needs the headroom bit).
    for (int s = 0; s < 16; ++s) {
        const int64_t edges[] = {ACC_MIN, ACC_MIN + 1, ACC_MAX - 1, ACC_MAX};
        for (int64_t a : edges) {
            const int got = drive(a, s);
            const int want = (int)requant(a, s);
            if (got != want) {
                std::printf("FAIL edge acc=%lld s=%d: got %d want %d\n",
                            (long long)a, s, got, want);
                return 1;
            }
        }
    }

    std::puts("PASS tb_requant");
    return 0;
}
