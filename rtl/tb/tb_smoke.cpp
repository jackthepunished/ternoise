// rtl/tb/tb_smoke.cpp - proves verilate/build/run/exit-code flow
#include "Vsmoke.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char** argv) {
    VerilatedContext ctx;
    ctx.commandArgs(argc, argv);
    Vsmoke dut{&ctx};
    dut.rst = 1; dut.clk = 0;
    for (int i = 0; i < 4; ++i) { dut.clk ^= 1; dut.eval(); }
    dut.rst = 0;
    for (int i = 0; i < 20; ++i) { dut.clk ^= 1; dut.eval(); }  // 10 posedges
    if (dut.count != 10) { std::printf("FAIL smoke: count=%d want 10\n", (int)dut.count); return 1; }
    std::puts("PASS tb_smoke");
    return 0;
}
