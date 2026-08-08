// The ternary processing element - the whole thesis of the project.
//
// A weight is one of {-1, 0, +1}, delivered as the QUANT_SPEC section 6 2-bit
// code (00 = 0, 01 = +1, 10 = -1; 11 is illegal and rejected at load time).
// So a "multiply-accumulate" is an add, a subtract, or nothing at all: there
// is no multiplier in this file, and there is no multiplier anywhere below it.
//
// Output is 10 bits because -(-128) = +128 does not fit in 8, and the caller
// sign-extends into its per-layer accumulator.
module pe (
    input  logic [1:0]        code,  // 00=0, 01=+x, 10=-x (11 illegal, checked at load)
    input  logic signed [7:0] x,
    output logic signed [9:0] y
);
    always_comb unique case (code)
        2'b01:   y = {{2{x[7]}}, x};
        2'b10:   y = -{{2{x[7]}}, x};
        default: y = '0;
    endcase
endmodule
