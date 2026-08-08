// QUANT_SPEC section 4: requant(acc, s) = clamp(rha(acc, s), -128, 127).
//
// Round-half-away-from-zero without a divider and without a multiplier: take
// the magnitude, add half an LSB of the target scale, shift right logically,
// then restore the sign and clamp to int8. This is why the spec pins requant
// scales to powers of two - an arbitrary fixed-point scale would smuggle a
// multiplier back into the datapath.
//
// Purely combinational; the caller registers the result.
module requant #(parameter int ACC_W = 18) (
    input  logic signed [ACC_W-1:0] acc,
    input  logic [3:0]              shift,
    output logic signed [7:0]       q
);
    // One headroom bit for the +half; magnitude path realizes round-half-away
    // with a logical shift.
    logic signed [ACC_W:0] acc_x;
    logic [ACC_W:0]        mag, half, shifted;
    logic signed [ACC_W:0] rounded;
    always_comb begin
        acc_x   = {acc[ACC_W-1], acc};
        mag     = acc_x[ACC_W] ? unsigned'(-acc_x) : unsigned'(acc_x);
        half    = '0;
        if (shift != 0) half = ({{ACC_W{1'b0}}, 1'b1} << (shift - 4'd1));
        shifted = (mag + half) >> shift;
        rounded = acc_x[ACC_W] ? -signed'(shifted) : signed'(shifted);
        if (rounded > $signed({{(ACC_W-7){1'b0}}, 8'sd127}))       q = 8'sd127;
        else if (rounded < -$signed({{(ACC_W-7){1'b0}}, 8'sd128})) q = -8'sd128;
        else                                                        q = rounded[7:0];
    end
endmodule
