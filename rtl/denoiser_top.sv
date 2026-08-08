// The whole v1 network as one streaming block: 3x3 ternary convolutions
// 7 -> 16 -> 16 -> 16 -> 16 -> 3, relu after layers 1-4, and the clamped
// residual add of the noisy RGB (QUANT_SPEC sections 3-5).
//
// Shape:
//
//   s_data(7ch) --+--> lb1 --> conv1 --> lb2 --> conv2 --> ... --> conv5 --+
//                 |                                        (3ch residual)  |
//                 +--> delay_fifo (RGB) ---------------------------------> +
//                                                                          |
//                                             clamp(x + r, -128, 127) --> m_data
//
// Each of the five stages is a line buffer feeding an output-channel-serial
// conv layer, so the chain's throughput is set by the widest layer: 16 output
// channels plus a handshake cycle, i.e. 17 cycles per pixel in steady state,
// after a fill of roughly five rows.
//
// Alignment. The FIFO is pushed on every accepted input pixel and popped on
// every layer-5 output pixel; both streams stay in raster order and neither
// drops or duplicates, so entry k is always pixel k. Depth 8192 covers the
// pixels in flight - about five rows plus a column, at most 5 * (512 + 1) at
// the line buffers' W_MAX - with room to spare, which is also what rules out
// a deadlock (the FIFO can never fill while the chain still needs input).
//
// Handshakes. The input is a fork: a pixel is consumed only when the first
// line buffer and the FIFO can both take it. The output is a join: a result
// leaves only when layer 5 and the FIFO both have data. Neither module's
// ready depends on the other's valid, so no combinational loop closes. Every
// stage's s_ready is combinational from its downstream m_ready, so one ready
// path runs from the output back to the input; if synthesis timing ever
// complains the fix is a registered-ready (skid) stage between layers.
//
// No multipliers: the residual adder is a 9-bit signed add against constant
// clamp bounds, and everything below it is line buffers, ternary taps and
// shifts. The only *, / and % characters in this file are inside localparams
// and port widths, evaluated at elaboration.
module denoiser_top (
    input  logic         clk,
    input  logic         rst,
    input  logic [9:0]   width,
    input  logic [9:0]   height,
    input  logic [19:0]  shifts,      // 5 x 4-bit, layer i in bits [i*4 +: 4]
    input  logic         s_valid,     // 7-channel noisy pixel word
    output logic         s_ready,
    input  logic [55:0]  s_data,
    output logic         m_valid,     // 3-channel denoised RGB pixel word
    input  logic         m_ready,
    output logic [23:0]  m_data
);

    // Channel counts, QUANT_SPEC section 5. Pixel words are C*8 bits, window
    // buses C*9*8 bits.
    localparam int C0 = 7;
    localparam int C1 = 16;
    localparam int C2 = 16;
    localparam int C3 = 16;
    localparam int C4 = 16;
    localparam int C5 = 3;

    // ---- inter-stage buses -----------------------------------------------
    logic                 lb1_m_valid, lb1_m_ready;
    logic [C0*9*8-1:0]    lb1_win;
    logic                 c1_valid, c1_ready;
    logic [C1*8-1:0]      c1_data;

    logic                 lb2_s_valid, lb2_s_ready;
    logic                 lb2_m_valid, lb2_m_ready;
    logic [C1*9*8-1:0]    lb2_win;
    logic                 c2_valid, c2_ready;
    logic [C2*8-1:0]      c2_data;

    logic                 lb3_m_valid, lb3_m_ready;
    logic [C2*9*8-1:0]    lb3_win;
    logic                 c3_valid, c3_ready;
    logic [C3*8-1:0]      c3_data;

    logic                 lb4_m_valid, lb4_m_ready;
    logic [C3*9*8-1:0]    lb4_win;
    logic                 c4_valid, c4_ready;
    logic [C4*8-1:0]      c4_data;

    logic                 lb5_m_valid, lb5_m_ready;
    logic [C4*9*8-1:0]    lb5_win;
    logic                 c5_valid, c5_ready;
    logic [C5*8-1:0]      c5_data;

    // ---- input fork: line buffer 1 and the residual FIFO ------------------
    logic lb1_s_valid, lb1_s_ready;
    logic fifo_s_valid, fifo_s_ready;
    logic fifo_m_valid, fifo_m_ready;
    logic [C5*8-1:0] fifo_data;

    // Both sinks must take the pixel in the same cycle. Neither ready looks at
    // a valid, so gating each valid with the other ready is loop-free.
    assign lb1_s_valid  = s_valid && fifo_s_ready;
    assign fifo_s_valid = s_valid && lb1_s_ready;
    assign s_ready      = lb1_s_ready && fifo_s_ready;

    delay_fifo #(.WIDTH(C5*8), .DEPTH_LOG2(13)) u_fifo (
        .clk(clk), .rst(rst),
        .s_valid(fifo_s_valid), .s_ready(fifo_s_ready),
        .s_data(s_data[C5*8-1:0]),          // channels 0-2 = noisy RGB
        .m_valid(fifo_m_valid), .m_ready(fifo_m_ready), .m_data(fifo_data)
    );

    // ---- layer 1: 7 -> 16 -------------------------------------------------
    linebuffer #(.C(C0)) u_lb1 (
        .clk(clk), .rst(rst), .width(width), .height(height),
        .s_valid(lb1_s_valid), .s_ready(lb1_s_ready), .s_data(s_data),
        .m_valid(lb1_m_valid), .m_ready(lb1_m_ready), .m_win(lb1_win)
    );

    conv3x3 #(
        .C_IN(C0), .C_OUT(C1), .RELU(1'b1),
        .WKEY("w1"), .BKEY("b1"),
        .WFILE("synth/case04/w1.mem"), .BFILE("synth/case04/b1.mem")
    ) u_conv1 (
        .clk(clk), .rst(rst), .shift(shifts[3:0]),
        .s_valid(lb1_m_valid), .s_ready(lb1_m_ready), .s_win(lb1_win),
        .m_valid(c1_valid), .m_ready(c1_ready), .m_data(c1_data)
    );

    // ---- layer 2: 16 -> 16 ------------------------------------------------
    assign lb2_s_valid = c1_valid;
    assign c1_ready    = lb2_s_ready;

    linebuffer #(.C(C1)) u_lb2 (
        .clk(clk), .rst(rst), .width(width), .height(height),
        .s_valid(lb2_s_valid), .s_ready(lb2_s_ready), .s_data(c1_data),
        .m_valid(lb2_m_valid), .m_ready(lb2_m_ready), .m_win(lb2_win)
    );

    conv3x3 #(
        .C_IN(C1), .C_OUT(C2), .RELU(1'b1),
        .WKEY("w2"), .BKEY("b2"),
        .WFILE("synth/case04/w2.mem"), .BFILE("synth/case04/b2.mem")
    ) u_conv2 (
        .clk(clk), .rst(rst), .shift(shifts[7:4]),
        .s_valid(lb2_m_valid), .s_ready(lb2_m_ready), .s_win(lb2_win),
        .m_valid(c2_valid), .m_ready(c2_ready), .m_data(c2_data)
    );

    // ---- layer 3: 16 -> 16 ------------------------------------------------
    linebuffer #(.C(C2)) u_lb3 (
        .clk(clk), .rst(rst), .width(width), .height(height),
        .s_valid(c2_valid), .s_ready(c2_ready), .s_data(c2_data),
        .m_valid(lb3_m_valid), .m_ready(lb3_m_ready), .m_win(lb3_win)
    );

    conv3x3 #(
        .C_IN(C2), .C_OUT(C3), .RELU(1'b1),
        .WKEY("w3"), .BKEY("b3"),
        .WFILE("synth/case04/w3.mem"), .BFILE("synth/case04/b3.mem")
    ) u_conv3 (
        .clk(clk), .rst(rst), .shift(shifts[11:8]),
        .s_valid(lb3_m_valid), .s_ready(lb3_m_ready), .s_win(lb3_win),
        .m_valid(c3_valid), .m_ready(c3_ready), .m_data(c3_data)
    );

    // ---- layer 4: 16 -> 16 ------------------------------------------------
    linebuffer #(.C(C3)) u_lb4 (
        .clk(clk), .rst(rst), .width(width), .height(height),
        .s_valid(c3_valid), .s_ready(c3_ready), .s_data(c3_data),
        .m_valid(lb4_m_valid), .m_ready(lb4_m_ready), .m_win(lb4_win)
    );

    conv3x3 #(
        .C_IN(C3), .C_OUT(C4), .RELU(1'b1),
        .WKEY("w4"), .BKEY("b4"),
        .WFILE("synth/case04/w4.mem"), .BFILE("synth/case04/b4.mem")
    ) u_conv4 (
        .clk(clk), .rst(rst), .shift(shifts[15:12]),
        .s_valid(lb4_m_valid), .s_ready(lb4_m_ready), .s_win(lb4_win),
        .m_valid(c4_valid), .m_ready(c4_ready), .m_data(c4_data)
    );

    // ---- layer 5: 16 -> 3, no relu (QUANT_SPEC section 5) -----------------
    linebuffer #(.C(C4)) u_lb5 (
        .clk(clk), .rst(rst), .width(width), .height(height),
        .s_valid(c4_valid), .s_ready(c4_ready), .s_data(c4_data),
        .m_valid(lb5_m_valid), .m_ready(lb5_m_ready), .m_win(lb5_win)
    );

    conv3x3 #(
        .C_IN(C4), .C_OUT(C5), .RELU(1'b0),
        .WKEY("w5"), .BKEY("b5"),
        .WFILE("synth/case04/w5.mem"), .BFILE("synth/case04/b5.mem")
    ) u_conv5 (
        .clk(clk), .rst(rst), .shift(shifts[19:16]),
        .s_valid(lb5_m_valid), .s_ready(lb5_m_ready), .s_win(lb5_win),
        .m_valid(c5_valid), .m_ready(c5_ready), .m_data(c5_data)
    );

    // ---- output join + residual head --------------------------------------
    // A result leaves only when the residual and the delayed input are both
    // there; each side is consumed only in that same cycle, so nothing is
    // dropped and the two streams stay in step.
    assign m_valid     = c5_valid && fifo_m_valid;
    assign c5_ready    = m_ready && fifo_m_valid;
    assign fifo_m_ready = m_ready && c5_valid;

    // out_rgb[ch] = clamp(x_in[ch] + y5[ch], -128, 127). The sum of two int8
    // values spans -256 .. 254, so a 9-bit signed adder is exact and the
    // clamp is two comparisons against constants.
    genvar gch;
    generate
        for (gch = 0; gch < C5; gch++) begin : g_res
            localparam int CB = gch << 3;
            logic signed [7:0] xin, res;
            logic signed [8:0] sum;
            assign xin = fifo_data[CB +: 8];
            assign res = c5_data[CB +: 8];
            assign sum = $signed({xin[7], xin}) + $signed({res[7], res});
            assign m_data[CB +: 8] = (sum > 9'sd127)  ?  8'sd127 :
                                     (sum < -9'sd128) ? -8'sd128 : sum[7:0];
        end
    endgenerate

endmodule
