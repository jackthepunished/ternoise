// One 3x3 ternary convolution layer, output-channel-serial.
//
// Takes a 3x3 window (all C_IN channels, from linebuffer) and produces one
// pixel word carrying all C_OUT output channels. Internally it spends one
// cycle per output channel: the whole 9*C_IN-tap ternary dot product for a
// single co happens combinationally in one cycle, so the layer needs
// C_OUT + 1 cycles per pixel and exactly 9*C_IN adder inputs of hardware -
// none of which is a multiplier (see pe.sv).
//
// Datapath operator rule: the only *, / and % characters in this file are in
// localparams, port widths and $clog2 arguments, all of which are evaluated at
// elaboration and never become hardware. Every expression that computes a
// signal uses adds, shifts, masks and muxes. In particular the per-output-
// channel weight base is an accumulating register (base <= base + NTAP_C, a
// constant add) rather than co * C_IN * 9, which would be a runtime multiply.
//
// Control (three observable states, two flops):
//   idle  (busy=0)  s_ready high once any pending output has been accepted;
//                   a window handshake latches s_win, clears co/base, sets busy
//   busy  (busy=1)  one output channel per cycle, co = 0 .. C_OUT-1; each cycle
//                   writes obyte[co]; the last one clears busy and raises
//                   m_valid
//   held  (m_valid) m_data stable until m_ready; the same cycle it is accepted
//                   can also accept the next window, so the steady-state period
//                   is C_OUT + 1 cycles per pixel
//
// QUANT_SPEC: section 3 (conv + int32 bias), section 4 (requant), section 5
// (relu after requant), section 6 (2-bit weight codes, 11 illegal).
module conv3x3 #(
    parameter int    C_IN   = 16,
    parameter int    C_OUT  = 16,
    parameter bit    RELU   = 1'b1,
    parameter        WKEY   = "w1",   // plusarg key for the weight .mem path
    parameter        BKEY   = "b1",   // plusarg key for the bias .mem path
    parameter        WFILE  = "synth/case04/w1.mem",  // synthesis default
    parameter        BFILE  = "synth/case04/b1.mem",
    localparam int   ACC_W  = $clog2(9 * C_IN * 128) + 3
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic [3:0]            shift,
    input  logic                  s_valid,   // window in (from linebuffer)
    output logic                  s_ready,
    input  logic [C_IN*9*8-1:0]   s_win,
    output logic                  m_valid,   // pixel word out, all C_OUT channels
    input  logic                  m_ready,
    output logic [C_OUT*8-1:0]    m_data
);

    // ---- elaboration-time geometry ---------------------------------------
    localparam int NTAP   = C_IN * 9;            // taps per output channel
    localparam int NCODE  = C_OUT * NTAP;        // weight codes in this layer
    localparam int NBYTE  = (NCODE + 3) / 4;     // packed bytes, 4 codes each
    // idx = base + tap never exceeds NCODE-1, so $clog2(NCODE) bits is exact
    // and idx[MEM_AW+1:2] is the byte address with idx[1:0] the code within it.
    localparam int IDX_W  = $clog2(NCODE);
    localparam int MEM_AW = $clog2(NBYTE);
    localparam int CO_W   = (C_OUT > 1) ? $clog2(C_OUT) : 1;

    localparam logic [IDX_W-1:0] NTAP_C  = IDX_W'(NTAP);
    localparam logic [CO_W-1:0]  CO_LAST = CO_W'(C_OUT - 1);
    localparam logic [CO_W-1:0]  CO_ONE  = CO_W'(1);

    // ---- weight and bias memories ----------------------------------------
    // QUANT_SPEC section 6: code for flat index i lives in byte i/4, bits
    // [2*(i%4) +: 2]. Written as a shift and a mask so no divider appears.
    logic [7:0]  wmem [0:NBYTE-1];
    logic [31:0] bmem [0:C_OUT-1];

`ifndef SYNTHESIS
    // Simulation: the testbench points each instance at a case directory.
    initial begin
        string wpath, bpath;
        wpath = WFILE;
        bpath = BFILE;
        void'($value$plusargs({WKEY, "=%s"}, wpath));
        void'($value$plusargs({BKEY, "=%s"}, bpath));
        $readmemh(wpath, wmem);
        $readmemh(bpath, bmem);
        // 2'b11 is not a weight. A loader that lets it through would silently
        // turn it into a zero tap; the spec says error out instead.
        for (int i = 0; i < NCODE; i++)
            if (wmem[i >> 2][{i[1:0], 1'b0} +: 2] == 2'b11)
                $fatal(1, "conv3x3: illegal weight code 11 at flat index %0d in %s",
                       i, wpath);
    end
`else
    initial begin
        $readmemh(WFILE, wmem);
        $readmemh(BFILE, bmem);
    end
`endif

    // ---- control ---------------------------------------------------------
    logic                       busy;
    logic [CO_W-1:0]            co;
    logic [IDX_W-1:0]           base;      // flat weight index of tap 0 for co
    logic [C_IN*9*8-1:0]        win_r;
    logic [7:0]                 obyte [0:C_OUT-1];

    // s_ready looks only at state and downstream ready, never at s_valid.
    assign s_ready = !busy && (!m_valid || m_ready);

    // ---- ternary tap array ------------------------------------------------
    logic [1:0]        tap_code [0:NTAP-1];
    logic signed [7:0] tap_x    [0:NTAP-1];
    logic signed [9:0] tap_y    [0:NTAP-1];

    genvar gt;
    generate
        for (gt = 0; gt < NTAP; gt++) begin : g_tap
            localparam logic [IDX_W-1:0] TOFF = IDX_W'(gt);
            localparam int               XBIT = gt << 3;
            logic [IDX_W-1:0] idx;
            assign idx          = base + TOFF;
            assign tap_code[gt] = wmem[idx[MEM_AW+1:2]][{idx[1:0], 1'b0} +: 2];
            assign tap_x[gt]    = win_r[XBIT +: 8];
            pe u_pe (.code(tap_code[gt]), .x(tap_x[gt]), .y(tap_y[gt]));
        end
    endgenerate

    // ---- adder tree + bias ------------------------------------------------
    // Flat sum in an always_comb loop; the synthesiser balances it into a tree.
    logic signed [ACC_W-1:0] bias_v, sum;
    assign bias_v = bmem[co][ACC_W-1:0];

    always_comb begin
        sum = bias_v;
        for (int t = 0; t < NTAP; t++)
            sum = sum + $signed({{(ACC_W-10){tap_y[t][9]}}, tap_y[t]});
    end

    // ---- requant + relu ---------------------------------------------------
    logic signed [7:0] q_val, q_out;
    requant #(.ACC_W(ACC_W)) u_requant (.acc(sum), .shift(shift), .q(q_val));

    always_comb begin
        if (RELU && q_val[7]) q_out = 8'sd0;   // relu after requant, on int8
        else                  q_out = q_val;
    end

    // ---- sequencer --------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            busy    <= 1'b0;
            m_valid <= 1'b0;
            co      <= '0;
            base    <= '0;
            win_r   <= '0;
            for (int i = 0; i < C_OUT; i++) obyte[i] <= '0;
        end else begin
            if (m_valid && m_ready) m_valid <= 1'b0;
            if (busy) begin
                obyte[co] <= q_out;
                if (co == CO_LAST) begin
                    busy    <= 1'b0;
                    m_valid <= 1'b1;
                end else begin
                    co   <= co + CO_ONE;
                    base <= base + NTAP_C;
                end
            end else if (s_valid && s_ready) begin
                win_r <= s_win;
                co    <= '0;
                base  <= '0;
                busy  <= 1'b1;
            end
        end
    end

    // ---- output word ------------------------------------------------------
    genvar go;
    generate
        for (go = 0; go < C_OUT; go++) begin : g_out
            localparam int OBIT = go << 3;
            assign m_data[OBIT +: 8] = obyte[go];
        end
    endgenerate

endmodule
