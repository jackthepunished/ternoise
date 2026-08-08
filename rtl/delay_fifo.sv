// Plain synchronous FIFO: the skew buffer that carries the noisy RGB from the
// input of the chain to the residual add at the far end.
//
// The five line buffers together delay the stream by about five rows, so by
// the time layer 5 emits the residual for pixel p the input side has long
// moved on. This FIFO holds the original pixel values in the meantime. It is
// a delay line only in the sense that its occupancy happens to equal the
// number of pixels in flight; there is no counting or matching logic, the
// stream ordering does the alignment and the testbench's bit-exact diff is
// what proves it.
//
// Storage is a single memory with a synchronous read straight into the output
// register - one BRAM-shaped port, one cycle of read latency, no bypass. A
// pop only happens when the memory is non-empty, and when it is non-empty the
// read and write pointers differ, so a read-during-write collision on the
// same address can never occur.
//
// Capacity is DEPTH entries in the memory plus the output register. Depth is
// a power of two so the pointers wrap for free and no comparator against a
// non-constant bound is needed. No multipliers: pointers are counters.
module delay_fifo #(
    parameter int WIDTH      = 24,   // 3 channels x 8 bits
    parameter int DEPTH_LOG2 = 13    // 8192 > 5 x (512 + 1) pixels in flight
) (
    input  logic             clk,
    input  logic             rst,
    input  logic             s_valid,
    output logic             s_ready,
    input  logic [WIDTH-1:0] s_data,
    output logic             m_valid,
    input  logic             m_ready,
    output logic [WIDTH-1:0] m_data
);

    localparam int DEPTH = 1 << DEPTH_LOG2;
    localparam int CNT_W = DEPTH_LOG2 + 1;

    localparam logic [DEPTH_LOG2-1:0] PTR_ONE = DEPTH_LOG2'(1);
    localparam logic [CNT_W-1:0]      CNT_ONE = CNT_W'(1);

    logic [WIDTH-1:0]      mem [0:DEPTH-1];
    logic [DEPTH_LOG2-1:0] wptr, rptr;
    logic [CNT_W-1:0]      cnt;      // entries held in mem, 0 .. DEPTH
    logic                  push, pop;

    // cnt tops out at DEPTH == 1 << DEPTH_LOG2, so its top bit is set for
    // exactly one value: full. Cheaper than a comparator, and it keeps the
    // width cast out of the datapath.
    //
    // These are three separate continuous assignments, not one always_comb,
    // and that matters: s_ready must depend on nothing but state. Bundling it
    // with push (which reads s_valid) into a single block makes the whole
    // block one node to the tools, and the input fork in denoiser_top - where
    // this FIFO's ready gates the line buffer's valid and vice versa - then
    // looks like circular combinational logic (verilator UNOPTFLAT).
    assign s_ready = !cnt[DEPTH_LOG2];
    assign push    = s_valid && s_ready;
    // Pop into the output register whenever it is free (or being drained this
    // cycle) and the memory has something to give.
    assign pop     = (cnt != '0) && (!m_valid || m_ready);

    always_ff @(posedge clk) begin
        if (rst) begin
            wptr    <= '0;
            rptr    <= '0;
            cnt     <= '0;
            m_valid <= 1'b0;
            m_data  <= '0;
        end else begin
            if (m_valid && m_ready) m_valid <= 1'b0;
            if (push) begin
                mem[wptr] <= s_data;
                wptr      <= wptr + PTR_ONE;
            end
            if (pop) begin
                m_data  <= mem[rptr];
                rptr    <= rptr + PTR_ONE;
                m_valid <= 1'b1;
            end
            if      (push && !pop) cnt <= cnt + CNT_ONE;
            else if (!push && pop) cnt <= cnt - CNT_ONE;
        end
    end

endmodule
