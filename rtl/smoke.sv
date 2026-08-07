module smoke (
    input  logic       clk,
    input  logic       rst,
    output logic [7:0] count
);
    // always_ff + logic + parameters: if yosys/verilator accept this file,
    // they accept the project's dialect.
    always_ff @(posedge clk) begin
        if (rst) count <= '0;
        else     count <= count + 8'd1;
    end
endmodule
