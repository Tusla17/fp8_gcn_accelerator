// ==============================================================================
// Module: pe_group.v
// Description: PE Group containing 16 parallel PE Units (Hidden Dimension = 16)
//              Multiplies 1 sparse scalar element with a 16-element dense vector.
// ==============================================================================

`timescale 1ns / 1ps

module pe_group #(
    parameter DIM = 16 // Hidden dimension = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    input  wire [7:0]           a_sparse_fp8,     // Broadcast scalar element
    input  wire [DIM*8-1:0]     b_dense_vec_fp8,  // Packed 16-element FP8 vector
    output wire                 valid_out,
    output wire [DIM*8-1:0]     prod_vec_fp8      // Packed 16-element FP8 product
);

    wire [DIM-1:0] valid_outs;
    assign valid_out = valid_outs[0];

    genvar i;
    generate
        for (i = 0; i < DIM; i = i + 1) begin : gen_pe_units
            pe_unit u_pe (
                .clk           (clk),
                .rst_n         (rst_n),
                .valid_in      (valid_in),
                .a_sparse_fp8  (a_sparse_fp8),
                .b_dense_fp8   (b_dense_vec_fp8[i*8 +: 8]),
                .valid_out     (valid_outs[i]),
                .prod_fp8      (prod_vec_fp8[i*8 +: 8])
            );
        end
    endgenerate

endmodule
