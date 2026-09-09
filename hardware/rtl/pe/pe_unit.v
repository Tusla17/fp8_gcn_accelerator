// ==============================================================================
// Module: pe_unit.v
// Description: Fine-grained Processing Element Unit using FP8 L-Mul Multiplier
// ==============================================================================

`timescale 1ns / 1ps

module pe_unit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid_in,
    input  wire [7:0] a_sparse_fp8, // Sparse matrix non-zero value
    input  wire [7:0] b_dense_fp8,  // Dense matrix element
    output wire       valid_out,
    output wire [7:0] prod_fp8      // Partial product result in FP8
);

    // Instantiate FP8 L-Mul core (DSP-Free)
    fp8_lmul #(
        .PIPELINED(1)
    ) u_fp8_lmul (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .a_fp8     (a_sparse_fp8),
        .b_fp8     (b_dense_fp8),
        .valid_out (valid_out),
        .res_fp8   (prod_fp8)
    );

endmodule
