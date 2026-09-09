// ==============================================================================
// Module: pe_array.v
// Description: Unified PE Array consisting of 16 PE Groups (256 Approximate Multipliers)
//              Executes Gustavson's element-wise sparse-dense matrix multiplication.
// ==============================================================================

`timescale 1ns / 1ps

module pe_array #(
    parameter NUM_GROUPS = 16, // 16 PE Groups running concurrently
    parameter DIM        = 16  // 16 Multipliers per PE Group (Hidden Dim = 16)
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              valid_in,
    input  wire [NUM_GROUPS*8-1:0]           sparse_vals_fp8, // 16 scalar sparse elements
    input  wire [NUM_GROUPS*DIM*8-1:0]       dense_vecs_fp8,  // 16 x 16 dense vectors
    output wire [NUM_GROUPS-1:0]             valid_out,
    output wire [NUM_GROUPS*DIM*8-1:0]       prod_vecs_fp8    // 16 x 16 partial product vectors
);

    genvar g;
    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_pe_groups
            pe_group #(
                .DIM(DIM)
            ) u_pe_group (
                .clk             (clk),
                .rst_n           (rst_n),
                .valid_in        (valid_in),
                .a_sparse_fp8    (sparse_vals_fp8[g*8 +: 8]),
                .b_dense_vec_fp8 (dense_vecs_fp8[g*DIM*8 +: DIM*8]),
                .valid_out       (valid_out[g]),
                .prod_vec_fp8    (prod_vecs_fp8[g*DIM*8 +: DIM*8])
            );
        end
    endgenerate

endmodule
