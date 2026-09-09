// ==============================================================================
// Module: accumulator.v
// Description: Multi-cycle Row-wise Accumulator with Fixed-to-FP8 Re-quantization
// ==============================================================================

`timescale 1ns / 1ps

module fixed24_to_fp8 (
    input  wire signed [23:0] fixed_in,
    output reg  [7:0]         fp8_out
);
    wire sign = fixed_in[23];
    wire [22:0] abs_val = sign ? (-fixed_in[22:0]) : fixed_in[22:0];

    // Leading one detector (LOD) for 23-bit
    function [4:0] lod23;
        input [22:0] v;
        integer idx;
        begin
            lod23 = 5'd0;
            for (idx = 0; idx < 23; idx = idx + 1) begin
                if (v[idx]) lod23 = idx[4:0];
            end
        end
    endfunction

    wire [4:0] lead_pos = lod23(abs_val);
    
    // In Q16.8 format: bit 8 is 2^0 (exponent = 0, biased exp = 7)
    // lead_pos = 8 -> exp_unbiased = 0 -> exp_biased = 7
    wire signed [5:0] exp_calc = {1'b0, lead_pos} - 6'sd8 + 6'sd7;

    always @(*) begin
        if (abs_val == 23'd0) begin
            fp8_out = 8'b00000000;
        end else if (exp_calc <= 0) begin
            fp8_out = 8'b00000000;
        end else if (exp_calc >= 15) begin
            // Saturate
            fp8_out = {sign, 4'b1110, 3'b111};
        end else begin
            // Extract 3 mantissa bits below leading 1
            if (lead_pos >= 3)
                fp8_out = {sign, exp_calc[3:0], abs_val[lead_pos-1 -: 3]};
            else
                fp8_out = {sign, exp_calc[3:0], (abs_val << (3 - lead_pos)) & 3'b111};
        end
    end
endmodule


module accumulator #(
    parameter DIM = 16
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      valid_in,
    input  wire                      row_end,         // Flag indicating end of current sparse row
    input  wire [DIM*24-1:0]         sum_fixed_packed,// Packed 16 x 24-bit fixed point inputs
    output reg                       valid_out,
    output wire [DIM*8-1:0]          row_out_fp8
);

    reg signed [31:0] acc_reg [0:DIM-1];
    
    genvar d;
    generate
        for (d = 0; d < DIM; d = d + 1) begin : gen_fixed_to_fp8
            fixed24_to_fp8 u_f2fp8 (
                .fixed_in (acc_reg[d][31:8]),
                .fp8_out  (row_out_fp8[d*8 +: 8])
            );
        end
    endgenerate

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (i = 0; i < DIM; i = i + 1) acc_reg[i] <= 32'sd0;
        end else if (valid_in) begin
            if (row_end) begin
                valid_out <= 1'b1;
                for (i = 0; i < DIM; i = i + 1) begin
                    acc_reg[i] <= $signed(sum_fixed_packed[i*24 +: 24]); // Reset with new row element
                end
            end else begin
                valid_out <= 1'b0;
                for (i = 0; i < DIM; i = i + 1) begin
                    acc_reg[i] <= acc_reg[i] + $signed(sum_fixed_packed[i*24 +: 24]);
                end
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
