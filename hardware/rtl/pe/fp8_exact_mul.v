// ==============================================================================
// Module: fp8_exact_mul.v
// Description: Standard Exact FP8 (E4M3) Multiplier for Baseline Comparison
// ==============================================================================

`timescale 1ns / 1ps

module fp8_exact_mul #(
    parameter PIPELINED = 1
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid_in,
    input  wire [7:0] a_fp8,
    input  wire [7:0] b_fp8,
    output reg        valid_out,
    output reg  [7:0] res_fp8
);

    wire       sign_a = a_fp8[7];
    wire [3:0] exp_a  = a_fp8[6:3];
    wire [2:0] man_a  = a_fp8[2:0];

    wire       sign_b = b_fp8[7];
    wire [3:0] exp_b  = b_fp8[6:3];
    wire [2:0] man_b  = b_fp8[2:0];

    wire sign_res = sign_a ^ sign_b;
    wire is_zero  = (exp_a == 4'b0000) || (exp_b == 4'b0000);

    wire signed [5:0] exp_sum_signed = {2'b00, exp_a} + {2'b00, exp_b} - 6'sd7;

    // Exact mantissa multiplication: (1.man_a) * (1.man_b)
    // {1'b1, man_a} is 4-bit, {1'b1, man_b} is 4-bit -> Product is 8-bit
    wire [7:0] man_prod = {1'b1, man_a} * {1'b1, man_b}; // Range: 16 to 225

    // Normalize: if man_prod >= 32 (bit 7 or bit 6 is 1 -> actually if product >= 32, i.e., bit 7 is set)
    // Since 4-bit * 4-bit max is 15 * 15 = 225 (8'b1110_0001), 1.xxx * 1.yyy has 1 implicit bit at bit 6 or 7.
    wire man_overflow = man_prod[7];

    wire signed [5:0] exp_normalized = man_overflow ? (exp_sum_signed + 6'sd1) : exp_sum_signed;
    // Extract 3 fractional bits (with rounding)
    wire [2:0] man_normalized = man_overflow ? man_prod[6:4] : man_prod[5:3];

    wire exp_underflow = (exp_normalized <= 6'sd0);
    wire exp_overflow  = (exp_normalized >= 6'sd15);

    reg [7:0] comb_res;
    always @(*) begin
        if (is_zero || exp_underflow) begin
            comb_res = 8'b00000000;
        end else if (exp_overflow) begin
            comb_res = {sign_res, 4'b1110, 3'b111};
        end else begin
            comb_res = {sign_res, exp_normalized[3:0], man_normalized};
        end
    end

    generate
        if (PIPELINED) begin : gen_pipe
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_out <= 1'b0;
                    res_fp8   <= 8'b00000000;
                end else begin
                    valid_out <= valid_in;
                    res_fp8   <= comb_res;
                end
            end
        end else begin : gen_comb
            always @(*) begin
                valid_out = valid_in;
                res_fp8   = comb_res;
            end
        end
    endgenerate

endmodule
