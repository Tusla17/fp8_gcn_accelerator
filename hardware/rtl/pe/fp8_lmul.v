// ==============================================================================
// Module: fp8_lmul.v
// Description: DSP-Free FP8 Approximate Multiplier using L-Mul Algorithm
//              Paper: "FP8GCN: An Edge-FPGA-Based Graph Convolutional Network 
//                      Accelerator with FP8 Approximate Multipliers" (IEEE LES)
// Format: FP8 E4M3 (1 Sign bit, 4 Exponent bits with bias=7, 3 Mantissa bits)
// Logic: Replaces mantissa multiplication mx * my with linear shift-add logic:
//        M_res = 1 + mx + my + 2^(-l(m)), where l(3) = 3 -> offset = 1 LSB
// Pipelined: 1 Cycle Latency, Fully combinational / registered options.
// ==============================================================================

`timescale 1ns / 1ps

module fp8_lmul #(
    parameter PIPELINED = 1  // 1: Pipelined with output register, 0: Combinational
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid_in,
    input  wire [7:0] a_fp8,    // Operand A: [7]=Sign, [6:3]=Exp, [2:0]=Mantissa
    input  wire [7:0] b_fp8,    // Operand B: [7]=Sign, [6:3]=Exp, [2:0]=Mantissa
    output reg        valid_out,
    output reg  [7:0] res_fp8   // Result: [7]=Sign, [6:3]=Exp, [2:0]=Mantissa
);

    // 1. Unpack fields
    wire       sign_a = a_fp8[7];
    wire [3:0] exp_a  = a_fp8[6:3];
    wire [2:0] man_a  = a_fp8[2:0];

    wire       sign_b = b_fp8[7];
    wire [3:0] exp_b  = b_fp8[6:3];
    wire [2:0] man_b  = b_fp8[2:0];

    // 2. Sign Logic (XOR)
    wire sign_res = sign_a ^ sign_b;

    // 3. Zero / Subnormal Detection
    wire is_zero = (exp_a == 4'b0000) || (exp_b == 4'b0000);

    // 4. Exponent Addition with Bias=7
    // exp_sum = exp_a + exp_b - 7
    // Range of exp_a + exp_b: 0 to 30. Subtract 7: -7 to 23
    wire signed [5:0] exp_sum_signed = {2'b00, exp_a} + {2'b00, exp_b} - 6'sd7;

    // 5. Mantissa Linear Addition (L-Mul for m=3 bits):
    // Standard mantissas: M_a = 1.man_a, M_b = 1.man_b
    // L-Mul formula: M_res = 1 + man_a/8 + man_b/8 + 1/8
    // In integer units (multiplied by 8):
    // 8 * M_res = 8 + man_a + man_b + 1
    // Range: 8 + 0 + 0 + 1 = 9  to  8 + 7 + 7 + 1 = 23 (fits in 5 bits)
    wire [4:0] man_sum_raw = 5'd8 + {2'b00, man_a} + {2'b00, man_b} + 5'd1;

    // 6. Normalization & Post-processing
    // If man_sum_raw >= 16 (i.e. bit 4 is 1), mantissa overflow occurred:
    // We shift right by 1 bit (divide by 2) and increment exponent.
    wire man_overflow = man_sum_raw[4];

    wire signed [5:0] exp_normalized = man_overflow ? (exp_sum_signed + 6'sd1) : exp_sum_signed;
    wire [2:0]        man_normalized = man_overflow ? man_sum_raw[3:1] : man_sum_raw[2:0];

    // 7. Exponent Saturation / Underflow Handling
    wire exp_underflow = (exp_normalized <= 6'sd0);
    wire exp_overflow  = (exp_normalized >= 6'sd15); // Max normal exponent in E4M3 is 14 or 15

    reg [7:0] comb_res;
    always @(*) begin
        if (is_zero || exp_underflow) begin
            comb_res = 8'b00000000;
        end else if (exp_overflow) begin
            // Saturate to max positive / negative representable normal value
            comb_res = {sign_res, 4'b1110, 3'b111};
        end else begin
            comb_res = {sign_res, exp_normalized[3:0], man_normalized};
        end
    end

    // 8. Pipeline Register Output
    generate
        if (PIPELINED) begin : gen_pipelined
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_out <= 1'b0;
                    res_fp8   <= 8'b00000000;
                end else begin
                    valid_out <= valid_in;
                    res_fp8   <= comb_res;
                end
            end
        end else begin : gen_combinational
            always @(*) begin
                valid_out = valid_in;
                res_fp8   = comb_res;
            end
        end
    endgenerate

endmodule
