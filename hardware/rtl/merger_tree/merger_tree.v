// ==============================================================================
// Module: merger_tree.v
// Description: Multi-stage Pipelined Merger Tree combining per-cycle outputs from
//              16 PE Groups. Converts FP8 to Fixed-point for high-precision addition.
// ==============================================================================

`timescale 1ns / 1ps

module fp8_to_fixed16 (
    input  wire [7:0]          fp8_in,
    output reg  signed [15:0]  fixed_out
);
    wire       sign = fp8_in[7];
    wire [3:0] exp  = fp8_in[6:3];
    wire [2:0] man  = fp8_in[2:0];

    // Unbiased exponent with bias=7
    wire signed [4:0] exp_unbiased = {1'b0, exp} - 5'sd7;
    // Significand with implicit 1: M = {1'b1, man} (4 bits, fixed point format 1.3)
    wire [3:0] sig = {1'b1, man};

    always @(*) begin
        if (exp == 4'b0000) begin
            fixed_out = 16'sd0;
        end else begin
            // Shift significand according to exponent: Q8.8 format
            // Bias at 7 means 2^0 is scaled to 8'h01 (in 8 fractional bits)
            // base = sig << (exp - 7 + 5)
            if (exp_unbiased >= 0) begin
                if (exp_unbiased <= 5)
                    fixed_out = ({12'b0, sig} << (exp_unbiased + 4));
                else
                    fixed_out = 16'sh7FFF; // Saturation
            end else begin
                fixed_out = ({12'b0, sig} >> (-exp_unbiased - 4));
            end
            
            if (sign)
                fixed_out = -fixed_out;
        end
    end
endmodule


module merger_tree #(
    parameter NUM_INPUTS = 16 // 16 PE Groups to be merged
)(
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               valid_in,
    input  wire [NUM_INPUTS*8-1:0]            fp8_inputs,    // 16 parallel FP8 scalar elements
    output reg                                valid_out,
    output reg  signed [23:0]                 sum_fixed_out  // Accumulated fixed-point sum
);

    // Step 1: Convert all 16 FP8 inputs to 16-bit fixed-point
    wire signed [15:0] fixed_vals [0:NUM_INPUTS-1];
    genvar i;
    generate
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin : gen_fp8_conv
            fp8_to_fixed16 u_conv (
                .fp8_in    (fp8_inputs[i*8 +: 8]),
                .fixed_out (fixed_vals[i])
            );
        end
    endgenerate

    // Step 2: 4-Stage Pipelined Adder Tree (16 -> 8 -> 4 -> 2 -> 1)
    // Stage 1: 16 -> 8
    reg valid_stg1;
    reg signed [16:0] stg1_sums [0:7];
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_stg1 <= 1'b0;
            for (j = 0; j < 8; j = j + 1) stg1_sums[j] <= 17'sd0;
        end else begin
            valid_stg1 <= valid_in;
            for (j = 0; j < 8; j = j + 1) begin
                stg1_sums[j] <= fixed_vals[2*j] + fixed_vals[2*j+1];
            end
        end
    end

    // Stage 2: 8 -> 4
    reg valid_stg2;
    reg signed [18:0] stg2_sums [0:3];
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_stg2 <= 1'b0;
            for (k = 0; k < 4; k = k + 1) stg2_sums[k] <= 19'sd0;
        end else begin
            valid_stg2 <= valid_stg1;
            for (k = 0; k < 4; k = k + 1) begin
                stg2_sums[k] <= stg1_sums[2*k] + stg1_sums[2*k+1];
            end
        end
    end

    // Stage 3: 4 -> 2
    reg valid_stg3;
    reg signed [20:0] stg3_sums [0:1];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_stg3 <= 1'b0;
            stg3_sums[0] <= 21'sd0;
            stg3_sums[1] <= 21'sd0;
        end else begin
            valid_stg3 <= valid_stg2;
            stg3_sums[0] <= stg2_sums[0] + stg2_sums[1];
            stg3_sums[1] <= stg2_sums[2] + stg2_sums[3];
        end
    end

    // Stage 4: 2 -> 1
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out     <= 1'b0;
            sum_fixed_out <= 24'sd0;
        end else begin
            valid_out     <= valid_stg3;
            sum_fixed_out <= stg3_sums[0] + stg3_sums[1];
        end
    end

endmodule
