// ==============================================================================
// Module: drum4_mul.v
// Description: Dynamic Range Approximate Multiplier (DRUM-4) for 8-bit Signed Int
// ==============================================================================

`timescale 1ns / 1ps

module drum4_mul #(
    parameter PIPELINED = 1
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid_in,
    input  wire signed [7:0] a_in,
    input  wire signed [7:0] b_in,
    output reg               valid_out,
    output reg  signed [15:0] res_out
);

    wire sign_a = a_in[7];
    wire sign_b = b_in[7];
    wire sign_res = sign_a ^ sign_b;

    wire [7:0] abs_a = sign_a ? (-a_in) : a_in;
    wire [7:0] abs_b = sign_b ? (-b_in) : b_in;

    // Leading one detector (LOD) for 8-bit
    function [2:0] lod8;
        input [7:0] val;
        begin
            if (val[7]) lod8 = 3'd7;
            else if (val[6]) lod8 = 3'd6;
            else if (val[5]) lod8 = 3'd5;
            else if (val[4]) lod8 = 3'd4;
            else if (val[3]) lod8 = 3'd3;
            else if (val[2]) lod8 = 3'd2;
            else if (val[1]) lod8 = 3'd1;
            else lod8 = 3'd0;
        end
    endfunction

    wire [2:0] lod_a = lod8(abs_a);
    wire [2:0] lod_b = lod8(abs_b);

    // Shifts for k=4
    wire [2:0] shift_a = (lod_a >= 3'd3) ? (lod_a - 3'd3) : 3'd0;
    wire [2:0] shift_b = (lod_b >= 3'd3) ? (lod_b - 3'd3) : 3'd0;

    wire [3:0] k_a = abs_a >> shift_a;
    wire [3:0] k_b = abs_b >> shift_b;

    wire [7:0] k_prod = k_a * k_b;
    wire [15:0] shifted_prod = k_prod << (shift_a + shift_b);

    wire signed [15:0] comb_res = (abs_a == 0 || abs_b == 0) ? 16'sd0 :
                                  (sign_res ? -shifted_prod : shifted_prod);

    generate
        if (PIPELINED) begin : gen_pipe
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_out <= 1'b0;
                    res_out   <= 16'sd0;
                end else begin
                    valid_out <= valid_in;
                    res_out   <= comb_res;
                end
            end
        end else begin : gen_comb
            always @(*) begin
                valid_out = valid_in;
                res_out   = comb_res;
            end
        end
    endgenerate

endmodule
