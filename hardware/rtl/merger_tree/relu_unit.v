// ==============================================================================
// Module: relu_unit.v
// Description: FP8 ReLU Activation Unit
//              If input sign bit is 1 (negative), output is 0x00.
//              If input sign bit is 0 (positive), output is unchanged.
// ==============================================================================

`timescale 1ns / 1ps

module relu_unit #(
    parameter DIM = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    input  wire [DIM*8-1:0]     data_in_fp8,
    output reg                  valid_out,
    output reg  [DIM*8-1:0]     data_out_fp8
);

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out    <= 1'b0;
            data_out_fp8 <= {DIM*8{1'b0}};
        end else begin
            valid_out <= valid_in;
            for (i = 0; i < DIM; i = i + 1) begin
                if (data_in_fp8[i*8 + 7] == 1'b1) begin
                    // Negative number -> Zero
                    data_out_fp8[i*8 +: 8] <= 8'b00000000;
                end else begin
                    // Positive number -> Unchanged
                    data_out_fp8[i*8 +: 8] <= data_in_fp8[i*8 +: 8];
                end
            end
        end
    end

endmodule
