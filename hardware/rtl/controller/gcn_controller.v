// ==============================================================================
// Module: gcn_controller.v
// Description: Central FSM Controller coordinating the A(XW) execution dataflow
// ==============================================================================

`timescale 1ns / 1ps

module gcn_controller #(
    parameter DIM = 16
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] num_nodes,
    input  wire [15:0] num_features,
    input  wire [15:0] num_classes,
    
    // Status signals
    output reg         done,
    output reg         busy,
    output reg  [2:0]  current_phase, // 0: Idle, 1: GEMM1, 2: SpMM1, 3: GEMM2, 4: SpMM2, 5: Done
    
    // Dataflow control to Buffers & PE Array
    output reg         pe_valid_in,
    output reg         buffer_switch,
    output reg         relu_enable,
    output reg         writeback_en
);

    // State definitions
    localparam STATE_IDLE         = 3'd0;
    localparam STATE_GEMM_LAYER1  = 3'd1; // Z0 = X * W0
    localparam STATE_SPMM_LAYER1  = 3'd2; // H1 = ReLU(A * Z0)
    localparam STATE_GEMM_LAYER2  = 3'd3; // Z1 = H1 * W1
    localparam STATE_SPMM_LAYER2  = 3'd4; // H2 = A * Z1
    localparam STATE_DONE         = 3'd5;

    reg [2:0]  state, next_state;
    reg [15:0] step_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= STATE_IDLE;
            step_counter <= 16'd0;
        end else begin
            state <= next_state;
            if (state != next_state)
                step_counter <= 16'd0;
            else
                step_counter <= step_counter + 16'd1;
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            STATE_IDLE: begin
                if (start) next_state = STATE_GEMM_LAYER1;
            end
            STATE_GEMM_LAYER1: begin
                // In full hardware, monitored via buffer/row done signals
                if (step_counter >= num_nodes) next_state = STATE_SPMM_LAYER1;
            end
            STATE_SPMM_LAYER1: begin
                if (step_counter >= num_nodes) next_state = STATE_GEMM_LAYER2;
            end
            STATE_GEMM_LAYER2: begin
                if (step_counter >= num_nodes) next_state = STATE_SPMM_LAYER2;
            end
            STATE_SPMM_LAYER2: begin
                if (step_counter >= num_nodes) next_state = STATE_DONE;
            end
            STATE_DONE: begin
                next_state = STATE_IDLE;
            end
            default: next_state = STATE_IDLE;
        endcase
    end

    // Output control signals
    always @(*) begin
        current_phase = state;
        done          = (state == STATE_DONE);
        busy          = (state != STATE_IDLE && state != STATE_DONE);
        pe_valid_in   = busy;
        relu_enable   = (state == STATE_SPMM_LAYER1);
        buffer_switch = (step_counter == num_nodes);
        writeback_en  = (state == STATE_SPMM_LAYER2);
    end

endmodule
