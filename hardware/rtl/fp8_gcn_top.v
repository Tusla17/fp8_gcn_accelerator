// ==============================================================================
// Module: fp8_gcn_top.v
// Description: Top-Level FP8 Graph Convolutional Network Accelerator Core
//              - DSP-Free Architecture using FP8 L-Mul Multipliers (256 PEs)
//              - Gustavson Sparse-Dense Element-wise Matrix Multiplication
//              - Double-buffering Ping-Pong Memory Architecture
// ==============================================================================

`timescale 1ns / 1ps

module fp8_gcn_top #(
    parameter NUM_PE_GROUPS = 16,
    parameter DIM           = 16
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Control Interface (AXI4-Lite register mapping)
    input  wire        start_inference,
    input  wire [15:0] cfg_num_nodes,
    input  wire [15:0] cfg_num_features,
    input  wire [15:0] cfg_num_classes,
    output wire        done_inference,
    output wire        busy_inference,
    output wire [2:0]  current_phase,
    
    // DMA / Buffer Streaming Interface
    input  wire                           dma_wr_en,
    input  wire [9:0]                     dma_wr_addr,
    input  wire [NUM_PE_GROUPS*8-1:0]     dma_sparse_data,
    input  wire [NUM_PE_GROUPS*DIM*8-1:0] dma_dense_data,
    
    // Accelerator Output Interface
    output wire                           out_valid,
    output wire [DIM*8-1:0]               out_node_features_fp8
);

    // Internal Control Signals
    wire pe_valid_in;
    wire buffer_switch;
    wire relu_enable;
    wire writeback_en;

    // 1. Controller FSM
    gcn_controller #(
        .DIM(DIM)
    ) u_controller (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start_inference),
        .num_nodes     (cfg_num_nodes),
        .num_features  (cfg_num_features),
        .num_classes   (cfg_num_classes),
        .done          (done_inference),
        .busy          (busy_inference),
        .current_phase (current_phase),
        .pe_valid_in   (pe_valid_in),
        .buffer_switch (buffer_switch),
        .relu_enable   (relu_enable),
        .writeback_en  (writeback_en)
    );

    // 2. Buffers (Ping-Pong double buffers for Sparse and Dense Data)
    wire [NUM_PE_GROUPS*8-1:0]     sparse_stream_fp8;
    wire [NUM_PE_GROUPS*DIM*8-1:0] dense_stream_fp8;

    ping_pong_buffer #(
        .DATA_WIDTH(NUM_PE_GROUPS*8),
        .ADDR_WIDTH(10)
    ) u_sparse_buffer (
        .clk         (clk),
        .rst_n       (rst_n),
        .bank_switch (buffer_switch),
        .wr_en       (dma_wr_en),
        .wr_addr     (dma_wr_addr),
        .wr_data     (dma_sparse_data),
        .rd_en       (pe_valid_in),
        .rd_addr     (dma_wr_addr),
        .rd_data     (sparse_stream_fp8)
    );

    ping_pong_buffer #(
        .DATA_WIDTH(NUM_PE_GROUPS*DIM*8),
        .ADDR_WIDTH(10)
    ) u_dense_buffer (
        .clk         (clk),
        .rst_n       (rst_n),
        .bank_switch (buffer_switch),
        .wr_en       (dma_wr_en),
        .wr_addr     (dma_wr_addr),
        .wr_data     (dma_dense_data),
        .rd_en       (pe_valid_in),
        .rd_addr     (dma_wr_addr),
        .rd_data     (dense_stream_fp8)
    );

    // 3. Unified PE Array (256 Multipliers: 16 Groups x 16 Dimensions)
    wire [NUM_PE_GROUPS-1:0]       pe_valid_outs;
    wire [NUM_PE_GROUPS*DIM*8-1:0] pe_prod_vecs_fp8;

    pe_array #(
        .NUM_GROUPS (NUM_PE_GROUPS),
        .DIM        (DIM)
    ) u_pe_array (
        .clk             (clk),
        .rst_n           (rst_n),
        .valid_in        (pe_valid_in),
        .sparse_vals_fp8 (sparse_stream_fp8),
        .dense_vecs_fp8  (dense_stream_fp8),
        .valid_out       (pe_valid_outs),
        .prod_vecs_fp8   (pe_prod_vecs_fp8)
    );

    // 4. Reduction & Merger Trees (1 Tree per Dimension: 16 Trees in parallel)
    wire [DIM-1:0]        merger_valids;
    wire [DIM*24-1:0]     merger_sums_packed;

    genvar d;
    generate
        for (d = 0; d < DIM; d = d + 1) begin : gen_mergers
            // Collect d-th dimension component from all 16 PE Groups
            wire [NUM_PE_GROUPS*8-1:0] d_inputs_fp8;
            genvar g;
            for (g = 0; g < NUM_PE_GROUPS; g = g + 1) begin : gen_dim_gather
                assign d_inputs_fp8[g*8 +: 8] = pe_prod_vecs_fp8[(g*DIM + d)*8 +: 8];
            end

            merger_tree #(
                .NUM_INPUTS(NUM_PE_GROUPS)
            ) u_merger_tree (
                .clk           (clk),
                .rst_n         (rst_n),
                .valid_in      (pe_valid_outs[0]),
                .fp8_inputs    (d_inputs_fp8),
                .valid_out     (merger_valids[d]),
                .sum_fixed_out (merger_sums_packed[d*24 +: 24])
            );
        end
    endgenerate

    // 5. Row Accumulator Module
    wire                  accum_valid;
    wire [DIM*8-1:0]      accum_row_fp8;

    accumulator #(
        .DIM(DIM)
    ) u_accumulator (
        .clk              (clk),
        .rst_n            (rst_n),
        .valid_in         (merger_valids[0]),
        .row_end          (1'b1), // Per-cycle partial sum accumulation flag
        .sum_fixed_packed (merger_sums_packed),
        .valid_out        (accum_valid),
        .row_out_fp8      (accum_row_fp8)
    );

    // 6. Post-processing: ReLU Unit
    wire             relu_valid;
    wire [DIM*8-1:0] relu_out_fp8;

    relu_unit #(
        .DIM(DIM)
    ) u_relu (
        .clk         (clk),
        .rst_n       (rst_n),
        .valid_in    (accum_valid),
        .data_in_fp8 (accum_row_fp8),
        .valid_out   (relu_valid),
        .data_out_fp8(relu_out_fp8)
    );

    // Final Output Multiplexer
    assign out_valid              = relu_enable ? relu_valid : accum_valid;
    assign out_node_features_fp8  = relu_enable ? relu_out_fp8 : accum_row_fp8;

endmodule
