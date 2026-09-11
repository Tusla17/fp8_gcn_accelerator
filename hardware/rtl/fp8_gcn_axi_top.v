// ==============================================================================
// Module: fp8_gcn_axi_top.v
// Description: Top-level SoC Wrapper with 32-bit/64-bit AXI-Stream & AXI-Lite
//              interfaces for Physical FPGA Deployment (Low Pin Count).
// ==============================================================================

`timescale 1ns / 1ps

module fp8_gcn_axi_top #(
    parameter AXI_DATA_WIDTH = 32
)(
    input  wire                      aclk,
    input  wire                      aresetn,
    
    // AXI-Lite Control Slave
    input  wire                      s_axi_ctrl_awvalid,
    input  wire [7:0]                s_axi_ctrl_awaddr,
    output wire                      s_axi_ctrl_awready,
    input  wire                      s_axi_ctrl_wvalid,
    input  wire [31:0]               s_axi_ctrl_wdata,
    output wire                      s_axi_ctrl_wready,
    output wire                      s_axi_ctrl_bvalid,
    input  wire                      s_axi_ctrl_bready,
    
    // AXI-Stream Slave (DMA Stream In: Features & Weights)
    input  wire                      s_axis_tvalid,
    input  wire [AXI_DATA_WIDTH-1:0] s_axis_tdata,
    input  wire                      s_axis_tlast,
    output wire                      s_axis_tready,
    
    // AXI-Stream Master (DMA Stream Out: Node Embeddings)
    output wire                      m_axis_tvalid,
    output wire [AXI_DATA_WIDTH-1:0] m_axis_tdata,
    output wire                      m_axis_tlast,
    input  wire                      m_axis_tready,
    
    // Status Interrupt / LED
    output wire                      irq_done,
    output wire                      busy_led
);

    // Internal Registers for AXI-Lite Config
    reg [15:0] reg_num_nodes;
    reg [15:0] reg_num_features;
    reg [15:0] reg_num_classes;
    reg        reg_start_pulse;
    
    reg        axi_awready;
    reg        axi_wready;
    reg        axi_bvalid;

    assign s_axi_ctrl_awready = axi_awready;
    assign s_axi_ctrl_wready  = axi_wready;
    assign s_axi_ctrl_bvalid  = axi_bvalid;
    
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            axi_awready      <= 1'b0;
            axi_wready       <= 1'b0;
            axi_bvalid       <= 1'b0;
            reg_num_nodes    <= 16'd32;
            reg_num_features <= 16'd1433;
            reg_num_classes  <= 16'd7;
            reg_start_pulse  <= 1'b0;
        end else begin
            reg_start_pulse <= 1'b0;
            if (~axi_awready && s_axi_ctrl_awvalid && s_axi_ctrl_wvalid) begin
                axi_awready <= 1'b1;
                axi_wready  <= 1'b1;
                axi_bvalid  <= 1'b1;
                case (s_axi_ctrl_awaddr[5:2])
                    4'h0: reg_start_pulse  <= s_axi_ctrl_wdata[0];
                    4'h1: reg_num_nodes    <= s_axi_ctrl_wdata[15:0];
                    4'h2: reg_num_features <= s_axi_ctrl_wdata[15:0];
                    4'h3: reg_num_classes  <= s_axi_ctrl_wdata[15:0];
                endcase
            end else begin
                axi_awready <= 1'b0;
                axi_wready  <= 1'b0;
                if (s_axi_ctrl_bready && axi_bvalid)
                    axi_bvalid <= 1'b0;
            end
        end
    end

    // Explicit tie-off for standard protocol bus bits (removes Synth 8-7129 warnings)
    (* keep = "true" *) wire unused_axi_bits = |{s_axi_ctrl_awaddr[7:6], s_axi_ctrl_awaddr[1:0], s_axi_ctrl_wdata[31:16], s_axis_tlast};

    // Internal Wires to Accelerator Core
    wire        core_valid_out;
    wire [127:0]core_out_feat;
    wire        core_busy;
    wire        core_done;
    wire [2:0]  core_phase;

    // Deserializer: Pack 32-bit AXI Stream into 2048-bit Dense Buffer / 128-bit Sparse Buffer
    reg [2047:0] dense_shift_reg;
    reg [127:0]  sparse_shift_reg;
    reg [5:0]    word_counter;
    reg          dma_wr_en_reg;
    reg [9:0]    dma_addr_reg;
    reg          axis_tready_reg;
    
    // Dynamic AXI-Stream backpressure: Ready when core is not busy (removes Synth 8-3917)
    assign s_axis_tready = axis_tready_reg;
    
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            axis_tready_reg  <= 1'b0;
            word_counter     <= 0;
            dma_wr_en_reg    <= 0;
            dma_addr_reg     <= 0;
            dense_shift_reg  <= 0;
            sparse_shift_reg <= 0;
        end else begin
            axis_tready_reg  <= ~core_busy;
            if (s_axis_tvalid && axis_tready_reg) begin
                dense_shift_reg  <= {dense_shift_reg[2047-AXI_DATA_WIDTH:0], s_axis_tdata};
                sparse_shift_reg <= {sparse_shift_reg[127-AXI_DATA_WIDTH:0], s_axis_tdata};
                
                if (word_counter == 6'd63) begin // 64 words x 32 bits = 2048 bits
                    word_counter  <= 0;
                    dma_wr_en_reg <= 1'b1;
                    dma_addr_reg  <= dma_addr_reg + 1'b1;
                end else begin
                    word_counter  <= word_counter + 1'b1;
                    dma_wr_en_reg <= 1'b0;
                end
            end else begin
                dma_wr_en_reg <= 1'b0;
            end
        end
    end

    // Instance of Top Core
    fp8_gcn_top u_core (
        .clk                  (aclk),
        .rst_n                (aresetn),
        .start_inference      (reg_start_pulse),
        .cfg_num_nodes        (reg_num_nodes),
        .cfg_num_features     (reg_num_features),
        .cfg_num_classes      (reg_num_classes),
        .done_inference       (core_done),
        .busy_inference       (core_busy),
        .current_phase        (core_phase),
        .dma_wr_en            (dma_wr_en_reg),
        .dma_wr_addr          (dma_addr_reg),
        .dma_sparse_data      (sparse_shift_reg),
        .dma_dense_data       (dense_shift_reg),
        .out_valid            (core_valid_out),
        .out_node_features_fp8(core_out_feat)
    );

    // Serializer for Output AXI-Stream
    reg [127:0] out_shift_reg;
    reg [2:0]   out_word_idx;
    reg         m_axis_valid_reg;
    
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            out_shift_reg    <= 0;
            out_word_idx     <= 0;
            m_axis_valid_reg <= 0;
        end else if (core_valid_out) begin
            out_shift_reg    <= core_out_feat;
            out_word_idx     <= 3'd4; // 128 bit / 32 bit = 4 words
            m_axis_valid_reg <= 1'b1;
        end else if (m_axis_valid_reg && m_axis_tready) begin
            if (out_word_idx > 1) begin
                out_shift_reg <= {32'd0, out_shift_reg[127:32]};
                out_word_idx  <= out_word_idx - 1'b1;
            end else begin
                m_axis_valid_reg <= 1'b0;
            end
        end
    end
    
    assign m_axis_tvalid = m_axis_valid_reg;
    assign m_axis_tdata  = out_shift_reg[31:0];
    assign m_axis_tlast  = (out_word_idx == 1);
    assign irq_done      = core_done;
    assign busy_led      = core_busy;

endmodule
