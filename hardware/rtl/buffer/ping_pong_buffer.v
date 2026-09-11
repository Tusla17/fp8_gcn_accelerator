// ==============================================================================
// Module: ping_pong_buffer.v
// Description: Double-buffering Ping-Pong SRAM/BRAM module for continuous streaming
//              between on-chip PE Array and off-chip memory (DDR/AXI).
// ==============================================================================

`timescale 1ns / 1ps

module ping_pong_buffer #(
    parameter DATA_WIDTH = 128, // 16 elements x 8-bit FP8 = 128 bits
    parameter ADDR_WIDTH = 10   // 1024 depth
)(
    input  wire                  clk,
    input  wire                  rst_n,
    
    // Switch ping-pong banks
    input  wire                  bank_switch,
    
    // Write Port (Streaming from DDR / DMA)
    input  wire                  wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [DATA_WIDTH-1:0] wr_data,
    
    // Read Port (Feeding into PE Groups)
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [DATA_WIDTH-1:0] rd_data
);

    reg active_bank; // 0: Bank0 write / Bank1 read, 1: Bank1 write / Bank0 read

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_bank <= 1'b0;
        end else if (bank_switch) begin
            active_bank <= ~active_bank;
        end
    end

    // Dual-port BRAM instances (Bank 0 and Bank 1)
    // Explicit Xilinx BRAM synthesis attribute
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem_bank0 [0:(1<<ADDR_WIDTH)-1];
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem_bank1 [0:(1<<ADDR_WIDTH)-1];

    reg [DATA_WIDTH-1:0] dout_bank0;
    reg [DATA_WIDTH-1:0] dout_bank1;

    // Bank 0 Memory Access (Standard Synchronous BRAM Template)
    always @(posedge clk) begin
        if (wr_en && (active_bank == 1'b0)) begin
            mem_bank0[wr_addr] <= wr_data;
        end
        if (rd_en) begin
            dout_bank0 <= mem_bank0[rd_addr];
        end
    end

    // Bank 1 Memory Access (Standard Synchronous BRAM Template)
    always @(posedge clk) begin
        if (wr_en && (active_bank == 1'b1)) begin
            mem_bank1[wr_addr] <= wr_data;
        end
        if (rd_en) begin
            dout_bank1 <= mem_bank1[rd_addr];
        end
    end

    // Output MUX (Bank0 write -> Bank1 read, Bank1 write -> Bank0 read)
    assign rd_data = (active_bank == 1'b0) ? dout_bank1 : dout_bank0;

endmodule
