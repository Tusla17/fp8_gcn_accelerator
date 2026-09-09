# ==============================================================================
# Vivado Synthesis & Implementation Script for FP8 GCN Accelerator
# Target: Kintex-7 xc7k325tffg900-2 (or Artix-7 / Zynq-7000)
# ==============================================================================

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set PROJECT_DIR "$SCRIPT_DIR/vivado_prj"
set OUTPUT_DIR  "$SCRIPT_DIR/output"

file mkdir $OUTPUT_DIR

puts "=================================================================="
puts "  STARTING VIVADO SYNTHESIS FOR FP8 GCN ACCELERATOR"
puts "=================================================================="

# 1. Read Verilog RTL Source Files
read_verilog -sv "$SCRIPT_DIR/../rtl/pe/fp8_lmul.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/pe/pe_unit.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/pe/pe_group.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/pe/pe_array.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/buffer/ping_pong_buffer.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/merger_tree/merger_tree.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/merger_tree/accumulator.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/merger_tree/relu_unit.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/controller/gcn_controller.v"
read_verilog -sv "$SCRIPT_DIR/../rtl/fp8_gcn_top.v"

# 2. Synthesis (DSP-Free Optimization)
# Target Part: Kintex-7 (xc7k325tffg900-2) or Zynq (xc7z020clg400-1)
synth_design -top fp8_gcn_top -part xc7k325tffg900-2 -max_dsp 0 -mode out_of_context

# 3. Create Clock Constraint (200 MHz -> Period: 5.0 ns)
create_clock -name clk -period 5.000 [get_ports clk]

# 4. Generate Synthesis Reports & Schematic Data
report_utilization -file "$OUTPUT_DIR/utilization_synth.rpt"
report_timing_summary -file "$OUTPUT_DIR/timing_synth.rpt"

# 5. Run Implementation (Place & Route)
opt_design
place_design
phys_opt_design
route_design

# 6. Generate Final Implementation & Power Reports
report_utilization -file "$OUTPUT_DIR/utilization_impl.rpt"
report_timing_summary -file "$OUTPUT_DIR/timing_impl.rpt"
report_power -file "$OUTPUT_DIR/power_impl.rpt"

puts "=================================================================="
puts "  SYNTHESIS & IMPLEMENTATION COMPLETED SUCCESSFULLY!"
puts "  Check report files in: $OUTPUT_DIR"
puts "    1. utilization_impl.rpt (LUTs, FFs, BRAMs, 0 DSPs)"
puts "    2. timing_impl.rpt      (Timing Slack & 200 MHz Fmax)"
puts "    3. power_impl.rpt       (Dynamic & Static Power)"
puts "=================================================================="
