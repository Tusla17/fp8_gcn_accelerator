# ==============================================================================
# Script to create standard Vivado Project (.xpr) inside D:/dacn/hardware/syn/
# Target: Zynq-7000 xc7z020clg400-1 (or Artix-7 xc7a100tcsg324-1)
# ==============================================================================

set PRJ_DIR "./fp8_gcn_project"

# 1. Create Vivado Project
create_project -force fp8_gcn_accel $PRJ_DIR -part xc7z020clg400-1

# 2. Add RTL Source Files
add_files -norecurse [list \
    "../rtl/pe/fp8_lmul.v" \
    "../rtl/pe/fp8_exact_mul.v" \
    "../rtl/pe/drum4_mul.v" \
    "../rtl/pe/pe_unit.v" \
    "../rtl/pe/pe_group.v" \
    "../rtl/pe/pe_array.v" \
    "../rtl/buffer/ping_pong_buffer.v" \
    "../rtl/merger_tree/merger_tree.v" \
    "../rtl/merger_tree/accumulator.v" \
    "../rtl/merger_tree/relu_unit.v" \
    "../rtl/controller/gcn_controller.v" \
    "../rtl/fp8_gcn_top.v" \
]

# 3. Add Simulation Testbenches
add_files -fileset sim_1 -norecurse [list \
    "../tb/tb_approx_multipliers.v" \
    "../tb/tb_fp8_gcn_top.v" \
]

# 4. Set Top-level Modules
set_property top fp8_gcn_top [current_fileset]
set_property top tb_fp8_gcn_top [get_filesets sim_1]

# 5. Set SystemVerilog Properties
set_property file_type {SystemVerilog} [get_files -filter {NAME =~ "*.v"}]

# 6. Update Hierarchy
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "=================================================================="
puts "  VIVADO PROJECT CREATED SUCCESSFULLY!"
puts "  Project File Location: $PRJ_DIR/fp8_gcn_accel.xpr"
puts "=================================================================="
