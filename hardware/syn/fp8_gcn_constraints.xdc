# ==============================================================================
# Timing & Clock Constraints for FP8 GCN Accelerator (ZCU106 / Zynq / Kintex)
# Target Clock: 200 MHz (Period = 5.000 ns)
# ==============================================================================

# 1. Primary Clock Definition (200 MHz)
create_clock -name aclk -period 5.000 [get_ports aclk]

# 2. Input Delays (AXI-Stream & AXI-Lite inputs)
set_input_delay -clock aclk -max 1.000 [get_ports {s_axi_ctrl* s_axis* aresetn}]
set_input_delay -clock aclk -min 0.200 [get_ports {s_axi_ctrl* s_axis* aresetn}]

# 3. Output Delays (AXI-Stream & AXI-Lite outputs & Status flags)
set_output_delay -clock aclk -max 1.000 [get_ports {s_axi_ctrl* m_axis* irq_done busy_led}]
set_output_delay -clock aclk -min 0.200 [get_ports {s_axi_ctrl* m_axis* irq_done busy_led}]
