# ==============================================================================
# Timing & Clock Constraints for FP8 GCN Accelerator (ZCU106 / Zynq / Kintex)
# Target Clock: 200 MHz (Period = 5.000 ns)
# ==============================================================================

# 1. Primary Clock Definition (200 MHz on internal core)
create_clock -name aclk -period 5.000 [get_ports aclk]

# 2. Relax External I/O Delays for Standalone Core Implementation
# (Prevents unassigned package pin delays from violating 200 MHz register-to-register timing)
set_false_path -from [get_ports -filter {DIRECTION == IN && NAME !~ "aclk"}]
set_false_path -to [get_ports -filter {DIRECTION == OUT}]

# 3. Allow Bitstream Generation with Unconstrained Virtual I/O Pins
# (Overrides DRC NSTD-1 and UCIO-1 so Bitgen succeeds without physical board pin lock)
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
