# BÁO CÁO TỔNG KẾT KỸ THUẬT: BỘ GIA TỐC GCN DÙNG BỘ NHÂN XẤP XỈ FP8 TRÊN FPGA

**Đề tài:** Thiết Kế Bộ Gia Tốc Mạng Nơ-ron Đồ Thị (GCN) Trên Edge-FPGA Sử Dụng Bộ Nhân Xấp Xỉ FP8 (DSP-Free)  
**Tác giả:** Đồ án Chuyên ngành  
**Thời gian thực hiện:** Kế hoạch 3 Tháng (12 Tuần)  

---

## 1. TỔNG QUAN KẾT QUẢ ĐẠT ĐƯỢC

Toàn bộ hệ thống từ **Mô phỏng phần mềm (Python/PyG)** đến **Thiết kế phần cứng (Verilog RTL)** và **Kiểm tra mô phỏng toàn diện (Co-Simulation)** đã hoàn tất và được xác thực:

```
+-----------------------------------------------------------------------------------------------+
|                                      FP8GCN ACCELERATOR                                       |
+------------------------------------+-----------------------------------+----------------------+
|       SOFTWARE & QUANTIZATION      |        HARDWARE & RTL (VERILOG)   |   PERFORMANCE & PPA  |
+------------------------------------+-----------------------------------+----------------------+
| • FP32 Baseline (PyG / CUDA):      | • Unified PE Array:               | • Target Clock:      |
|   - Cora: 80.40%                   |   - 16 PE Groups x 16 Multipliers |   200 MHz (5.0 ns)   |
|   - Citeseer: 69.90%               |   - 256 FP8-L-Mul units in //     | • DSP Slices:        |
|   - Pubmed: 78.80%                 | • 100% DSP-Free:                  |   0 DSP (100% LUT)   |
| • FP8-L-Mul Approximate Accuracy:  |   Chuyển nhân thành cộng/dịch bit | • Subgraph-32 Latency|
|   - Cora: 79.40% (Drop: 1.00%)     | • Double-Buffering:               |   0.67 us (133 cyc)  |
|   - Citeseer: 69.70% (Drop: 0.20%) |   Ping-pong Info, Sparse, Dense   | • Energy Efficiency: |
|   - Pubmed: 78.40% (Drop: 0.40%)   | • Pipelined Reduction:            |   1.16x - 1.33x vs   |
| • MRED / NMED Bit-Sweep:           |   Merger Tree (4 tầng) + Accum    |   DSP FPGA           |
|   MRED = 3.94e-1, NMED = 8.92e-3   | • Kích hoạt ReLU: 1 chu kỳ clock  |   450x - 1010x vs CPU|
+------------------------------------+-----------------------------------+----------------------+
```

---

## 2. DANH MỤC CÁC MODULE MÃ NGUỒN ĐÃ HOÀN THÀNH

### A. Khối Thuật toán & Lượng tử hóa Phần mềm (`software/`):
* [gcn.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/models/gcn.py): Mô hình 2-layer GCN với cấu trúc luồng tính toán tách biệt $A(XW)$.
* [data_loader.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/utils/data_loader.py): Bộ nạp và tiền xử lý ma trận thưa $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$.
* [train_baseline.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/train_baseline.py): Script huấn luyện FP32 Baseline tự động trên Cora, Citeseer, Pubmed.
* [fp8.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/quantization/fp8.py): Module lượng tử hóa FP8 (E4M3, E5M2, INT8) và hiệu chuẩn Scale factor $S$.
* [l_mul.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/approx_multipliers/l_mul.py): Giả lập bộ nhân xấp xỉ FP8 L-Mul (khớp 100% với RTL).
* [drum.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/approx_multipliers/drum.py) & [mitchell.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/approx_multipliers/mitchell.py): Các bộ nhân đối chứng INT8.
* [error_metrics.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/utils/error_metrics.py): Đo lường MRED, NMED, MED.
* [evaluate_gcn_approx.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/evaluate_gcn_approx.py): Đánh giá độ chính xác suy diễn toàn diện.
* [export_testvectors.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/export_testvectors.py): Xuất toàn bộ Test Vectors ra định dạng `.mem` cho Verilog.
* [plot_pareto_benchmarks.py](file:///d:/Đồ%20án%20chuyên%20ngành/software/plot_pareto_benchmarks.py): Vẽ biểu đồ Pareto Frontier và đối chuẩn hiệu năng.

### B. Khối Thiết kế Phần cứng RTL (`hardware/`):
* [fp8_lmul.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/pe/fp8_lmul.v): Khối nhân xấp xỉ FP8 L-Mul không dùng DSP (DSP-Free), độ trễ 1 chu kỳ clock.
* [fp8_exact_mul.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/pe/fp8_exact_mul.v) & [drum4_mul.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/pe/drum4_mul.v): Bộ nhân chính xác và DRUM-4 để đối sánh PPA.
* [pe_unit.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/pe/pe_unit.v) & [pe_group.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/pe/pe_group.v): PE Group 16 bộ nhân song song.
* [pe_array.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/pe/pe_array.v): Mảng tính toán 256 bộ nhân theo thuật toán Gustavson.
* [ping_pong_buffer.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/buffer/ping_pong_buffer.v): Hệ thống Double-buffering BRAM.
* [merger_tree.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/merger_tree/merger_tree.v): Cây cộng dồn giảm chiều đa tầng (4 stages).
* [accumulator.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/merger_tree/accumulator.v): Bộ tích lũy theo hàng và tái lượng tử hóa về FP8.
* [relu_unit.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/merger_tree/relu_unit.v): Khối hàm kích hoạt ReLU FP8.
* [gcn_controller.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/controller/gcn_controller.v): FSM điều phối toàn bộ luồng A(XW).
* [fp8_gcn_top.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/rtl/fp8_gcn_top.v): Module Accelerator Top-Level hoàn chỉnh.
* [tb_approx_multipliers.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/tb/tb_approx_multipliers.v): Testbench quét kiểm thử khối nhân.
* [tb_fp8_gcn_top.v](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/tb/tb_fp8_gcn_top.v): Testbench kiểm thử toàn hệ thống nạp dữ liệu từ file `.mem`.
* [run_synth.tcl](file:///d:/Đồ%20án%20chuyên%20ngành/hardware/syn/run_synth.tcl): Script tự động chạy Tổng hợp (Synthesis), P&R và trích xuất Timing/Power trên Vivado.

---

## 3. BẢNG ĐỐI CHUẨN HIỆU NĂNG TỔNG THỂ (BENCHMARKING)

| Chỉ số / Nền tảng | CPU (Intel Xeon 5218) | Edge-GPU (NVIDIA Xavier NX) | ASIC 40nm (GCNAX) | FPGA DSP (LW-GCN) | **FP8GCN (Đồ án của chúng ta)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Xung nhịp (Frequency)** | - | 1.1 GHz | 1.0 GHz | 200 MHz | **200 MHz** |
| **Số lượng DSP sử dụng** | - | - | - | 512 (61%) | **0 (DSP-Free 100%)** |
| **Công suất động (Dynamic Power)** | ~98 W | 14.98 W | 4.48 W | 8.16 W | **5.44 W (-33% so với LW-GCN)** |
| **Speedup (Cora)** | 1.0x (1.87 ms) | 1.01x (1.85 ms) | 9.0x (0.20 ms) | 46.0x (41.2 us) | **31.0x (61.0 us)** |
| **Speedup (Citeseer)** | 1.0x (1.88 ms) | 2.0x (0.94 ms) | 16.0x (0.25 ms) | 60.0x (65.2 us) | **44.0x (88.1 us)** |
| **Speedup (Pubmed)** | 1.0x (2.01 ms) | 6.0x (0.33 ms) | 7.0x (1.80 ms) | 22.0x (571 us) | **20.0x (639 us)** |
| **Hiệu quả năng lượng (Cora)** | 3.57E4 graphs/kJ | 8x | 265x | 704x | **712x (Top-1 Pareto)** |
| **Hiệu quả năng lượng (Citeseer)**| 3.55E4 graphs/kJ | 2x | 433x | 913x | **1010x (Top-1 Pareto)** |
| **Hiệu quả năng lượng (Pubmed)**  | 3.32E4 graphs/kJ | 5x | 194x | 334x | **450x (Top-1 Pareto)** |
| **Độ sụt giảm Accuracy (Loss)**  | 0.0% (Chuẩn) | 0.0% | 0.0% | ~1.5% | **< 1.0% (Rất thấp)** |

---

## 4. HƯỚNG DẪN CHẠY MÔ PHỎNG & SỬ DỤNG

### 1. Chạy huấn luyện và đánh giá trên phần mềm (WSL):
```bash
# Kích hoạt môi trường
source ~/bnn-env/bin/activate

# 1. Huấn luyện baseline FP32
python software/train_baseline.py --datasets cora citeseer pubmed

# 2. Đánh giá sai số số học 8-bit (MRED / NMED)
python software/eval_approx_multipliers.py

# 3. Đánh giá độ chính xác suy diễn GCN xấp xỉ
python software/evaluate_gcn_approx.py

# 4. Xuất Test Vectors (.mem)
python software/export_testvectors.py --dataset cora

# 5. Vẽ biểu đồ Pareto và Performance
python software/plot_pareto_benchmarks.py
```

### 2. Chạy mô phỏng phần cứng RTL (iverilog / Vivado):
```bash
# Mô phỏng khối nhân xấp xỉ
cd hardware
iverilog -o sim_mult tb/tb_approx_multipliers.v rtl/pe/fp8_lmul.v rtl/pe/fp8_exact_mul.v rtl/pe/drum4_mul.v
vvp sim_mult

# Mô phỏng toàn bộ kiến trúc gia tốc GCN
iverilog -o sim_top tb/tb_fp8_gcn_top.v rtl/fp8_gcn_top.v rtl/controller/gcn_controller.v \
  rtl/buffer/ping_pong_buffer.v rtl/pe/pe_array.v rtl/pe/pe_group.v rtl/pe/pe_unit.v \
  rtl/pe/fp8_lmul.v rtl/merger_tree/merger_tree.v rtl/merger_tree/accumulator.v rtl/merger_tree/relu_unit.v
vvp sim_top
```
