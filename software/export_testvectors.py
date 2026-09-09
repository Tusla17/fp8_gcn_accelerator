import os
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from quantization.fp8 import FP8_E4M3, float_to_fp8_bits, fp8_bits_to_float, quantize_tensor_fp8
from utils.data_loader import load_planetoid_dataset
from approx_multipliers.l_mul import l_mul_gemm, l_mul_fp8

def write_mem_file_hex(filename, data_bits, comments=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        if comments:
            f.write(f"// {comments}\n")
        for val in data_bits:
            if isinstance(val, (int, np.integer)):
                f.write(f"{val:02X}\n")
            elif isinstance(val, (list, np.ndarray)):
                hex_str = "".join([f"{b:02X}" for b in val])
                f.write(f"{hex_str}\n")
    print(f"  [+] Created: {filename} ({len(data_bits)} entries)")

def export_testvectors_for_dataset(dataset_name="cora", num_subgraph_nodes=32):
    # Determine base directory of software module
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(curr_dir, "testvectors")
    checkpoint_dir = os.path.join(curr_dir, "checkpoints")
    data_dir = os.path.join(curr_dir, "data")
    
    print(f"\n{'='*70}")
    print(f"  EXPORTING TEST VECTORS FOR DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # 1. Load Data
    data = load_planetoid_dataset(dataset_name=dataset_name, root_dir=data_dir)
    features = data['features']
    adj_norm_coo = data['adj_norm_coo']
    adj_csr = adj_norm_coo.tocsr()
    labels = data['labels']
    
    # 2. Load Weights
    ckpt_path = os.path.join(checkpoint_dir, f"gcn_{dataset_name.lower()}_fp32.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    w0 = ckpt['state_dict']['conv1.weight']
    w1 = ckpt['state_dict']['conv2.weight']
    
    # Quantize inputs and weights to FP8 E4M3
    x_q, x_bits, s_x = quantize_tensor_fp8(features, FP8_E4M3)
    w0_q, w0_bits, s_w0 = quantize_tensor_fp8(w0, FP8_E4M3)
    w1_q, w1_bits, s_w1 = quantize_tensor_fp8(w1, FP8_E4M3)
    
    # 3. Compute Golden Intermediate and Output using FP8-L-Mul
    print("  Computing bit-accurate Golden Outputs with FP8-L-Mul...")
    z0 = l_mul_gemm(x_q, w0_q, fp8_fmt=FP8_E4M3) # [N, 16]
    h1 = adj_csr.dot(z0.numpy())                  # [N, 16]
    h1_relu = np.maximum(0, h1)                   # ReLU
    h1_relu_tensor = torch.from_numpy(h1_relu)
    h1_q, h1_bits, s_h1 = quantize_tensor_fp8(h1_relu_tensor, FP8_E4M3)
    
    z1 = l_mul_gemm(h1_q, w1_q, fp8_fmt=FP8_E4M3) # [N, num_classes]
    h2 = adj_csr.dot(z1.numpy())                  # [N, num_classes]
    pred = np.argmax(h2, axis=1)
    
    # Export full dataset vectors
    ds_dir = os.path.join(output_dir, dataset_name.lower())
    os.makedirs(ds_dir, exist_ok=True)
    
    # 4. Export CSR Adjacency Matrix
    adj_val_bits = float_to_fp8_bits(adj_csr.data, fp8_fmt=FP8_E4M3)
    write_mem_file_hex(os.path.join(ds_dir, "adj_row_ptr.mem"), adj_csr.indptr, comments="CSR row pointers")
    write_mem_file_hex(os.path.join(ds_dir, "adj_col_idx.mem"), adj_csr.indices, comments="CSR column indices")
    write_mem_file_hex(os.path.join(ds_dir, "adj_val_fp8.mem"), adj_val_bits, comments="CSR non-zero values in FP8 E4M3")
    
    # 5. Export Feature and Weight Matrices
    write_mem_file_hex(os.path.join(ds_dir, "features_fp8.mem"), x_bits.flatten(), comments="Features in FP8")
    write_mem_file_hex(os.path.join(ds_dir, "w0_fp8.mem"), w0_bits.flatten(), comments="W0 in FP8")
    write_mem_file_hex(os.path.join(ds_dir, "w1_fp8.mem"), w1_bits.flatten(), comments="W1 in FP8")
    
    # 6. Export Golden Intermediate Values
    z0_bits = float_to_fp8_bits(z0.numpy(), fp8_fmt=FP8_E4M3)
    z1_bits = float_to_fp8_bits(z1.numpy(), fp8_fmt=FP8_E4M3)
    write_mem_file_hex(os.path.join(ds_dir, "golden_z0_fp8.mem"), z0_bits.flatten(), comments="Golden Z0 = X * W0 (FP8)")
    write_mem_file_hex(os.path.join(ds_dir, "golden_h1_relu_fp8.mem"), h1_bits.flatten(), comments="Golden H1 = ReLU(A * Z0) (FP8)")
    write_mem_file_hex(os.path.join(ds_dir, "golden_z1_fp8.mem"), z1_bits.flatten(), comments="Golden Z1 = H1 * W1 (FP8)")
    write_mem_file_hex(os.path.join(ds_dir, "golden_predictions.mem"), pred, comments="Golden Node Class Predictions")

    # 7. Export Small Subgraph for Quick RTL Simulation (32 Nodes)
    sub_dir = os.path.join(output_dir, f"{dataset_name.lower()}_subgraph_{num_subgraph_nodes}")
    os.makedirs(sub_dir, exist_ok=True)
    
    sub_x_bits = x_bits[:num_subgraph_nodes, :]
    sub_adj = adj_csr[:num_subgraph_nodes, :num_subgraph_nodes].tocsr()
    sub_adj_val_bits = float_to_fp8_bits(sub_adj.data, fp8_fmt=FP8_E4M3)
    
    sub_z0_bits = z0_bits[:num_subgraph_nodes, :]
    sub_h1_bits = h1_bits[:num_subgraph_nodes, :]
    sub_z1_bits = z1_bits[:num_subgraph_nodes, :]
    sub_pred = pred[:num_subgraph_nodes]
    
    write_mem_file_hex(os.path.join(sub_dir, "sub_adj_row_ptr.mem"), sub_adj.indptr, comments="Subgraph CSR row pointers")
    write_mem_file_hex(os.path.join(sub_dir, "sub_adj_col_idx.mem"), sub_adj.indices, comments="Subgraph CSR col indices")
    write_mem_file_hex(os.path.join(sub_dir, "sub_adj_val_fp8.mem"), sub_adj_val_bits, comments="Subgraph CSR values (FP8)")
    write_mem_file_hex(os.path.join(sub_dir, "sub_features_fp8.mem"), sub_x_bits.flatten(), comments="Subgraph features (FP8)")
    write_mem_file_hex(os.path.join(sub_dir, "w0_fp8.mem"), w0_bits.flatten(), comments="W0 weights (FP8)")
    write_mem_file_hex(os.path.join(sub_dir, "w1_fp8.mem"), w1_bits.flatten(), comments="W1 weights (FP8)")
    write_mem_file_hex(os.path.join(sub_dir, "golden_sub_pred.mem"), sub_pred, comments="Golden Subgraph Predictions")

    print(f"\n  [SUCCESS] All Test Vectors exported to: {ds_dir} and {sub_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Test Vectors for FPGA Simulation")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name")
    parser.add_argument("--subgraph_nodes", type=int, default=32, help="Number of nodes for small testbench")
    args = parser.parse_args()
    
    export_testvectors_for_dataset(dataset_name=args.dataset, num_subgraph_nodes=args.subgraph_nodes)
