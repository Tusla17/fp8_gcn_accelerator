import os
import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models.gcn import GCN2Layer
from utils.data_loader import load_planetoid_dataset
from quantization.fp8 import FP8_E4M3, FP8_E5M2, quantize_tensor_fp8
from approx_multipliers.l_mul import l_mul_gemm
from approx_multipliers.drum import drum_gemm
from approx_multipliers.mitchell import mitchell_gemm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def evaluate_model_approx(dataset_name="cora", checkpoint_dir=None, data_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
        
    # 1. Load Data
    data = load_planetoid_dataset(dataset_name=dataset_name, root_dir=data_dir)
    features = data['features']
    adj_norm = data['adj_norm']
    labels = data['labels']
    test_mask = data['test_mask']
    
    # 2. Load Checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"gcn_{dataset_name.lower()}_fp32.pth")
    if not os.path.exists(ckpt_path):
        print(f"[-] Checkpoint not found at {ckpt_path}. Auto-training baseline for {dataset_name.upper()}...")
        from train_baseline import run_experiment
        run_experiment(dataset_name=dataset_name, save_dir=checkpoint_dir, device="cuda" if torch.cuda.is_available() else "cpu")
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    w0 = ckpt['state_dict']['conv1.weight']
    w1 = ckpt['state_dict']['conv2.weight']
    
    # ==========================================
    # 1. FP32 Exact Baseline Inference
    # ==========================================
    z0_fp32 = torch.mm(features, w0)
    h1_fp32 = torch.spmm(adj_norm, z0_fp32) if adj_norm.is_sparse else torch.mm(adj_norm, z0_fp32)
    h1_relu_fp32 = F.relu(h1_fp32)
    
    z1_fp32 = torch.mm(h1_relu_fp32, w1)
    h2_fp32 = torch.spmm(adj_norm, z1_fp32) if adj_norm.is_sparse else torch.mm(adj_norm, z1_fp32)
    pred_fp32 = h2_fp32.max(1)[1]
    acc_fp32 = pred_fp32[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    
    # ==========================================
    # 2. FP8 (E4M3) Exact Multiplier
    # ==========================================
    x_q, _, _ = quantize_tensor_fp8(features, FP8_E4M3)
    w0_q, _, _ = quantize_tensor_fp8(w0, FP8_E4M3)
    w1_q, _, _ = quantize_tensor_fp8(w1, FP8_E4M3)
    
    z0_fp8_exact = torch.mm(x_q, w0_q)
    h1_fp8_exact = torch.spmm(adj_norm, z0_fp8_exact)
    h1_relu_fp8 = F.relu(h1_fp8_exact)
    h1_relu_q, _, _ = quantize_tensor_fp8(h1_relu_fp8, FP8_E4M3)
    
    z1_fp8_exact = torch.mm(h1_relu_q, w1_q)
    h2_fp8_exact = torch.spmm(adj_norm, z1_fp8_exact)
    pred_fp8_exact = h2_fp8_exact.max(1)[1]
    acc_fp8_exact = pred_fp8_exact[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    
    # ==========================================
    # 3. FP8 (E4M3) L-Mul Approximate Multiplier
    # ==========================================
    z0_lmul_e4m3 = l_mul_gemm(x_q, w0_q, fp8_fmt=FP8_E4M3)
    h1_lmul_e4m3 = torch.spmm(adj_norm, z0_lmul_e4m3)
    h1_relu_lmul_e4m3 = F.relu(h1_lmul_e4m3)
    h1_relu_lmul_e4m3_q, _, _ = quantize_tensor_fp8(h1_relu_lmul_e4m3, FP8_E4M3)
    
    z1_lmul_e4m3 = l_mul_gemm(h1_relu_lmul_e4m3_q, w1_q, fp8_fmt=FP8_E4M3)
    h2_lmul_e4m3 = torch.spmm(adj_norm, z1_lmul_e4m3)
    pred_lmul_e4m3 = h2_lmul_e4m3.max(1)[1]
    acc_lmul_e4m3 = pred_lmul_e4m3[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    
    # ==========================================
    # 4. FP8 (E5M2) L-Mul Approximate Multiplier
    # ==========================================
    x_q_e5, _, _ = quantize_tensor_fp8(features, FP8_E5M2)
    w0_q_e5, _, _ = quantize_tensor_fp8(w0, FP8_E5M2)
    w1_q_e5, _, _ = quantize_tensor_fp8(w1, FP8_E5M2)
    
    z0_lmul_e5m2 = l_mul_gemm(x_q_e5, w0_q_e5, fp8_fmt=FP8_E5M2)
    h1_lmul_e5m2 = torch.spmm(adj_norm, z0_lmul_e5m2)
    h1_relu_e5 = F.relu(h1_lmul_e5m2)
    h1_relu_e5_q, _, _ = quantize_tensor_fp8(h1_relu_e5, FP8_E5M2)
    
    z1_lmul_e5m2 = l_mul_gemm(h1_relu_e5_q, w1_q_e5, fp8_fmt=FP8_E5M2)
    h2_lmul_e5m2 = torch.spmm(adj_norm, z1_lmul_e5m2)
    pred_lmul_e5m2 = h2_lmul_e5m2.max(1)[1]
    acc_lmul_e5m2 = pred_lmul_e5m2[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    
    # ==========================================
    # 5. INT8 DRUM-4 Approximate Multiplier
    # ==========================================
    z0_drum4 = drum_gemm(features, w0, k=4)
    h1_drum4 = torch.spmm(adj_norm, z0_drum4)
    h1_relu_drum4 = F.relu(h1_drum4)
    
    z1_drum4 = drum_gemm(h1_relu_drum4, w1, k=4)
    h2_drum4 = torch.spmm(adj_norm, z1_drum4)
    pred_drum4 = h2_drum4.max(1)[1]
    acc_drum4 = pred_drum4[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    
    # ==========================================
    # 6. INT8 Mitchell Approximate Multiplier
    # ==========================================
    z0_mitch = mitchell_gemm(features, w0)
    h1_mitch = torch.spmm(adj_norm, z0_mitch)
    h1_relu_mitch = F.relu(h1_mitch)
    
    z1_mitch = mitchell_gemm(h1_relu_mitch, w1)
    h2_mitch = torch.spmm(adj_norm, z1_mitch)
    pred_mitch = h2_mitch.max(1)[1]
    acc_mitch = pred_mitch[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    
    return {
        'dataset': dataset_name.upper(),
        'acc_fp32': acc_fp32 * 100,
        'acc_fp8_exact': acc_fp8_exact * 100,
        'acc_lmul_e4m3': acc_lmul_e4m3 * 100,
        'acc_lmul_e5m2': acc_lmul_e5m2 * 100,
        'acc_drum4': acc_drum4 * 100,
        'acc_mitch': acc_mitch * 100,
        'drop_lmul_e4m3': (acc_fp32 - acc_lmul_e4m3) * 100,
        'drop_lmul_e5m2': (acc_fp32 - acc_lmul_e5m2) * 100,
        'drop_drum4': (acc_fp32 - acc_drum4) * 100,
        'drop_mitch': (acc_fp32 - acc_mitch) * 100
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate GCN Inference with Approximate Multipliers")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"], help="Datasets")
    args = parser.parse_args()
    
    results = []
    for d in args.datasets:
        print(f"\nEvaluating GCN Inference for dataset: {d.upper()}...")
        res = evaluate_model_approx(d)
        if res:
            results.append(res)
            
    print("\n" + "="*85)
    print("  GCN INFERENCE ACCURACY & ACCURACY LOSS ACROSS MULTIPLIER CONFIGURATIONS")
    print("="*85)
    print(f"  {'Dataset':<10} | {'FP32 Exact':<11} | {'FP8 Exact':<11} | {'FP8 L-Mul(E4M3)':<16} | {'INT8 DRUM-4':<13} | {'INT8 Mitchell'}")
    print("-" * 85)
    for r in results:
        print(f"  {r['dataset']:<10} | {r['acc_fp32']:<6.2f}%    | {r['acc_fp8_exact']:<6.2f}%    | {r['acc_lmul_e4m3']:<6.2f}% (-{r['drop_lmul_e4m3']:.2f}%) | {r['acc_drum4']:<6.2f}% (-{r['drop_drum4']:.2f}%) | {r['acc_mitch']:<6.2f}% (-{r['drop_mitch']:.2f}%)")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
