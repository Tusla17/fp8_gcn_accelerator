import numpy as np
import os
import sys

from quantization.fp8 import FP8_E4M3, FP8_E5M2, fp8_bits_to_float
from approx_multipliers.l_mul import l_mul_scalar_bits
from approx_multipliers.drum import drum_mul_scalar
from approx_multipliers.mitchell import mitchell_mul_scalar
from utils.error_metrics import compute_all_error_metrics

def evaluate_fp8_lmul_full_space(fp8_fmt=FP8_E4M3):
    """
    Sweeps all 256 x 256 input bit combinations for FP8 L-Mul vs Exact FP8 Multiplication.
    """
    exact_list = []
    approx_list = []
    
    # 256 bit patterns
    for bx in range(256):
        fx = fp8_bits_to_float(np.uint8(bx), fp8_fmt=fp8_fmt)
        for by in range(256):
            fy = fp8_bits_to_float(np.uint8(by), fp8_fmt=fp8_fmt)
            
            # Exact multiplication
            exact_val = fx * fy
            
            # Hardware L-Mul result
            res_bits = l_mul_scalar_bits(bx, by, fp8_fmt=fp8_fmt)
            approx_val = fp8_bits_to_float(res_bits, fp8_fmt=fp8_fmt)
            
            exact_list.append(exact_val)
            approx_list.append(approx_val)
            
    exact_arr = np.array(exact_list, dtype=np.float64)
    approx_arr = np.array(approx_list, dtype=np.float64)
    
    metrics = compute_all_error_metrics(exact_arr, approx_arr)
    return metrics

def evaluate_int8_multipliers_full_space():
    """
    Sweeps all 256 x 256 combinations for INT8 multipliers (DRUM, Mitchell).
    """
    exact_list = []
    drum4_list = []
    mitchell_list = []
    
    for x in range(-128, 128):
        for y in range(-128, 128):
            exact = x * y
            drum4 = drum_mul_scalar(x, y, k=4)
            mitch = mitchell_mul_scalar(x, y)
            
            exact_list.append(exact)
            drum4_list.append(drum4)
            mitchell_list.append(mitch)
            
    exact_arr = np.array(exact_list, dtype=np.float64)
    drum4_arr = np.array(drum4_list, dtype=np.float64)
    mitch_arr = np.array(mitchell_list, dtype=np.float64)
    
    drum4_metrics = compute_all_error_metrics(exact_arr, drum4_arr)
    mitch_metrics = compute_all_error_metrics(exact_arr, mitch_arr)
    
    return drum4_metrics, mitch_metrics

def main():
    print("="*75)
    print("  EVALUATION OF 8-BIT APPROXIMATE MULTIPLIERS (FULL BIT-SPACE SWEEP)")
    print("="*75)
    
    print("\n1. Evaluating FP8 Multipliers (256 x 256 = 65,536 combinations):")
    fp8_e4m3_metrics = evaluate_fp8_lmul_full_space(FP8_E4M3)
    fp8_e5m2_metrics = evaluate_fp8_lmul_full_space(FP8_E5M2)
    
    print("\n2. Evaluating INT8 Approximate Multipliers (256 x 256 = 65,536 combinations):")
    drum4_metrics, mitch_metrics = evaluate_int8_multipliers_full_space()
    
    print("\n" + "="*75)
    print(f"  {'Multiplier Design':<25} | {'Format':<10} | {'MRED':<12} | {'NMED':<12} | {'Max Error'}")
    print("-" * 75)
    print(f"  {'FP8-L-Mul (E4M3)':<25} | {'FP8-E4M3':<10} | {fp8_e4m3_metrics['MRED']:<12.4e} | {fp8_e4m3_metrics['NMED']:<12.4e} | {fp8_e4m3_metrics['MaxED']:.2f}")
    print(f"  {'FP8-L-Mul (E5M2)':<25} | {'FP8-E5M2':<10} | {fp8_e5m2_metrics['MRED']:<12.4e} | {fp8_e5m2_metrics['NMED']:<12.4e} | {fp8_e5m2_metrics['MaxED']:.2f}")
    print(f"  {'DRUM-4':<25} | {'INT8':<10} | {drum4_metrics['MRED']:<12.4e} | {drum4_metrics['NMED']:<12.4e} | {drum4_metrics['MaxED']:.2f}")
    print(f"  {'Mitchell Logarithmic':<25} | {'INT8':<10} | {mitch_metrics['MRED']:<12.4e} | {mitch_metrics['NMED']:<12.4e} | {mitch_metrics['MaxED']:.2f}")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
