import numpy as np
import torch
from quantization.fp8 import FP8_E4M3, FP8_E5M2, float_to_fp8_bits, fp8_bits_to_float

def get_l_factor(m):
    """
    Piecewise parameter l(m) for L-Mul algorithm:
    l(m) = m if m <= 3, 3 if m = 4, 4 if m >= 4
    """
    if m <= 3:
        return m
    elif m == 4:
        return 3
    else:
        return 4

def l_mul_scalar_bits(x_byte, y_byte, fp8_fmt=FP8_E4M3):
    """
    Bit-level simulation of FP8-L-Mul hardware multiplier.
    Directly reflects the Verilog logic in Figure 2(c) of the paper:
    1. Sign: Sx ^ Sy
    2. Exponent: Ex + Ey - bias
    3. Mantissa: 1 + mx + my + 2^(-l(m))
    """
    if x_byte == 0 or y_byte == 0:
        return 0
        
    sign_x = (x_byte >> (fp8_fmt.exp_bits + fp8_fmt.man_bits)) & 1
    sign_y = (y_byte >> (fp8_fmt.exp_bits + fp8_fmt.man_bits)) & 1
    sign_res = sign_x ^ sign_y
    
    exp_mask = ((1 << fp8_fmt.exp_bits) - 1)
    man_mask = (1 << fp8_fmt.man_bits) - 1
    
    exp_x = (x_byte >> fp8_fmt.man_bits) & exp_mask
    exp_y = (y_byte >> fp8_fmt.man_bits) & exp_mask
    
    man_x = x_byte & man_mask
    man_y = y_byte & man_mask
    
    if exp_x == 0 or exp_y == 0:
        # Zero or subnormal handling
        return 0
        
    # Unbiased exponents
    E_unbiased = int(exp_x) + int(exp_y) - fp8_fmt.bias
    
    # Mantissa fraction in fixed-point (with implicit 1: M = 2^m + man)
    # L-Mul formula: M_res = 1 + mx + my + 2^(-l(m))
    # In integer units scaled by 2^m:
    # 2^m * M_res = 2^m + man_x + man_y + 2^(m - l(m))
    l_m = get_l_factor(fp8_fmt.man_bits)
    l_offset = 1 << (fp8_fmt.man_bits - l_m) if (fp8_fmt.man_bits >= l_m) else 0
    
    # Mantissa sum: (1 << m) + man_x + man_y + l_offset
    man_sum = (1 << fp8_fmt.man_bits) + int(man_x) + int(man_y) + l_offset
    
    # Check for mantissa overflow (if sum >= 2^(m+1))
    if man_sum >= (1 << (fp8_fmt.man_bits + 1)):
        # Shift right by 1 and increment exponent
        man_sum = man_sum >> 1
        E_unbiased += 1
        
    # Extract new mantissa bits (remove leading 1)
    new_man = man_sum & man_mask
    
    # Handle exponent saturation
    if E_unbiased >= fp8_fmt.max_normal_exp:
        new_exp = fp8_fmt.max_normal_exp
        new_man = fp8_fmt.max_man
    elif E_unbiased <= 0:
        return 0
    else:
        new_exp = E_unbiased
        
    # Pack result
    res_byte = (sign_res << (fp8_fmt.exp_bits + fp8_fmt.man_bits)) | \
               (new_exp << fp8_fmt.man_bits) | \
               new_man
               
    return np.uint8(res_byte)

def l_mul_fp8(x_float, y_float, fp8_fmt=FP8_E4M3):
    """
    Vectorized L-Mul on float tensors / arrays.
    Simulates FP8 Quantization -> FP8 L-Mul -> Float Output.
    """
    is_torch = isinstance(x_float, torch.Tensor)
    device = x_float.device if is_torch else None
    
    x_np = x_float.detach().cpu().numpy() if is_torch else np.asarray(x_float, dtype=np.float32)
    y_np = y_float.detach().cpu().numpy() if is_torch else np.asarray(y_float, dtype=np.float32)
    
    # Quantize to FP8 bits
    x_bits = float_to_fp8_bits(x_np, fp8_fmt=fp8_fmt)
    y_bits = float_to_fp8_bits(y_np, fp8_fmt=fp8_fmt)
    
    # Vectorized bit-level L-Mul
    sign_mask = 1 << (fp8_fmt.exp_bits + fp8_fmt.man_bits)
    exp_mask = ((1 << fp8_fmt.exp_bits) - 1) << fp8_fmt.man_bits
    man_mask = (1 << fp8_fmt.man_bits) - 1
    
    sign_x = (x_bits & sign_mask) != 0
    sign_y = (y_bits & sign_mask) != 0
    sign_res = np.logical_xor(sign_x, sign_y).astype(np.uint8)
    
    exp_x = (x_bits & exp_mask) >> fp8_fmt.man_bits
    exp_y = (y_bits & exp_mask) >> fp8_fmt.man_bits
    
    man_x = x_bits & man_mask
    man_y = y_bits & man_mask
    
    zero_mask = (exp_x == 0) | (exp_y == 0) | (x_bits == 0) | (y_bits == 0)
    
    l_m = get_l_factor(fp8_fmt.man_bits)
    l_offset = 1 << (fp8_fmt.man_bits - l_m) if (fp8_fmt.man_bits >= l_m) else 0
    
    # Mantissa addition
    man_sum = (1 << fp8_fmt.man_bits) + man_x.astype(np.int32) + man_y.astype(np.int32) + l_offset
    E_unbiased = exp_x.astype(np.int32) + exp_y.astype(np.int32) - fp8_fmt.bias
    
    # Overflow normalization
    overflow = man_sum >= (1 << (fp8_fmt.man_bits + 1))
    man_sum[overflow] = man_sum[overflow] >> 1
    E_unbiased[overflow] += 1
    
    new_man = (man_sum & man_mask).astype(np.uint8)
    new_exp = np.clip(E_unbiased, 0, fp8_fmt.max_normal_exp).astype(np.uint8)
    
    underflow = E_unbiased <= 0
    zero_mask = zero_mask | underflow
    
    res_bits = (sign_res << (fp8_fmt.exp_bits + fp8_fmt.man_bits)) | \
               (new_exp << fp8_fmt.man_bits) | \
               new_man
               
    res_bits[zero_mask] = 0
    
    # Decode back to float
    out_float = fp8_bits_to_float(res_bits, fp8_fmt=fp8_fmt)
    
    if is_torch:
        return torch.from_numpy(out_float).to(device=device, dtype=torch.float32)
    return out_float

def l_mul_gemm(A, B, fp8_fmt=FP8_E4M3):
    """
    Performs Matrix Multiplication A x B using FP8 L-Mul for all element-wise multiplications.
    A: [M, K]
    B: [K, N]
    Output: [M, N]
    """
    is_torch = isinstance(A, torch.Tensor)
    device = A.device if is_torch else None
    
    A_np = A.detach().cpu().numpy() if is_torch else np.asarray(A)
    B_np = B.detach().cpu().numpy() if is_torch else np.asarray(B)
    
    # A: [M, K, 1], B: [1, K, N]
    A_expanded = np.expand_dims(A_np, axis=2) # [M, K, 1]
    B_expanded = np.expand_dims(B_np.T, axis=0) # [1, N, K] -> transpose for broadcasting
    
    M, K = A_np.shape
    _, N = B_np.shape
    
    # Element-wise L-Mul over K dimension
    out = np.zeros((M, N), dtype=np.float32)
    
    # Block-wise or loop-wise multiplication to conserve RAM
    for i in range(M):
        # A_row: [1, K], B: [K, N]
        a_row = A_np[i:i+1, :] # [1, K]
        # Brodcast a_row and B columns
        prod = l_mul_fp8(a_row.T, B_np, fp8_fmt=fp8_fmt) # [K, N]
        out[i, :] = np.sum(prod, axis=0)
        
    if is_torch:
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)
    return out
