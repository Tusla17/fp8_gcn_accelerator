import numpy as np
import torch

def mitchell_mul_scalar(x_int, y_int):
    """
    Mitchell's Logarithmic Multiplier for 8-bit unsigned/signed integers:
    A * B approx 2^(k1 + k2) * (1 + x1 + x2)
    where A = 2^k1 * (1 + x1), B = 2^k2 * (1 + x2)
    """
    if x_int == 0 or y_int == 0:
        return 0
        
    sign = 1
    if x_int < 0:
        sign = -sign
        x_int = -x_int
    if y_int < 0:
        sign = -sign
        y_int = -y_int
        
    k1 = int(np.floor(np.log2(x_int)))
    k2 = int(np.floor(np.log2(y_int)))
    
    # Fractions: x1 = (A - 2^k1) / 2^k1
    # Scaled fraction sum: (x1 + x2) * 2^(k1 + k2)
    # Mitchell formula: if x1 + x2 < 1 => 2^(k1 + k2) + x1*2^(k1+k2) + x2*2^(k1+k2)
    #                   if x1 + x2 >= 1 => 2^(k1 + k2 + 1) + (x1 + x2 - 1)*2^(k1 + k2)
    
    rem1 = x_int - (1 << k1)
    rem2 = y_int - (1 << k2)
    
    # Scaled integer arithmetic
    if (rem1 << k2) + (rem2 << k1) < (1 << (k1 + k2)):
        approx = (1 << (k1 + k2)) + (rem1 << k2) + (rem2 << k1)
    else:
        approx = (1 << (k1 + k2 + 1)) + (rem1 << k2) + (rem2 << k1) - (1 << (k1 + k2))
        
    return sign * int(approx)

def mitchell_gemm(A, B):
    """
    INT8 Mitchell GEMM.
    """
    is_torch = isinstance(A, torch.Tensor)
    device = A.device if is_torch else None
    
    A_np = A.detach().cpu().numpy() if is_torch else np.asarray(A)
    B_np = B.detach().cpu().numpy() if is_torch else np.asarray(B)
    
    scale_a = 127.0 / (np.max(np.abs(A_np)) + 1e-6)
    scale_b = 127.0 / (np.max(np.abs(B_np)) + 1e-6)
    
    A_int = np.clip(np.round(A_np * scale_a), -127, 127).astype(np.int32)
    B_int = np.clip(np.round(B_np * scale_b), -127, 127).astype(np.int32)
    
    M, K = A_np.shape
    _, N = B_np.shape
    out_int = np.zeros((M, N), dtype=np.float32)
    
    for i in range(M):
        for j in range(N):
            acc = 0
            for p in range(K):
                acc += mitchell_mul_scalar(A_int[i, p], B_int[p, j])
            out_int[i, j] = acc
            
    out_float = out_int / (scale_a * scale_b)
    if is_torch:
        return torch.from_numpy(out_float).to(device=device, dtype=torch.float32)
    return out_float
