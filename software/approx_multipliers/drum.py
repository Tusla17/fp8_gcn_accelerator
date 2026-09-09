import numpy as np
import torch

def drum_mul_scalar(x_int, y_int, k=4, bit_width=8):
    """
    Dynamic Range Approximate Multiplier (DRUM-k):
    Selects leading k-bits from both operands and multiplies them.
    x_int, y_int: signed 8-bit integers [-128, 127]
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
        
    # Find leading one position
    lx = int(np.floor(np.log2(x_int))) if x_int > 0 else 0
    ly = int(np.floor(np.log2(y_int))) if y_int > 0 else 0
    
    # Extract k bits from lx and ly
    shift_x = max(0, lx - k + 1)
    shift_y = max(0, ly - k + 1)
    
    kx = (x_int >> shift_x)
    ky = (y_int >> shift_y)
    
    # Approximate product
    prod = (kx * ky) << (shift_x + shift_y)
    return sign * prod

def drum_gemm(A, B, k=4):
    """
    INT8 GEMM using DRUM-k multiplication.
    A: [M, K], B: [K, N] (Float tensors/arrays, will be quantized to INT8)
    """
    is_torch = isinstance(A, torch.Tensor)
    device = A.device if is_torch else None
    
    A_np = A.detach().cpu().numpy() if is_torch else np.asarray(A)
    B_np = B.detach().cpu().numpy() if is_torch else np.asarray(B)
    
    # Scale to INT8 [-127, 127]
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
                acc += drum_mul_scalar(A_int[i, p], B_int[p, j], k=k)
            out_int[i, j] = acc
            
    out_float = out_int / (scale_a * scale_b)
    if is_torch:
        return torch.from_numpy(out_float).to(device=device, dtype=torch.float32)
    return out_float
