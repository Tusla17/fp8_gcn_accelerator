import numpy as np
import torch

class FP8Format:
    """
    FP8 specification:
    - E4M3: 1 sign, 4 exp (bias=7), 3 mantissa. Max val: ~448 (or 240/448 depending on standard)
    - E5M2: 1 sign, 5 exp (bias=15), 2 mantissa. Max val: ~57344
    """
    def __init__(self, exp_bits=4, man_bits=3):
        self.exp_bits = exp_bits
        self.man_bits = man_bits
        self.bias = (1 << (exp_bits - 1)) - 1  # 2^(e-1) - 1
        
        # Ranges
        self.max_exp = (1 << exp_bits) - 1
        # Max normal number: exp = max_exp - 1 (reserve max_exp for inf/nan) or max_exp for E4M3
        self.max_normal_exp = self.max_exp - 1
        self.max_man = (1 << man_bits) - 1
        
        self.max_val = (2.0 ** (self.max_normal_exp - self.bias)) * (1.0 + (self.max_man / (2.0 ** man_bits)))
        self.min_normal_val = 2.0 ** (1 - self.bias)
        self.min_subnormal_val = (2.0 ** (1 - self.bias)) * (1.0 / (2.0 ** man_bits))
        
    def __repr__(self):
        return f"FP8(E{self.exp_bits}M{self.man_bits}, bias={self.bias}, max_val={self.max_val:.4f})"

# Standard FP8 Formats
FP8_E4M3 = FP8Format(exp_bits=4, man_bits=3) # bias=7, max_val=448.0
FP8_E5M2 = FP8Format(exp_bits=5, man_bits=2) # bias=15, max_val=57344.0

def float_to_fp8_bits(x, fp8_fmt=FP8_E4M3):
    """
    Encodes float32 numpy array or scalar to 8-bit integer representation (uint8).
    Bits: [Sign (1) | Exponent (e) | Mantissa (m)]
    """
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    x_flat = x.flatten()
    
    signs = (x_flat < 0).astype(np.uint8)
    abs_x = np.abs(x_flat)
    
    # Clip to max representable value
    abs_x = np.clip(abs_x, 0.0, fp8_fmt.max_val)
    
    out_bits = np.zeros(abs_x.shape, dtype=np.uint8)
    
    nonzero_mask = abs_x > (fp8_fmt.min_subnormal_val / 2.0)
    if not np.any(nonzero_mask):
        return out_bits.reshape(shape)
        
    nz_x = abs_x[nonzero_mask]
    
    # Exponent calculation
    log2_x = np.floor(np.log2(nz_x))
    exp_unbiased = np.clip(log2_x, 1 - fp8_fmt.bias, fp8_fmt.max_normal_exp - fp8_fmt.bias)
    exp_biased = (exp_unbiased + fp8_fmt.bias).astype(np.uint8)
    
    # Mantissa calculation: x / 2^E - 1
    scale = 2.0 ** exp_unbiased
    frac = (nz_x / scale) - 1.0
    # Quantize fraction to m bits: round(frac * 2^m)
    man_int = np.clip(np.round(frac * (2.0 ** fp8_fmt.man_bits)), 0, fp8_fmt.max_man).astype(np.uint8)
    
    # Handle mantissa overflow (rounding up to next exponent)
    overflow = man_int > fp8_fmt.max_man
    if np.any(overflow):
        exp_biased[overflow] = np.clip(exp_biased[overflow] + 1, 0, fp8_fmt.max_normal_exp)
        man_int[overflow] = 0
        
    # Pack bits: [Sign | Exp | Mantissa]
    packed = (signs[nonzero_mask] << (fp8_fmt.exp_bits + fp8_fmt.man_bits)) | \
             (exp_biased << fp8_fmt.man_bits) | \
             man_int
             
    out_bits[nonzero_mask] = packed
    return out_bits.reshape(shape)

def fp8_bits_to_float(bits, fp8_fmt=FP8_E4M3):
    """
    Decodes 8-bit integer representation (uint8) back to float32.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    shape = bits.shape
    b_flat = bits.flatten()
    
    sign_mask = 1 << (fp8_fmt.exp_bits + fp8_fmt.man_bits)
    exp_mask = ((1 << fp8_fmt.exp_bits) - 1) << fp8_fmt.man_bits
    man_mask = (1 << fp8_fmt.man_bits) - 1
    
    signs = ((b_flat & sign_mask) != 0).astype(np.float32)
    sign_mult = 1.0 - 2.0 * signs  # 0 -> +1.0, 1 -> -1.0
    
    exp_biased = (b_flat & exp_mask) >> fp8_fmt.man_bits
    man_int = b_flat & man_mask
    
    out = np.zeros(b_flat.shape, dtype=np.float32)
    
    # Normal numbers (E > 0)
    normal_mask = exp_biased > 0
    if np.any(normal_mask):
        E = exp_biased[normal_mask].astype(np.float32) - fp8_fmt.bias
        M = 1.0 + (man_int[normal_mask].astype(np.float32) / (2.0 ** fp8_fmt.man_bits))
        out[normal_mask] = sign_mult[normal_mask] * (2.0 ** E) * M
        
    # Subnormal numbers (E == 0, M = 0 + man / 2^m)
    subnormal_mask = (exp_biased == 0) & (man_int > 0)
    if np.any(subnormal_mask):
        E = 1.0 - fp8_fmt.bias
        M = man_int[subnormal_mask].astype(np.float32) / (2.0 ** fp8_fmt.man_bits)
        out[subnormal_mask] = sign_mult[subnormal_mask] * (2.0 ** E) * M
        
    return out.reshape(shape)

def quantize_tensor_fp8(tensor, fp8_fmt=FP8_E4M3, scale=None, percentile=99.9):
    """
    Quantizes a PyTorch tensor to FP8 format with scale factor calibration:
    X_scaled = Clip(X / S)
    X_q = FP8_Decode(FP8_Encode(X_scaled)) * S
    """
    orig_type = tensor.dtype
    orig_device = tensor.device
    
    tensor_np = tensor.detach().cpu().numpy()
    
    # Determine scale factor S
    if scale is None:
        abs_max = np.percentile(np.abs(tensor_np), percentile)
        if abs_max == 0:
            abs_max = 1e-6
        scale = float(abs_max / fp8_fmt.max_val)
        if scale == 0:
            scale = 1.0
            
    # Scale and encode to FP8
    scaled_np = tensor_np / scale
    fp8_bits = float_to_fp8_bits(scaled_np, fp8_fmt=fp8_fmt)
    
    # Decode back and rescale
    quantized_np = fp8_bits_to_float(fp8_bits, fp8_fmt=fp8_fmt) * scale
    
    quantized_tensor = torch.from_numpy(quantized_np).to(dtype=orig_type, device=orig_device)
    return quantized_tensor, fp8_bits, scale
