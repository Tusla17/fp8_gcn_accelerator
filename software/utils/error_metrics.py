import numpy as np

def compute_mred(exact, approx, eps=1e-8):
    """
    Mean Relative Error Distance (MRED):
    MRED = (1/N) * sum( |Exact - Approx| / |Exact| ) for Exact != 0
    """
    exact = np.asarray(exact, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)
    
    mask = np.abs(exact) > eps
    if not np.any(mask):
        return 0.0
        
    red = np.abs(exact[mask] - approx[mask]) / np.abs(exact[mask])
    return float(np.mean(red))

def compute_nmed(exact, approx, max_val=None):
    """
    Normalized Mean Error Distance (NMED):
    NMED = (1 / (N * MaxVal)) * sum( |Exact - Approx| )
    """
    exact = np.asarray(exact, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)
    
    ed = np.abs(exact - approx)
    med = np.mean(ed)
    
    if max_val is None:
        max_val = np.max(np.abs(exact))
        if max_val == 0:
            max_val = 1.0
            
    return float(med / max_val)

def compute_all_error_metrics(exact, approx):
    """
    Computes MRED, NMED, Mean Error Distance (MED), and Max Error Distance.
    """
    exact = np.asarray(exact, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)
    
    ed = np.abs(exact - approx)
    med = float(np.mean(ed))
    max_ed = float(np.max(ed))
    
    mred = compute_mred(exact, approx)
    nmed = compute_nmed(exact, approx)
    
    return {
        'MRED': mred,
        'NMED': nmed,
        'MED': med,
        'MaxED': max_ed
    }
