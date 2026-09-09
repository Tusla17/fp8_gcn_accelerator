import torch
import torch_geometric
import scipy
import numpy as np

print("="*60)
print("WSL Environment Verification:")
print(f"  PyTorch Version      : {torch.__version__}")
print(f"  PyTorch Geometric    : {torch_geometric.__version__}")
print(f"  Scipy Version        : {scipy.__version__}")
print(f"  NumPy Version        : {np.__version__}")
print(f"  CUDA GPU Available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Device Name      : {torch.cuda.get_device_name(0)}")
print("="*60)
