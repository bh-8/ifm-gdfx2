import sys
import torch

try:
    if not torch.cuda.is_available():
        raise ImportError("CUDA platform not available!")
    print(f"Hello, World!")

    # TODO..
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
sys.exit(0)
