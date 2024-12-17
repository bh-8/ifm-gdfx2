import argparse
import sys
import torch
import torchvision

try:
    if not torch.cuda.is_available():
        raise ImportError("CUDA platform not available")
    print(f"Hello, World!")

    dataset = torchvision.datasets.ImageFolder("/home/gdfx2/io")
    # RNN or LSTM
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    print(dataset[0])
    #for idx, (sample, target) in enumerate(dataset):
    #    print(sample, target)

    # TODO..
except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    sys.exit(1)
sys.exit(0)
