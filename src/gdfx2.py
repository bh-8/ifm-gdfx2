import argparse
import sys
import torch
import traceback
from isd import DF40ImageSequenceDataset

#https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9

try:
    if not torch.cuda.is_available():
        raise ImportError("CUDA platform not available")
    print(f"Hello, World!")

    # TODO: training and test datasets erstellen
    dataset_train = DF40ImageSequenceDataset("/home/gdfx2/io/train")
    print(len(dataset_train))
    print(dataset_train[0])
    #print(f"first item is '{dataset_train[1]}'")

    # BiLSTM
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    #print(dataset)
    #for idx, (sample, target) in enumerate(dataset):
    #    print(idx, sample, target)

    # TODO..
except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
