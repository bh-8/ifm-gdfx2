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

    dataset_train = DF40ImageSequenceDataset("/home/gdfx2/io/train", 8)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 64, shuffle = True)

    # BiLSTM
    # TODO...

except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
