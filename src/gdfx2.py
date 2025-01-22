import argparse
import sys
import torch
import torchvision
import traceback
from isd import DF40ImageSequenceDataset
from models import BiLSTM

#https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9
INPUT_SIZE = 3 * 256 * 256 # total amount of input features per timestamp
SEQUENCE_LENGTH = 8 # 8 is shortest, 32 is longest sequence in DF40 dataset
HIDDEN_SIZE = 3 * 69 # amount of internal lstm features
NUM_LAYERS = 1
BATCH_SIZE = 64
CLASSES = {
    "face_reenact": 0,
    "face_swap": 1
}

try:
    if not torch.cuda.is_available():
        raise ImportError("CUDA platform not available")
    print(f"LSTM parameters:")
    print(f"\tN = {BATCH_SIZE} (batch size)")
    print(f"\tL = {SEQUENCE_LENGTH} (sequence length)")
    print(f"\tCLASSES = {len(CLASSES.keys())}")

    image_transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor()
    ])
    dataset_train = DF40ImageSequenceDataset("/home/gdfx2/io/train", SEQUENCE_LENGTH, image_transformation)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)

    bilstm: BiLSTM = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(CLASSES.keys()))
    print(f"model: {bilstm}")

    for label, images in dataloader_train:
        image_sequence = torch.stack(images)
        input_tensor = image_sequence.view(SEQUENCE_LENGTH, BATCH_SIZE, -1)
        labels = [CLASSES[l] for l in label]

        out = bilstm.forward(input_tensor)
        print(out)
        # BiLSTM
        # TODO...

        break
except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
