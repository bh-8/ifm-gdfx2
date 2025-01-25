import argparse
import sys
import torch
import torchvision
import traceback
from isd import DF40ImageSequenceDataset
from models import BiLSTM
from alive_progress import alive_bar

#https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865/9
INPUT_SIZE = 3 * 256 * 256 # total amount of input features per timestamp
SEQUENCE_LENGTH = 16 # 8 is shortest, 32 is longest sequence in DF40 dataset
HIDDEN_SIZE = 3 * 69 # amount of internal lstm features
NUM_LAYERS = 1
BATCH_SIZE = 96
EPOCHS = 3
LEARNING_RATE = 0.1
#MOMENTUM = 0.9
CLASSES = {
    "face_reenact": 0,
    "face_swap": 1
}

try:
    if not torch.cuda.is_available():
        raise ImportError("CUDA platform not available")

    image_transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor()
    ])
    dataset_train = DF40ImageSequenceDataset("/home/gdfx2/io", train = True, sequence_length = SEQUENCE_LENGTH, transform = image_transformation)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)

    bilstm: BiLSTM = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(CLASSES.keys()))
    bilstm = bilstm.cuda()

    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(bilstm.parameters(), lr = LEARNING_RATE)
    print(f"(>) model: {bilstm}")

    batches: int = len(dataloader_train) - 1
    with alive_bar(EPOCHS * batches, title=f"BiLSTM {EPOCHS}*{batches}@{BATCH_SIZE}") as pbar:
        loss_running = 0.
        loss_last = 0.

        for e in range(EPOCHS):
            print(f"epoch {e + 1}")
            loss_epoch = 0.

            bilstm.train(True)
            for i, data in enumerate(dataloader_train):
                if i >= batches:
                    break
                #print(f"  batch {i + 1}/{batches}")
                labels, images = data

                image_sequences = torch.stack(images)
                input_tensor = image_sequences.view(SEQUENCE_LENGTH, BATCH_SIZE, -1)
                input_tensor = input_tensor.cuda()
                input_labels = torch.tensor([CLASSES[l] for l in labels], dtype = torch.long)
                input_labels = input_labels.cuda()

                # start with clean gradient
                optim.zero_grad()
                output = bilstm.forward(input_tensor)

                # compute loss and gradients
                loss = lossf(output, input_labels)
                loss.backward()
                
                optim.step() # adjust weights

                # debug
                loss_epoch += loss.item()
                #print(f"    > {loss.item()}")
                pbar(1)
            loss_epoch = loss_epoch / batches
            bilstm.eval()
            print(f"  >>> {loss_epoch}")

except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
