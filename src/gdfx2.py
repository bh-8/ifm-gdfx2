import argparse
import sys
import torch
import torchvision
import traceback
from pathlib import Path
from isd import DF40ImageSequenceDataset
from models import BiLSTM
from alive_progress import alive_bar

# MOMENTUM?
# BiLSTM: Wie mit den beiden Richtungen umgehen?

parser = argparse.ArgumentParser(prog = "gdfx2", description = "generalizable deepfake detection framework")
parser.add_argument("mode", choices = ["train", "test"], help = "mode of operation")
parser.add_argument("-sl", "--sequence-length", type = int, default = 16, help = "sequence length (8 to 32 for DF40), default is 16")
parser.add_argument("-lr", "--learning-rate", type = float, default = 0.1, help = "learning rate (0 to 1), default is 0.1")
parser.add_argument("-ep", "--epochs", type = int, default = 3, help = "training epochs, default is 3")
parser.add_argument("-bs", "--batch-size", type = int, default = 16, help = "samples per batch, default is 16")
parser.add_argument("-hs", "--hidden-size", type = int, default = 69, help = "internal lstm features, default is 69")
parser.add_argument("-nl", "--num-layers", type = int, default = 1, help = "stacked lstm layers, default is 1")
parser.add_argument("-sm", "--save-model", type = str, default = None, help = "save model after training to given path")
parser.add_argument("-lm", "--load-model", type = str, default = None, help = "load model prior training/for testing from given path")
parser.add_argument("-f", "--force", action = "store_true", help = "force overwriting when using -sm")
args = parser.parse_args()

MODE_TRAIN: bool = args.mode == "train"
MODE_TEST: bool = args.mode == "test"
if MODE_TRAIN == MODE_TEST:
    raise AssertionError("rip")

if args.sequencelength < 8 or args.sequencelength > 32:
    raise AssertionError("sequence length out of range")
if args.learningrate <= 0 or args.learningrate >= 1:
    raise AssertionError("learning rate out of range")
if args.epochs < 1:
    raise AssertionError("epoch size out of range")
if args.batchsize < 1:
    raise AssertionError("batch size out of range")
if args.hiddensize < 1:
    raise AssertionError("hidden size out of range")
if args.numlayers < 1:
    raise AssertionError("num layers out of range")

SAVE_MODEL = None
if args.savemodel:
    SAVE_MODEL: Path = Path(args.savemodel).resolve()
    if SAVE_MODEL.exists() and not args.force:
        raise FileExistsError(f"file '{SAVE_MODEL}' does already exist, use -f to override")
LOAD_MODEL = None
if args.loadmodel:
    LOAD_MODEL: Path = Path(args.loadmodel).resolve()
    if not LOAD_MODEL.exists():
        raise FileNotFoundError(f"file '{LOAD_MODEL} does not exist'")

INPUT_SIZE = 3 * 256 * 256 # total amount of input features per timestamp is equal to size of image tensor
SEQUENCE_LENGTH = args.sequencelength
LEARNING_RATE = args.learningrate
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
HIDDEN_SIZE = args.hiddensize
NUM_LAYERS = args.numlayers
CLASSES = {
    "face_reenact": 0,
    "face_swap": 1
}
if not torch.cuda.is_available():
    raise ImportError("CUDA platform not available")

DF40_TRANSFORMATION = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor()
])

sys.exit(0)

try:
    if MODE_TRAIN:
        print("(>) initializing dataset")
        dataset_train = DF40ImageSequenceDataset("/home/gdfx2/io", train = True, sequence_length = SEQUENCE_LENGTH, transform = DF40_TRANSFORMATION)

        print("(>) setting up dataloader")
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)
        batches: int = len(dataloader_train) - 1 # crash fix - 1

        bilstm: BiLSTM = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(CLASSES.keys()))
        bilstm = bilstm.cuda()
        if LOAD_MODEL:
            print(f"(>) loading model from '{LOAD_MODEL}'")
            bilstm.load_state_dict(torch.load(LOAD_MODEL, weights_only=True))
            bilstm.eval()
            # TODO: load optimizer state_dict

        lossf = torch.nn.CrossEntropyLoss()
        optim = torch.optim.SGD(bilstm.parameters(), lr = LEARNING_RATE)
        print(f"(>) model: {bilstm}")

        with alive_bar(EPOCHS * batches, title=f"BiLSTM {EPOCHS}*{batches}@{BATCH_SIZE}") as pbar:
            loss_running = 0.
            loss_last = 0.

            for e in range(EPOCHS):
                print(f"epoch {e + 1}")
                loss_epoch = 0.

                bilstm.train(True)
                for i, data in enumerate(dataloader_train):
                    if i >= batches:
                        break # crash fix

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

                    # adjust weights
                    optim.step()

                    # sum up error
                    loss_epoch += loss.item()
                    pbar(1)
                loss_epoch = loss_epoch / batches
                bilstm.eval()
                print(f"  >>> {loss_epoch}")
        if SAVE_MODEL:
            print(f"(>) saving model to '{SAVE_MODEL}'")
            torch.save(bilstm.state_dict(), SAVE_MODEL)
        print("(>) done")

    if MODE_TEST:
        pass

except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
