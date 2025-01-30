import argparse
import sys
import torch
import torchvision
import torcheval
from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_auroc
import traceback
from pathlib import Path
from isd import DF40ImageSequenceDataset
from models import BiLSTM
from alive_progress import alive_bar

# MOMENTUM?
# BiLSTM: Wie mit den beiden Richtungen umgehen?
# DropOut?
# Parallelisierung

parser = argparse.ArgumentParser(prog = "gdfx2", description = "generalizable deepfake detection framework")
parser.add_argument("mode", choices = ["train", "test"], help = "mode of operation")

# dataset
parser.add_argument("-sl", "--sequence-length", type = int, default = 16, help = "sequence length (8 to 32 for DF40), default is 16")

# training
parser.add_argument("-bs", "--batch-size", type = int, default = 10, help = "samples per batch, default is 10")
parser.add_argument("-ep", "--epochs", type = int, default = 1, help = "training epochs, default is 1")
parser.add_argument("-lr", "--learning-rate", type = float, default = 0.01, help = "learning rate (0 to 1), default is 0.01")

# model
parser.add_argument("-hs", "--hidden-size", type = int, default = 256, help = "internal lstm features (net width), default is 256")
parser.add_argument("-nl", "--num-layers", type = int, default = 3, help = "stacked lstm layers (net depth), default is 3")

# io actions
parser.add_argument("-sm", "--save-model", type = str, default = None, help = "save model after training to given path")
parser.add_argument("-lm", "--load-model", type = str, default = None, help = "load model prior training/for testing from given path")
parser.add_argument("-f", "--force", action = "store_true", help = "force overwriting when using -sm")

args = parser.parse_args()

MODE_TRAIN: bool = args.mode == "train"
MODE_TEST: bool = args.mode == "test"
if MODE_TRAIN == MODE_TEST:
    raise AssertionError("rip")

if args.sequence_length < 8 or args.sequence_length > 32:
    raise AssertionError("sequence length out of range")
if args.batch_size < 1:
    raise AssertionError("batch size out of range")
if args.epochs < 1:
    raise AssertionError("epoch size out of range")
if args.learning_rate <= 0 or args.learning_rate >= 1:
    raise AssertionError("learning rate out of range")
if args.hidden_size < 1:
    raise AssertionError("hidden size out of range")
if args.num_layers < 1:
    raise AssertionError("num layers out of range")

SAVE_MODEL = None
if args.save_model:
    SAVE_MODEL: Path = Path(args.save_model).resolve()
    if SAVE_MODEL.exists() and not args.force:
        raise FileExistsError(f"file '{SAVE_MODEL}' does already exist, use -f to override")
LOAD_MODEL = None
if args.load_model:
    LOAD_MODEL: Path = Path(args.load_model).resolve()
    if not LOAD_MODEL.exists():
        raise FileNotFoundError(f"file '{LOAD_MODEL} does not exist'")

INPUT_SIZE = 3 * 256 * 256 # total amount of input features per timestamp is equal to size of image tensor
SEQUENCE_LENGTH = args.sequence_length
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
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

try:
    print("(>) parameters")
    print(f"\tinput size: {INPUT_SIZE}")
    print(f"\tsequence length: {SEQUENCE_LENGTH}")
    print(f"\tbatch size: {BATCH_SIZE}")
    print(f"\tepochs: {EPOCHS}")
    print(f"\tlearning rate: {LEARNING_RATE}")
    print(f"\thidden size: {HIDDEN_SIZE}")
    print(f"\tnum layers: {NUM_LAYERS}")
    print(f"\tclasses: {len(CLASSES.keys())}")

    print("(>) initializing dataset")
    dataset_df40 = DF40ImageSequenceDataset("/home/gdfx2/io", train = MODE_TRAIN, sequence_length = SEQUENCE_LENGTH, transform = DF40_TRANSFORMATION)
    print(f"\titems: {len(dataset_df40)}")

    print("(>) setting up dataloader")
    dataloader_df40 = torch.utils.data.DataLoader(dataset_df40, batch_size = BATCH_SIZE, shuffle = True)
    batches: int = len(dataloader_df40) - 1 # crash fix - 1
    print(f"\tbatches per epoch: {batches}")
    print(f"\ttotal batches: {EPOCHS * batches}")

    bilstm: BiLSTM = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(CLASSES.keys()))
    bilstm = bilstm.cuda()
    if LOAD_MODEL:
        print(f"(>) loading model from '{LOAD_MODEL}'")
        bilstm.load_state_dict(torch.load(LOAD_MODEL, weights_only=True))
        bilstm.eval()
        # TODO: load optimizer state_dict
    print(f"(>) model: {bilstm}")

    if MODE_TRAIN:
        lossf = torch.nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(bilstm.parameters(), lr = LEARNING_RATE)

        with alive_bar(EPOCHS * batches, title=f"BiLSTM train {EPOCHS}*{batches}@{BATCH_SIZE}") as pbar:
            for e in range(EPOCHS):
                print(f"(>) === epoch {e + 1} ===")
                epoch_total: int = 0
                epoch_correct: int = 0
                epoch_loss: float = 0.
                epoch_accuracy: float = 0.
                epoch_precision: float = 0.
                epoch_recall: float = 0.
                epoch_f1: float = 0.
                epoch_auroc: float = 0.

                bilstm.train(True)
                for i, data in enumerate(dataloader_df40):
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
                    output_tensor = bilstm.forward(input_tensor)

                    # compute loss, gradients and backprop
                    loss = lossf(output_tensor, input_labels)
                    loss.backward()

                    # adjust weights
                    optim.step()

                    # current predictions
                    _, predicted_labels = torch.max(output_tensor, 1)

                    # stats
                    epoch_total += BATCH_SIZE
                    epoch_correct += (predicted_labels == input_labels).sum().item()
                    epoch_loss += loss.item()
                    epoch_accuracy += multiclass_accuracy(predicted_labels, input_labels, num_classes = len(CLASSES.keys())).item()
                    epoch_precision += multiclass_precision(predicted_labels, input_labels, num_classes = len(CLASSES.keys())).item()
                    epoch_recall += multiclass_recall(predicted_labels, input_labels, num_classes = len(CLASSES.keys())).item()
                    epoch_f1 += multiclass_f1_score(predicted_labels, input_labels, num_classes = len(CLASSES.keys())).item()
                    epoch_auroc += multiclass_auroc(torch.nn.functional.softmax(output_tensor, dim=1), input_labels, num_classes = len(CLASSES.keys())).item()

                    pbar(1)
                bilstm.eval()

                #print(f"\tavg accuracy: {round(accuracy * 100, 3)}%")
                epoch_loss: float = epoch_loss / batches # calculated per batch
                epoch_accuracy: float = epoch_accuracy / batches # calculated per batch
                epoch_precision: float = epoch_precision / batches # calculated per batch
                epoch_recall: float = epoch_recall / batches # calculated per batch
                epoch_f1: float = epoch_f1 / batches # calculated per batch
                epoch_auroc: float = epoch_auroc / batches # calculated per batch
                print(f"\tcorrect: {epoch_correct}/{epoch_total}")
                print(f"\tavg loss: {round(epoch_loss, 5)}")
                print(f"\tavg accuracy: {round(epoch_accuracy, 5)}")
                print(f"\tavg precision: {round(epoch_precision, 5)}")
                print(f"\tavg recall: {round(epoch_recall, 5)}")
                print(f"\tavg f measure: {round(epoch_f1, 5)}")
                print(f"\tavg area under roc: {round(epoch_auroc, 5)}")
        if SAVE_MODEL:
            print(f"(>) saving model to '{SAVE_MODEL}'")
            torch.save(bilstm.state_dict(), SAVE_MODEL)
        print("(>) done")

    if MODE_TEST:
        bilstm.eval()

        with alive_bar(batches, title=f"BiLSTM test {batches}@{BATCH_SIZE}") as pbar:
            for i, data in enumerate(dataloader_df40):
                if i >= batches:
                    break # crash fix

                labels, images = data

                image_sequences = torch.stack(images)
                input_tensor = image_sequences.view(SEQUENCE_LENGTH, BATCH_SIZE, -1)
                input_tensor = input_tensor.cuda()
                input_labels = torch.tensor([CLASSES[l] for l in labels], dtype = torch.long)
                input_labels = input_labels.cuda()

                with torch.no_grad():
                    output_tensor = bilstm.forward(input_tensor)
                    _, predicted_labels = torch.max(output_tensor, 1)

                # TODO: see pytorch ignite / torch.sigmoid?

                pbar(1)

except Exception as e:
    print(f"[!] ERROR: {e} [!]")
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
