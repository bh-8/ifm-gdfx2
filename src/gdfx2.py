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

# Adaptive LR?

parser = argparse.ArgumentParser(prog = "gdfx2", description = "generalizable deepfake detection framework")
parser.add_argument("mode", choices = ["train", "test"], help = "mode of operation")

# dataset
parser.add_argument("-ds", "--df40-dataset", type = str, default = "io", help = "path to df40 dataset, default is 'io'")
parser.add_argument("-sl", "--sequence-length", type = int, default = 16, help = "sequence length (8 to 32 for DF40), default is 16")

# training
parser.add_argument("-bs", "--batch-size", type = int, default = 10, help = "samples per batch, default is 10")
parser.add_argument("-mb", "--max-batches", type = int, default = None, help = "cap the amount of batches per epoch")
parser.add_argument("-ep", "--epochs", type = int, default = 1, help = "training epochs, default is 1")
parser.add_argument("-lr", "--learning-rate", type = float, default = 1e-6, help = "learning rate (0 to 1), default is 1e-6")
parser.add_argument("-wd", "--weight-decay", type = float, default = 0.03, help = "weight decay (0 to 1), default is 0.03")

# model
parser.add_argument("-hs", "--hidden-size", type = int, default = 256, help = "internal lstm features (net width), default is 256")
parser.add_argument("-nl", "--num-layers", type = int, default = 3, help = "stacked lstm layers (net depth), default is 3")
parser.add_argument("-do", "--dropout", type = float, default = 0.3, help = "lstm dropout (0 to 1), default is 0.3")

# io actions
parser.add_argument("-sm", "--save-model", type = str, default = None, help = "save model after training to given path")
parser.add_argument("-lm", "--load-model", type = str, default = None, help = "load model prior training/for testing from given path")
parser.add_argument("-f", "--force", action = "store_true", help = "force overwriting when using -sm")

args = parser.parse_args()

MODE_TRAIN: bool = args.mode == "train"
MODE_TEST: bool = args.mode == "test"
if not (MODE_TRAIN or MODE_TEST):
    raise AssertionError("rip")

if args.sequence_length < 8 or args.sequence_length > 32:
    raise AssertionError("sequence length out of range")
if args.batch_size < 1:
    raise AssertionError("batch size out of range")
if args.max_batches and args.max_batches < 1:
    raise AssertionError("max batches out of range")
if args.epochs < 1:
    raise AssertionError("epoch size out of range")
if args.learning_rate <= 0 or args.learning_rate >= 1:
    raise AssertionError("learning rate out of range")
if args.weight_decay < 0 or args.weight_decay >= 1:
    raise AssertionError("weight decay out of range")
if args.hidden_size < 1:
    raise AssertionError("hidden size out of range")
if args.num_layers < 1:
    raise AssertionError("num layers out of range")
if args.dropout < 0 or args.dropout >= 1:
    raise AssertionError("dropout out of range")

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
WEIGHT_DECAY = args.weight_decay
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
DROPOUT = args.dropout
CLASSES = {
    "face_reenact": 0,
    "face_swap": 1
}
if not torch.cuda.is_available():
    raise ImportError("CUDA platform not available")
device = torch.device("cuda")
print(f"(>) cuda devices: {torch.cuda.device_count()}")

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
    print(f"\tweight decay: {WEIGHT_DECAY}")
    print(f"\thidden size: {HIDDEN_SIZE}")
    print(f"\tnum layers: {NUM_LAYERS}")
    print(f"\tlstm dropout: {DROPOUT}")
    print(f"\tclasses: {len(CLASSES.keys())}")

    print("(>) initializing dataset")
    dataset_df40 = DF40ImageSequenceDataset(args.df40_dataset, train = MODE_TRAIN, sequence_length = SEQUENCE_LENGTH, transform = DF40_TRANSFORMATION)
    print(f"\titems: {len(dataset_df40)}")

    print("(>) setting up dataloader")
    dataloader_df40 = torch.utils.data.DataLoader(dataset_df40, batch_size = BATCH_SIZE, shuffle = True)
    batches: int = min(len(dataloader_df40) - 1, args.max_batches) if args.max_batches else len(dataloader_df40) - 1 # crash fix - 1
    print(f"\tbatches per epoch: {batches}")
    print(f"\ttotal batches: {EPOCHS * batches}")

    bilstm: BiLSTM = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(CLASSES.keys()))
    bilstm = torch.nn.DataParallel(bilstm)
    bilstm = bilstm.to(device)
    if LOAD_MODEL:
        print(f"(>) loading model from '{LOAD_MODEL}'")
        bilstm.load_state_dict(torch.load(LOAD_MODEL, weights_only=True))
        bilstm.eval()
        # TODO: load optimizer state_dict
    print(f"(>) model: {bilstm}")

    if MODE_TRAIN:
        lossf = torch.nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(bilstm.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        with alive_bar(EPOCHS * batches, title=f"BiLSTM train {EPOCHS}*{batches}@{BATCH_SIZE}") as pbar:
            for e in range(EPOCHS):
                print(f"(>) === epoch {e + 1} ===")
                epoch_total: int = 0
                epoch_correct: int = 0
                loss_values: list[float] = []
                accuracy_values: list[float] = []
                precision_values: list[float] = []
                recall_values: list[float] = []
                fmeasure_values: list[float] = []
                auroc_values: list[float] = []

                bilstm.train(True)
                for i, data in enumerate(dataloader_df40):
                    if i >= batches:
                        break

                    labels, images = data
                    print(f"labels shape: {labels}")

                    image_sequences = torch.stack(images)
                    print(f"image_sequences: {image_sequences.shape}")
                    input_tensor = image_sequences.view(SEQUENCE_LENGTH, BATCH_SIZE, -1)
                    print(f"input_tensor: {input_tensor.shape}")
                    input_tensor = input_tensor.to(device)
                    input_labels = torch.tensor([CLASSES[l] for l in labels], dtype = torch.long)
                    input_labels = input_labels.to(device)

                    # start with clean gradient
                    output_tensor = bilstm.forward(input_tensor)

                    print(output_tensor)
                    # compute loss, gradients and backprop, adjust weights
                    print(f"output_tensor shape: {output_tensor.shape}")
                    print(f"input_labels shape: {input_labels.shape}")
                    loss = lossf(output_tensor, input_labels)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # current predictions
                    _, predicted_labels = torch.max(output_tensor, 1)
                    predicted_props = torch.nn.functional.softmax(output_tensor, dim=1)
                    print(predicted_props)
                    print(predicted_labels)
                    print(input_labels)
                    print(f"{(predicted_labels == input_labels).sum().item()}/{BATCH_SIZE}")

                    # stats
                    epoch_total += BATCH_SIZE
                    epoch_correct += (predicted_labels == input_labels).sum().item()
                    loss_values.append(loss.item())
                    accuracy_values.append(multiclass_accuracy(predicted_props, input_labels, num_classes = len(CLASSES.keys()), average = "macro").item())
                    precision_values.append(multiclass_precision(predicted_props, input_labels, num_classes = len(CLASSES.keys()), average = "macro").item())
                    recall_values.append(multiclass_recall(predicted_props, input_labels, num_classes = len(CLASSES.keys()), average = "macro").item())
                    fmeasure_values.append(multiclass_f1_score(predicted_props, input_labels, num_classes = len(CLASSES.keys()), average = "macro").item())
                    auroc_values.append(multiclass_auroc(predicted_props, input_labels, num_classes = len(CLASSES.keys()), average = "macro").item())

                    pbar(1)
                bilstm.eval()

                print(f"\tcorrect: {epoch_correct}/{epoch_total}")
                print(f"\tavg loss: {round(sum(loss_values) / len(loss_values), 5)}")
                print(f"\tavg accuracy: {round(sum(accuracy_values) / len(accuracy_values), 5)}")
                print(f"\tavg precision: {round(sum(precision_values) / len(precision_values), 5)}")
                print(f"\tavg recall: {round(sum(recall_values) / len(recall_values), 5)}")
                print(f"\tavg f measure: {round(sum(fmeasure_values) / len(fmeasure_values), 5)}")
                print(f"\tavg area under roc: {round(sum(auroc_values) / len(auroc_values), 5)}")
        if SAVE_MODEL:
            print(f"(>) saving model to '{SAVE_MODEL}'")
            torch.save(bilstm.state_dict(), SAVE_MODEL)
        print("(>) done")

    if MODE_TEST:
        raise NotImplementedError("TODO: not done yet")

        bilstm.eval()

        with alive_bar(batches, title=f"BiLSTM test {batches}@{BATCH_SIZE}") as pbar:
            for i, data in enumerate(dataloader_df40):
                if i >= batches:
                    break # crash fix

                labels, images = data

                image_sequences = torch.stack(images)
                input_tensor = image_sequences.view(SEQUENCE_LENGTH, BATCH_SIZE, -1)
                input_tensor = input_tensor.to(device)
                input_labels = torch.tensor([CLASSES[l] for l in labels], dtype = torch.long)
                input_labels = input_labels.to(device)

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
