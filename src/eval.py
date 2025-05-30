import collections
import numpy as np
import pathlib as pl
import tensorflow as tf
import tensorflow.keras.applications as ap
import tensorflow.keras.metrics as mt
import time

CLASS_LIST        = ["original", "face_swap", "face_reenact"]
IO_PATH           = "./io"
IMG_SIZE          = (224, 224, 3)

SEQ_LEN           = 12              # 8 / 12 / 16
BATCH_SIZE        = 8               # 12 / 8 / 4
FEATURE_EXTRACTOR = "resnet"        # resnet / efficientnet

print("############################## DATASET ##############################")

def df40_list_labeled_items(split_path: pl.Path):
    list_sequences: list[list[str]] = []
    list_labels: list[int] = []
    for class_id, class_name in enumerate(CLASS_LIST):
        class_path: pl.Path = pl.Path(split_path) / class_name
        if not class_path.exists():
            print(f"E {class_path} does not exists!")
            continue
        # gather and sort frames, truncate too long sequences
        for i in sorted([e for e in class_path.glob("**/") if e.match("frames/*")]):
            sequential_paths: list[pl.Path] = sorted([f for f in i.glob("*")], key = lambda x : int(x.stem))[:SEQ_LEN]
            if len(sequential_paths) >= SEQ_LEN: # minimum sequence length requirement
                list_sequences.append([str(x) for x in sequential_paths])
                list_labels.append(class_id)

    return list_sequences, list_labels

def df40_load_and_preprocess(path_sequence: list[str], label: int):
    def _load_image(image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE[:2])
        return image
    return tf.stack([ap.resnet.preprocess_input(_load_image(elem)) for elem in tf.unstack(path_sequence)] if FEATURE_EXTRACTOR == "resnet" else [_load_image(elem) for elem in tf.unstack(path_sequence)]), tf.one_hot(label, len(CLASS_LIST))

print("Enumerating items...")
test_sequences, test_labels = df40_list_labeled_items(pl.Path(IO_PATH).resolve() / "df40" / "test")

print("Preprocessing items...")
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
test_dataset_classes = collections.Counter([int(l.numpy()) for (_, l) in test_dataset])

print("Test Dataset:")
test_dataset_items = 0
for i, c in enumerate(CLASS_LIST):
    print(f"  {c}: {test_dataset_classes[i]}x")
    test_dataset_items = test_dataset_items + test_dataset_classes[i]

print("Prefetching items...")
test_dataset = test_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("############################## EVALUATION ##############################")

print(tf.config.list_physical_devices("GPU"))

metrics_list = [
    mt.CategoricalAccuracy(name="ca"),
    mt.CategoricalCrossentropy(name="cc"),
    mt.F1Score(name="f1"),
    mt.F1Score(name="f1w", average="weighted"),
    mt.Precision(name="p0", class_id=0),
    mt.Precision(name="p1", class_id=1),
    mt.Precision(name="p2", class_id=2),
    mt.Precision(name="p"),
    mt.Recall(name="r0", class_id=0),
    mt.Recall(name="r1", class_id=1), 
    mt.Recall(name="r2", class_id=2),
    mt.Recall(name="r")
]

model_path: pl.Path = pl.Path(IO_PATH).resolve() / f"model-{FEATURE_EXTRACTOR}-sl{SEQ_LEN:02d}-final.keras"
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(
    loss=None,
    optimizer=None,
    metrics=metrics_list
)
model.summary()

start_time = time.time()
final_results = model.evaluate(test_dataset)
end_time = time.time()
tps = (end_time - start_time) / test_dataset_items
fps = 1 / (tps / SEQ_LEN)
print(f"Klassifikationszeit pro Sample: {round(tps*1000)} ms, FPS: {round(fps)}")
for m in metrics_list:
    result = m.result()
    formatted_result = f"{result:.4f}" if tf.size(result) == 1 else result
    print(f"    {m.name} = {formatted_result}")
