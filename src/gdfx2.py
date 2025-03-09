import numpy as np
import pathlib as pl
import random
import tensorflow as tf
import tensorflow.keras.layers as ly
import tensorflow_datasets as tfds

CLASS_LIST = ["original", "face_swap", "face_reenact"]
IO_PATH = "./io"
IMG_SIZE = (256, 256, 3)
SEQ_LEN = 8
BATCH_SIZE = 12
EPOCHS = 3

print(tf.config.list_physical_devices('GPU'))

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
        image = tf.image.resize(image, [256, 256])
        image = image / 255.0
        return image
    return tf.stack([_load_image(elem) for elem in tf.unstack(path_sequence)]), tf.one_hot(label, len(CLASS_LIST))

print("Enumerating items...")
train_sequences, train_labels = df40_list_labeled_items(pl.Path(IO_PATH + "/df40/train").resolve())
test_sequences, test_labels = df40_list_labeled_items(pl.Path(IO_PATH + "/df40/test").resolve())

print("Shuffling items...")
train_data = list(zip(train_sequences, train_labels))
test_data = list(zip(test_sequences, test_labels))
random.shuffle(train_data)
random.shuffle(test_data)
train_sequences, train_labels = zip(*train_data)
test_sequences, test_labels = zip(*test_data)
train_sequences, train_labels = list(train_sequences), list(train_labels)
test_sequences, test_labels = list(test_sequences), list(test_labels)

print("Preprocessing items...")
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))

train_dataset = train_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

print("Batching items...")
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset_classes = np.concatenate([np.argmax(y, axis = -1) for x, y in train_dataset], axis = 0)
test_dataset_classes = np.concatenate([np.argmax(y, axis = -1) for x, y in test_dataset], axis = 0)

print("Train Dataset:")
for i, c in enumerate(CLASS_LIST):
    print(f" {i} {c} -> {(train_dataset_classes == i).sum()}")
print(train_dataset_classes)

print("Test Dataset:")
for i, c in enumerate(CLASS_LIST):
    print(f" {i} {c} -> {(test_dataset_classes == i).sum()}")
print(test_dataset_classes)

print("############################## MODEL ##############################")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=IO_PATH + "/model.weights.h5",
    save_weights_only=True,
    verbose=1
)

def create_model():
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    resnet50.trainable = False # freeze weights

    model = tf.keras.Sequential([
        ly.Input(shape=(SEQ_LEN, *IMG_SIZE)),
        ly.TimeDistributed(
            resnet50, name="resnet"
        ), ly.TimeDistributed(
            ly.GlobalAveragePooling2D(), name="pooling2d"
        ), ly.Bidirectional(
            ly.LSTM(128), name="bilstm"
        ), ly.Dense(len(CLASS_LIST), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["auc", "categorical_accuracy", "f1_score", "precision", "recall"])
    return model

model = create_model()
model.summary()

if pl.Path(IO_PATH + "/model.weights.h5").exists():
    model.load_weights(IO_PATH + "/model.weights.h5")
    print(f"Loaded initial weights from '{IO_PATH + '/model.weights.h5'}'")

print("############################## TRAINING ##############################")

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[cp_callback])

# TODO: Lernrate/WeightDecay/DropOut und Optimierungen aus altem Src übernehmen
