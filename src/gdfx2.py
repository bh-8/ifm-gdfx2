import collections
import numpy as np
import pathlib as pl
import random
import tensorflow as tf
import tensorflow.keras.layers as ly
import tensorflow_datasets as tfds

CLASS_LIST = ["original", "face_swap", "face_reenact"]
IO_PATH = "./io"
IMG_SIZE = (256, 256, 3)
SEQ_LEN = 12
BATCH_SIZE = 8
EPOCHS = 12

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

train_data = list(zip(train_sequences, train_labels))
test_data = list(zip(test_sequences, test_labels))
random.shuffle(train_data)
random.shuffle(test_data)
train_sequences, train_labels = zip(*train_data)
test_sequences, test_labels = zip(*test_data)
train_sequences, train_labels = list(train_sequences)[:1000], list(train_labels)[:1000]
test_sequences, test_labels = list(test_sequences)[:1000], list(test_labels)[:1000]

print("Preprocessing items...")
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
train_dataset_classes = collections.Counter([int(l.numpy()) for (_, l) in train_dataset])
test_dataset_classes = collections.Counter([int(l.numpy()) for (_, l) in test_dataset])

print("Train Dataset:")
for i, c in enumerate(CLASS_LIST):
    print(f" {i} {c} -> {train_dataset_classes[i]}")
print(train_dataset_classes)

print("Test Dataset:")
for i, c in enumerate(CLASS_LIST):
    print(f" {i} {c} -> {test_dataset_classes[i]}")
print(test_dataset_classes)

print("Prefetching items...")
train_dataset = train_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(int(float(train_dataset.cardinality()) * 0.025), reshuffle_each_iteration=True).batch(BATCH_SIZE)
test_dataset = test_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(int(float(test_dataset.cardinality()) * 0.025), reshuffle_each_iteration=True).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("############################## MODEL ##############################")

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=IO_PATH + "/model.weights.h5",
    save_weights_only=True,
    verbose=1
)

# LR-Scheduler (ReduceLROnPlateau)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

# Early-Stopping (Training, bis Modell sich nicht weiter verbessert)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

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
            ly.LSTM(256), name="bilstm"
        ),
        ly.Dropout(0.3), # Dropout Layer
        ly.Dense(len(CLASS_LIST), activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.003)) # L2-Regularisierung
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["auc", "categorical_accuracy", "f1_score"])
    return model

model = create_model()
model.summary()

if pl.Path(IO_PATH + "/model.weights.h5").exists():
    model.load_weights(IO_PATH + "/model.weights.h5")
    print(f"Loaded initial weights from '{IO_PATH + '/model.weights.h5'}'")

print("############################## TRAINING ##############################")

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, validation_freq=3, callbacks=[model_checkpoint, lr_scheduler, early_stopping])

print("############################## STORING/CONVERT ##############################")
print(f"Saving latest model state to '{IO_PATH + '/model_final.pb'}'")
model.save(IO_PATH + "/model_final.pb")

print(f"Converting quantized model...")
model_converter = tf.lite.TFLiteConverter.from_saved_model(IO_PATH + "/model_final.pb")
model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
model_quantized = model_converter.convert()
