import collections
import datetime
import numpy as np
import pathlib as pl
import random
import tensorflow as tf
import tensorflow.keras.layers as ly
import tensorflow_datasets as tfds

CLASS_LIST        = ["original", "face_swap", "face_reenact"]
IO_PATH           = "./io"
IMG_SIZE          = (224, 224, 3)
SEQ_LEN           = 12
BATCH_SIZE        = 8
EPOCHS            = 9
EPOCHS_PATIENCE   = 6
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 3e-3
DROPOUT           = 3e-1
FEATURE_EXTRACTOR = "efficientnet" # efficientnet/resnet

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
    return tf.stack([tf.keras.applications.resnet.preprocess_input(_load_image(elem)) for elem in tf.unstack(path_sequence)] if FEATURE_EXTRACTOR == "resnet" else [_load_image(elem) for elem in tf.unstack(path_sequence)]), tf.one_hot(label, len(CLASS_LIST))

print("Enumerating items...")
train_sequences, train_labels = df40_list_labeled_items(pl.Path(IO_PATH + "/df40/train").resolve())
test_sequences, test_labels = df40_list_labeled_items(pl.Path(IO_PATH + "/df40/test").resolve())

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
train_dataset_classes = collections.Counter([int(l.numpy()) for (_, l) in train_dataset])
test_dataset_classes = collections.Counter([int(l.numpy()) for (_, l) in test_dataset])

print("Train Dataset:")
for i, c in enumerate(CLASS_LIST):
    print(f"  {c}: {train_dataset_classes[i]}x")

class_weights = {i: len(train_dataset) / (c * len(train_dataset_classes)) for i, c in train_dataset_classes.items()}
print("Class Weights:")
for i, c in enumerate(CLASS_LIST):
    print(f"  {c}: {class_weights[i]}")

print("Test Dataset:")
for i, c in enumerate(CLASS_LIST):
    print(f"  {c}: {test_dataset_classes[i]}x")

print("Prefetching items...")
train_dataset = train_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(int(float(train_dataset.cardinality()) * 0.025), reshuffle_each_iteration=True).batch(BATCH_SIZE)
test_dataset = test_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("############################## MODEL ##############################")

print(tf.config.list_physical_devices("GPU"))

# Adam-Optimizer (inkl. opt. Weight Decay/adapt. LR)
#model_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model_optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def create_feature_extractor():
    feature_extractor = None
    if FEATURE_EXTRACTOR == "resnet":
        feature_extractor = tf.keras.applications.ResNet50(weights="imagenet", input_shape=IMG_SIZE, pooling="avg", include_top=False)
    elif FEATURE_EXTRACTOR == "efficientnet":
        feature_extractor = tf.keras.applications.EfficientNetB0(weights="imagenet", input_shape=IMG_SIZE, pooling="avg", include_top=False)
    else:
        return None
    feature_extractor.trainable = False
    return feature_extractor

def create_model():
    model = tf.keras.Sequential([
        ly.Input(shape=(SEQ_LEN, *IMG_SIZE)),
        ly.TimeDistributed(
            create_feature_extractor(), name="baseline"
        ), ly.Bidirectional(
            ly.LSTM(256), name="bilstm"
        ),
        ly.Dropout(DROPOUT),
        ly.Dense(len(CLASS_LIST), activation="softmax") # kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY) [use weight decay in Adam optimizer instead?!]
    ])
    model.compile(optimizer=model_optimizer, loss="categorical_crossentropy", metrics=["auc", "categorical_accuracy", "f1_score"])
    return model

model = create_model()

# Model Checkpoint
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=IO_PATH + "/model.weights.h5", save_weights_only=True, verbose=1)

# Early-Stopping (Training, bis Modell sich nicht weiter verbessert) KEIN VALIDATION LOSS! ZU UNGENAU BZW. ZU ZEITAUFWÄNDIG
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=EPOCHS_PATIENCE, restore_best_weights=True)

# Custom Callback to freeze baseline weights and update learning rate during training
class FreezeBaselineCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        model.get_layer("baseline").trainable = False
        fe_layer = model.get_layer("baseline").layer

        if epoch < EPOCHS_PATIENCE:
            unfreeze_layers: int = int(len(fe_layer.layers) / float(2 ** (epoch + 1)))
            for layer in fe_layer.layers[-unfreeze_layers:]:
                layer.trainable = True
            new_lr = (LEARNING_RATE / 10) / (2 ** epoch)
            model_optimizer.learning_rate.assign(new_lr)
            print(f"Epoch {epoch + 1}: unfreezed {unfreeze_layers}/{len(fe_layer.layers)} layers of baseline model, set learning rate to {new_lr}")
        else:
            new_lr = LEARNING_RATE / (2 ** epoch)
            model_optimizer.learning_rate.assign(new_lr)
            print(f"Epoch {epoch + 1}: freezed all layers of baseline model, set learning rate to {new_lr}")

model.summary()

print("############################## TRAINING ##############################")

history = model.fit(train_dataset, epochs=EPOCHS, class_weight=class_weights, validation_data=test_dataset, validation_steps=int(len(test_dataset) / (5 * BATCH_SIZE)), callbacks=[model_checkpoint, early_stopping, FreezeBaselineCallback()])
print(history)

print("############################## EVALUATION ##############################")
final_results = model.evaluate(test_dataset)
print(final_results)

print("############################## STORING ##############################")

store_path: str = IO_PATH + f"/model-{FEATURE_EXTRACTOR}-{EPOCHS}ep-{SEQ_LEN}sl.keras"
print(f"Saving final model state to '{store_path}'")
model.save(store_path)
