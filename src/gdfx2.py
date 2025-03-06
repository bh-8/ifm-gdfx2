import tensorflow as tf
import tensorflow.keras.layers as ly
import tensorflow_datasets as tfds

from pathlib import Path

import df40.builder as df40
from parameters import *

#print(tf.config.list_physical_devices('GPU'))

# DATASET STUFF

def df40_list_labeled_items(split_path: Path):
    for class_id, class_name in enumerate(CLASS_LIST):
        class_path: Path = Path(split_path) / class_name
        if not class_path.exists():
            print(f"E {class_path} does not exists!")
            continue
        # gather and sort frames, truncate too long sequences
        for i in sorted([e for e in class_path.glob("**/") if e.match("frames/*")]):
            sequential_data: list[Path] = sorted([x for x in i.glob("*")])[:SEQ_LEN]
            if len(sequential_data) >= SEQ_LEN: # minimum sequence length requirement
                yield sequential_data, class_id

def df40_load_and_preprocess(path_sequence: list[Path], label: int):
    def _load_image(image_path: Path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image)
        image = image / 255.0
        return image

    image_sequence = tf.map_fn(_load_image, path_sequence, dtype=tf.float32)
    return image_sequence, label


train_sequences, train_labels = df40_list_labeled_items(Path(DATASET_PATH + "/train").resolve())
test_sequences, test_labels = df40_list_labeled_items(Path(DATASET_PATH + "/test").resolve())

print(train_sequences)
print(train_labels)

#train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
#test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))

#train_dataset = train_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#test_dataset = test_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

#train_dataset = train_dataset.batch(16).shuffle(100).prefetch(tf.data.AUTOTUNE)
#test_dataset = test_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

import sys
sys.exit(0)

# MODEL STUFF

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=IO_PATH + "/model.weights.h5",
    save_weights_only=True,
    verbose=1
)

def create_model():
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    resnet50.trainable = False # freeze weights

    model = tf.keras.Sequential([
        ly.Input(shape=(SEQ_LEN, *IMG_SIZE)), # , batch_size=BATCH_SIZE input shape is (BATCH_SIZE, SEQ_LEN, *IMG_SIZE)
        ly.TimeDistributed(
            resnet50, name="resnet"
        ), ly.TimeDistributed(
            ly.GlobalAveragePooling2D(), name="pooling2d"
        ), ly.Bidirectional(
            ly.LSTM(128), name="bilstm"
        ), ly.Dense(len(CLASS_LIST), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = create_model()
model.summary()
#model.save_weights((IO_PATH + "/model.weights.h5").format(epoch=0))
#model.load_weights(IO_PATH + "/model.weights.h5")


# TRAINING STUFF

def one_hot_enc(s, l):
    return s, tf.one_hot(l, len(CLASS_LIST)) # transformation to 3x3

#tfds.builder("df40", data_dir=IO_PATH).download_and_prepare()

#df40_train = tfds.load("df40", split="train", as_supervised=True).interleave(
#    lambda x: tf.data.Dataset.from_tensors(x).map(one_hot_enc),
#    cycle_length=tf.data.AUTOTUNE,
#    num_parallel_calls=4
#).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
#df40_test = tfds.load("df40", split="test", as_supervised=True).interleave(
#    lambda x: tf.data.Dataset.from_tensors(x).map(one_hot_enc),
#    cycle_length=tf.data.AUTOTUNE,
#    num_parallel_calls=4
#).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

df40_train = tfds.load("df40", split="train", as_supervised=True).map(one_hot_enc).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
df40_test = tfds.load("df40", split="test", as_supervised=True).map(one_hot_enc).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


history = model.fit(df40_train, epochs=EPOCHS, validation_data=df40_test, callbacks=[cp_callback])

# TODO: Normalisierung von Bildwerten..?
# TODO: Lernrate/WeightDecay/DropOut und Optimierungen aus altem Src übernehmen
# TODO: https://www.tensorflow.org/tutorials/keras/save_and_load
