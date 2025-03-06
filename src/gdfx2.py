import tensorflow as tf
import tensorflow.keras.layers as ly
import tensorflow_datasets as tfds
import numpy as np

from pathlib import Path
from parameters import *

print(tf.config.list_physical_devices('GPU'))

# DATASET STUFF

def df40_list_labeled_items(split_path: Path):
    list_sequences: list[list[str]] = []
    list_labels: list[int] = []
    for class_id, class_name in enumerate(CLASS_LIST):
        class_path: Path = Path(split_path) / class_name
        if not class_path.exists():
            print(f"E {class_path} does not exists!")
            continue
        # gather and sort frames, truncate too long sequences
        for i in sorted([e for e in class_path.glob("**/") if e.match("frames/*")]):
            sequential_data: list[str] = sorted([str(x) for x in i.glob("*")])[:SEQ_LEN]
            if len(sequential_data) >= SEQ_LEN: # minimum sequence length requirement
                list_sequences.append(sequential_data)
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
    #return tf.map_fn(_load_image, path_sequence), tf.one_hot(label, len(CLASS_LIST))

train_sequences, train_labels = df40_list_labeled_items(Path(IO_PATH + "/df40/train").resolve())
test_sequences, test_labels = df40_list_labeled_items(Path(IO_PATH + "/df40/test").resolve())

train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))

train_dataset = train_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(df40_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(len(train_dataset)).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
        ly.Input(shape=(SEQ_LEN, *IMG_SIZE)),
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

print("##################################################")


t = np.concatenate([np.argmax(y, axis = -1) for x, y in train_dataset], axis = 0)
print(t)
print(len(t))



#history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[cp_callback])

# TODO: Lernrate/WeightDecay/DropOut und Optimierungen aus altem Src übernehmen
# TODO: https://www.tensorflow.org/tutorials/keras/save_and_load
