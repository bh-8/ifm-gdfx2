import tensorflow as tf
import tensorflow.keras.layers as ly
import tensorflow_datasets as tfds

import df40.builder as df40
from parameters import *

#print(tf.config.list_physical_devices('GPU'))

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

df40_train = tfds.load("df40", split="train", as_supervised=True).map(one_hot_enc).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
df40_test = tfds.load("df40", split="test", as_supervised=True).map(one_hot_enc).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history = model.fit(df40_train, epochs=EPOCHS, validation_data=df40_test, callbacks=[cp_callback])

# TODO: Normalisierung von Bildwerten..?
# TODO: Lernrate/WeightDecay/DropOut und Optimierungen aus altem Src übernehmen
# TODO: https://www.tensorflow.org/tutorials/keras/save_and_load
