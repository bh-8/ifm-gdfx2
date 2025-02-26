import tensorflow as tf
import tensorflow_datasets as tfds
import df40.builder as df40

print(tf.config.list_physical_devices('GPU'))

builder = df40.Builder(local_path="./io/df40")
builder.download_and_prepare()

df40_train = builder.as_dataset(split="train")
