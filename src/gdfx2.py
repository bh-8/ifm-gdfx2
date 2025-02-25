import tensorflow_datasets as tfds
import datasets.df40

dataset, dataset_info = tfds.load(
    "df40",  # Name des Datasets (wie in df40.py definiert)
    split="train",  # Oder "test"
    with_info=True,  # Gibt Metadaten zurück
    data_dir="datasets/"  # Falls du es an einem bestimmten Ort speichern willst
)

