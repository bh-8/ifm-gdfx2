import tensorflow_datasets as tfds
import tensorflow as tf
import os
import numpy as np

# Globale Parameter
DATASET_LOCATION = "./io"
IMG_SIZE = (256, 256, 3)
SEQ_LEN = 32  # 8-32
CLASSES = ["original", "face_swap", "face_reenact"]

class DF40(tfds.core.GeneratorBasedBuilder):
    """TensorFlow Dataset für DF40."""
    
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self):
        """Definiert das FeatureDict für DF40."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="TensorFlow Dataset für DF40 mit Bildsequenzen zur Klassifikation.",
            features=tfds.features.FeaturesDict({
                "sequence": tfds.features.Sequence(
                    tfds.features.Image(shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
                ),
                "label": tfds.features.ClassLabel(names=CLASSES),
            }),
            supervised_keys=("sequence", "label"),
            homepage="https://github.com/YZY-stack/DF40",
            citation=r"""@article{yan2024df40,title={DF40: Toward Next-Generation Deepfake Detection},author={Yan, Zhiyuan and Yao, Taiping and Chen, Shen and Zhao, Yandan and Fu, Xinghe and Zhu, Junwei and Luo, Donghao and Yuan, Li and Wang, Chengjie and Ding, Shouhong and others},journal={arXiv preprint arXiv:2406.13495},year={2024}}""",
        )

    def _split_generators(self):
        """Erstellt Split-Generatoren für Training und Test."""
        return {
            "train": self._generate_examples(os.path.join(DATASET_LOCATION, "train")),
            "test": self._generate_examples(os.path.join(DATASET_LOCATION, "test")),
        }

    def _generate_examples(self, path):
        """Erzeugt Sequenz-Items aus den Bildordnern."""
        for class_id, class_name in enumerate(CLASSES):
            class_path = os.path.join(path, class_name)
            if not os.path.isdir(class_path):
                continue

            for tool_folder in os.listdir(class_path):
                tool_path = os.path.join(class_path, tool_folder)
                data_path = os.path.join(tool_path, "frames")

                if not os.path.isdir(data_path):
                    continue

                for seq_folder in sorted(os.listdir(data_path)):
                    seq_path = os.path.join(data_path, seq_folder)
                    if not os.path.isdir(seq_path):
                        continue
                    
                    image_paths = sorted(tf.io.gfile.glob(seq_path + "/*.png"))
                    sequence = [self._load_image(p) for p in image_paths]
                    
                    # Padding falls notwendig
                    if len(sequence) < SEQ_LEN:
                        padding = [tf.zeros_like(sequence[0])] * (SEQ_LEN - len(sequence))
                        sequence.extend(padding)
                    else:
                        sequence = sequence[:SEQ_LEN]
                    
                    yield f"{class_name}_{seq_folder}", {
                        "sequence": sequence,
                        "label": class_name,
                    }

    def _load_image(self, path):
        """Lädt ein einzelnes Bild."""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        return image
