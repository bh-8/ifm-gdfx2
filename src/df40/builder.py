import tensorflow_datasets as tfds
import tensorflow as tf
import pathlib as pl
from parameters import *

class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # local_path: str, 
    def __init__(self, data_dir: str):
        self.local_path: pl.Path = pl.Path(IO_PATH).resolve() / "df40"
        self.img_size: tuple = IMG_SIZE
        self.seq_len: int = SEQ_LEN
        self.batch_size: int = BATCH_SIZE
        self.class_list: list[str] = CLASS_LIST
        super().__init__(data_dir=str(pl.Path(IO_PATH).resolve() / "tfcache"))

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="TensorFlow Dataset für DF40 mit Bildsequenzen zur Klassifikation.",
            features=tfds.features.FeaturesDict({
                "sequence": tfds.features.Sequence(
                    tfds.features.Image(shape=self.img_size),
                    length = self.seq_len
                ),
                "label": tfds.features.ClassLabel(names=self.class_list)
            }),
            supervised_keys=("sequence", "label"),
            homepage="https://github.com/YZY-stack/DF40",
            citation=r"""@article{yan2024df40,title={DF40: Toward Next-Generation Deepfake Detection},author={Yan, Zhiyuan and Yao, Taiping and Chen, Shen and Zhao, Yandan and Fu, Xinghe and Zhu, Junwei and Luo, Donghao and Yuan, Li and Wang, Chengjie and Ding, Shouhong and others},journal={arXiv preprint arXiv:2406.13495},year={2024}}""",
        )

    def _split_generators(self, dl_manager):

        return {
            "train": self._generate_examples(pl.Path(self.local_path) / "train"),
            "test": self._generate_examples(pl.Path(self.local_path) / "test")
        }

    def _generate_examples(self, split_path):
        """Erzeugt Sequenz-Items aus den Bildordnern."""
        for class_id, class_name in enumerate(CLASS_LIST):
            class_path: pl.Path = pl.Path(split_path) / class_name

            if not class_path.exists():
                print(f"E {class_path} does not exists!")
                continue

            # loop folders containing a frame sequence each
            for i in sorted([e for e in class_path.glob("**/") if e.match("frames/*")]):
                # gather frames, truncate too long sequences and only use files with name pattern XXX.png
                sequential_data: list[pl.Path] = sorted([x for x in i.glob("*") if len(x.stem) == 3])[:self.seq_len]
                if len(sequential_data) >= self.seq_len: # minimum sequence length requirement
                    yield f"{class_name}_{str(i).split('/')[-3]}_{i.name}", {
                        "sequence": [ # TODO. Warum .numpy() notwendig?
                            tf.image.decode_png(tf.io.read_file(str(x))).numpy() for x in sequential_data
                        ],
                        "label": class_name
                    }
