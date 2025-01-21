from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image

class DF40ImageSequenceDataset(Dataset):
    def __init__(self, root_path: str):
        self.root: Path = Path(root_path).resolve()

        # browse data location
        pathlist: list[Path] = [e for e in self.root.glob("**/") if e.match("frames/*")]

        # gather image sequences plus label
        self.items: list[tuple[str, list[Path]]] = [(l, sorted(i.glob("*"))) for i in pathlist if (l := str(i).split("/")[-4])]

    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return ('imagesequence', 'label')
