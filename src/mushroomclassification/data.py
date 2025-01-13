from pathlib import Path
import typer
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.classes = ["conditionally_edible", "deadly", "edible", "poisonous"]
        self.data_path = raw_data_path
        self.files = self._load_files()
        
    def _load_files(self):
        """Load all files from the dataset as a tuple (file_path, class_label)."""
        files = []
        for cls in self.classes:
            class_path = self.data_path / "Classes" / cls 
            for file_path in class_path.glob("**/*.*"):
                files.append((file_path, cls))
        return files

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.files)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        file_path, class_label = self.files[index]
        return {"path": file_path, "label": class_label}

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
