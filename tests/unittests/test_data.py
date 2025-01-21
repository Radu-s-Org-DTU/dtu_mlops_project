from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from mushroomclassification.data import MushroomDataset


def test_dataset_is_instance_of_torch_dataset():
    dataset = MushroomDataset(Path("data/raw_subset"))
    assert isinstance(dataset, Dataset), "the dataset is not a valid PyTorch dataset"

def test_dataset_is_not_empty():
    dataset = MushroomDataset(Path("data/raw_subset"))
    assert len(dataset.image_files) > 0, "the dataset is empty"

def test_sample_keys_and_values():
    dataset = MushroomDataset(Path("data/raw_subset"))
    sample = dataset[0]

    assert isinstance(sample, tuple), "Sample is not a tuple"
    assert len(sample) == 2, "Sample does not contain exactly two elements"

    image, label = sample
    assert isinstance(image, Image.Image), f"Image is not a PIL.Image.Image: {type(image)}"
    assert label in range(len(dataset.classes)), f"Invalid label {label}"

def test_all_files_have_valid_extensions():
    dataset = MushroomDataset(Path("data/raw_subset"))
    valid_extensions = {".png", ".jpg", ".jpeg"}
    for file in dataset.image_files:
        assert file[0].suffix.lower() in valid_extensions, f"file {file[0]} does not have a valid extension"

def test_dataset_length_matches_files():
    dataset = MushroomDataset(Path("data/raw_subset"))
    assert len(dataset) == len(dataset.image_files), "dataset length does not match the number of loaded files"

def test_all_files_belong_to_defined_classes():
    dataset = MushroomDataset(Path("data/raw_subset"))
    for file in dataset.image_files:
        assert 0 <= file[1] < len(dataset.classes), (
                f"Class label index {file[1]} is out of range for the defined classes"
            )
