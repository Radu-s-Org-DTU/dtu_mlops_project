from torch.utils.data import Dataset
from pathlib import Path
from mushroomclassification.data import MyDataset

def test_dataset_is_instance_of_torch_dataset():
    dataset = MyDataset(Path("data/raw_subset"))
    assert isinstance(dataset, Dataset), "the dataset is not a valid PyTorch dataset"

def test_dataset_is_not_empty():
    dataset = MyDataset(Path("data/raw_subset"))
    dataset.files = dataset._load_files()
    assert len(dataset) > 0, "the dataset is empty"

def test_sample_keys_and_values():
    dataset = MyDataset(Path("data/raw_subset"))
    sample = dataset[0]
    print(sample)
    assert "path" in sample and "label" in sample, "sample does not contain 'path' and 'label' keys"
    assert sample["path"].exists(), f"file {sample['path']} does not exist"
    assert sample["label"] in dataset.classes, f"invalid label {sample['label']}"
    
def test_all_files_have_valid_extensions():
    dataset = MyDataset(Path("data/raw_subset"))
    valid_extensions = {".png", ".jpg", ".jpeg"}
    for file_path, _ in dataset.files:
        assert file_path.suffix.lower() in valid_extensions, f"file {file_path} does not have a valid extension"
        
def test_dataset_length_matches_files():
    dataset = MyDataset(Path("data/raw_subset"))
    assert len(dataset) == len(dataset.files), "dataset length does not match the number of loaded files"

def test_all_files_belong_to_defined_classes():
    dataset = MyDataset(Path("data/raw_subset"))
    for _, class_label in dataset.files:
        assert class_label in dataset.classes, f"Class label {class_label} is not in the defined classes"

def test_no_duplicate_files():
    dataset = MyDataset(Path("data/raw_subset"))
    file_paths = [file_path for file_path, _ in dataset.files]
    assert len(file_paths) == len(set(file_paths)), "Dataset contains duplicate files"