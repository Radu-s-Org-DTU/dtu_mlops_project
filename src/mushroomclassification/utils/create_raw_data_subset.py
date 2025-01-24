import os
import random
import shutil

import typer

app = typer.Typer()

def create_raw_data_subset(
    source_dir: str,
    target_dir: str,
    classes: list[str],
    num_samples: int = 10
):
    """
    Creates a subset of the dataset with a specified number of random samples per class.

    Args:
        source_dir (str): Path to the raw dataset.
        target_dir (str): Path to where the subset should be saved.
        classes (list[str]): List of class names to sample from.
        num_samples (int): Number of samples per class. Default is 10.
    """
    os.makedirs(target_dir, exist_ok=True)

    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        subset_path = os.path.join(target_dir, "Classes", cls)
        os.makedirs(subset_path, exist_ok=True)

        files = []
        for root, _, filenames in os.walk(class_path):
            for f in filenames:
                file_path = os.path.join(root, f)
                if os.path.isfile(file_path):
                    files.append(file_path)

        if len(files) == 0:
            typer.echo(f"No files found in {class_path}")
            continue

        sampled_files = random.sample(files, min(len(files), num_samples))

        for file_path in sampled_files:
            rel_path = os.path.relpath(file_path, class_path)
            dest_path = os.path.join(subset_path, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file_path, dest_path)

    typer.echo(f"Subset created at {target_dir}")

@app.command()
def create_subset(
    source_dir: str = typer.Argument(..., help="path to the raw dataset containing class directories"),
    target_dir: str = typer.Argument(..., help="path to where the subset should be saved"),
    classes: str = typer.Option(..., help="comma-separated list of class names to sample from"),
    num_samples: int = typer.Option(10, help="number of samples per class"),
):
    class_list = classes.split(",")
    create_raw_data_subset(source_dir, target_dir, class_list, num_samples)

if __name__ == "__main__":
    app()
