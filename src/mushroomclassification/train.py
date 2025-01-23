import os

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from model import MushroomClassifier
from utils.config_loader import load_config
from utils.gcs import upload_to_gcs

from data import MushroomDatamodule


def train():
    local_model_dir = os.path.abspath("models/")
    os.makedirs(local_model_dir, exist_ok=True)

    checkpoint_filename = load_config()['model']['file_name']
    checkpoint_callback = ModelCheckpoint(
        dirpath=local_model_dir,
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    LightningCLI(
        MushroomClassifier,
        MushroomDatamodule,
        trainer_defaults={
            "callbacks": [checkpoint_callback],
            "logger": True
        },
    )

    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if bucket_name:
        model_file = os.path.join(local_model_dir, f"{checkpoint_filename}.ckpt")
        if os.path.exists(model_file):
            upload_to_gcs(bucket_name, model_file, f"models/{os.path.basename(model_file)}")
        else:
            print(f"Model file {model_file} not found.")
    else:
        print("GCS_BUCKET_NAME environment variable not set. Model saved locally.")

if __name__ == "__main__":
    train()
