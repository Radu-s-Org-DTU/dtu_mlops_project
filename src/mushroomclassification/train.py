from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from model import MushroomClassifier
from utils.config_loader import load_config

# from data import
from data import MushroomDatamodule


def train():
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename=load_config()['model']['file_name'],
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

if __name__ == "__main__":
    train()
