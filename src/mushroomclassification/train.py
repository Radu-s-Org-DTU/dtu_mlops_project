from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from model import MushroomClassifier
from utils.config_loader import load_config

# from data import
from data import MushroomDatamodule

config = load_config()


def train():
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename=config['model']['file_name'],
        save_top_k=1,
        mode="min",
        enable_version_counter=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="validation_accuracy", patience=3, verbose=True, mode="min"
    )

    LightningCLI(
        MushroomClassifier,
        MushroomDatamodule,
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stopping_callback],
            "logger": True
        },
    )

if __name__ == "__main__":
    train()
