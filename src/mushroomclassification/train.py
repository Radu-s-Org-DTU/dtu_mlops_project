from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.cli import LightningCLI
from model import MushroomClassifier
from utils.config_loader import load_config

# from data import
from data import MushroomDatamodule

import wandb
from dotenv import load_dotenv
import os

def train():
    load_dotenv()
    env = os.getenv
    config = load_config()

    wandb.init(
        # set the wandb project where this run will be logged
        project=env("WANDB_PROJECT"),

        # track hyperparameters and run metadata
        config={
            "learning_rate": config['trainer']['learning_rate'],
            "architecture": "CNN",
            "dataset": "Mushroom classification",
            "epochs": config['trainer']['max_epochs'],
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename=config['model']['file_name'],
        save_top_k=1,
        monitor="val_loss",  # Ensure monitoring validation loss
        mode="min",
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
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
