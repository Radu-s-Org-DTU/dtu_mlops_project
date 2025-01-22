from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.cli import LightningCLI
from model import MushroomClassifier
from utils.config_loader import load_config

# from data import
from data import MushroomDatamodule

def train():
    config = load_config()

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
