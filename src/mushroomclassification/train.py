import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

# from data import
from data import MushroomDatamodule
from model import MushroomClassifier

def train():
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        #filename="mushroom_model-{epoch:02d}-{train_loss:.2f}",
        filename="mushroom_model",
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
