import lightning as L
from lightning.pytorch.cli import LightningCLI

# from data import
from data import MushroomDatamodule
from model import MushroomClassifier

def cli_main():

    LightningCLI(
        MushroomClassifier,
        MushroomDatamodule
        )


if __name__ == "__main__":
    cli_main()
