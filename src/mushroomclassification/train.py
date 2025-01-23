import operator
import os

import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from loguru import logger
from model import MushroomClassifier
from utils.config_loader import load_config
from utils.gcs import upload_to_gcs

from data import MushroomDatamodule

env = os.getenv
config = load_config()

api = wandb.init(
    # set the wandb project where this run will be logged
    project=env("WANDB_PROJECT"),

    # track hyperparameters and run metadata
    config={
        "learning_rate": config['trainer']['learning_rate'],
        "epochs": config['trainer']['max_epochs'],
        "batch_size": config['data']['batch_size'],
        "percent_of_data": config['data']['percent_of_data'],
    }
)

def save_model_to_registry(run_name: str) -> None:
    artifact = wandb.Artifact(
        name=env("WANDB_MODEL_NAME"),
        type="model",
        description="A model trained to classify mushrooms",
        metadata={"accuracy": 1},
    )

    artifact.add_file("models/mushroom_model.ckpt")
    logged_artifact = api.log_artifact(artifact, aliases=[wandb.run.name])

    logger.info("Artifact saved to run.")

    api.link_artifact(
        artifact=logged_artifact,
        target_path="radugrecu97-dtu-org/wandb-registry-model/model_collection"
    )
    logger.info("Artifact saved to registry.")


def stage_best_model_to_registry(metric_name: str = "accuracy", higher_is_better: bool = True) -> None:
    """
    Stage the best model to the model registry.

    Args:
        collection_name: Name of the artifact collection.
        collection_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.

    """
    api = wandb.Api(
        api_key=env('WANDB_API_KEY'),
        overrides={"entity": env('WANDB_ENTITY'), "project": env('WANDB_PROJECT')},
    )

    artifact_collection = api.artifact_collection(type_name="model", name=f"{env('WANDB_MODEL_NAME')}")

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        print("#####")
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logger.error("No model found in registry.")
        return

    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    best_artifact.link(
        target_path=f"{env('WANDB_ENTITY')}-org/wandb-registry-model/{env('WANDB_REGISTRY')}",
        aliases=["best"],
    )

    best_artifact.save()
    logger.info("Model staged to registry.")


def train():
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename=config['model']['file_name'],
        save_top_k=1,
        mode="min",
        enable_version_counter=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
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
    load_dotenv(override=True)
    train()
    save_model_to_registry(wandb.run.name)
    stage_best_model_to_registry("accuracy")
