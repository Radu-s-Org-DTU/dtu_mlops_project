import operator
import os

from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from loguru import logger
from model import MushroomClassifier
from utils.config_loader import load_config

import wandb

# from data import
from data import MushroomDatamodule


def save_model_to_registry(run_name: str) -> None:
    # Initialize the API with the necessary credentials
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={
            "entity": os.getenv("WANDB_ENTITY"),
            "project": os.getenv("WANDB_PROJECT"),
        },
    )

    # Construct the artifact path using proper string formatting
    artifact_path = f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/{os.getenv('WANDB_MODEL_NAME')}:{run_name}"

    # Fetch the artifact using the API
    artifact = api.artifact(artifact_path)

    # Link the artifact to the desired target path in the model registry
    target_path = f"{os.getenv('WANDB_ENTITY')}/wandb-registry-model/{os.getenv("WANDB_REGISTRY")}"
    artifact.link(target_path)

    # Save the artifact (if necessary, though `link` usually suffices)
    artifact.save()

def stage_best_model_to_registry(model_name: str, metric_name: str = "accuracy", higher_is_better: bool = True) -> None:
    """
    Stage the best model to the model registry.

    Args:
        collection_name: Name of the artifact collection.
        collection_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.

    """
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    artifact_collection = api.artifact_collection(type_name="model", name=os.getenv("WANDB_REGISTRY"))

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logger.error("No model found in registry.")
        return

    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    best_artifact.link(
        target_path=f"{os.getenv('WANDB_ENTITY')}/{os.getenv("WANDB_REGISTRY")}/{model_name}",
        aliases=["best", "staging"],
    )
    best_artifact.save()
    logger.info("Model staged to registry.")


def train():
    config = load_config()

    api = wandb.init(
        # set the wandb project where this run will be logged
        project=os.getenv("WANDB_PROJECT"),

        # track hyperparameters and run metadata
        config={
            "learning_rate": config['trainer']['learning_rate'],
            "epochs": config['trainer']['max_epochs'],
            "batch_size": config['data']['batch_size'],
            "percent_of_data": config['data']['percent_of_data'],
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename=load_config()['model']['file_name'],
        save_top_k=1,
        mode="min",
        enable_version_counter="false",
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    LightningCLI(
        MushroomClassifier,
        MushroomDatamodule,
        trainer_defaults={
            "callbacks": [checkpoint_callback],
            "logger": True
        },
    )


    artifact = wandb.Artifact(
        name="classification-model",
        type="model",
        description="A model trained to classify mushrooms",
        # metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )

    artifact.add_file("models/mushroom_model.ckpt")
    api.log_artifact(artifact, aliases=[wandb.run.name])

    logger.info("Model saved to registry.")

if __name__ == "__main__":
    load_dotenv()
    train()
    # save_model_to_registry(wandb.run.name)
    # stage_best_model_to_registry("the-model", "val_acc")
