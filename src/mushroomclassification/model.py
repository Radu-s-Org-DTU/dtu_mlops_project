import operator
import os

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch import nn, optim
from torchmetrics import Accuracy
from utils.config_loader import load_config
from visualize import plot_classification_per_class, plot_training_loss

import wandb

load_dotenv(override=True)
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

def save_model_to_registry(run_name: str, metrics) -> None:
    artifact = wandb.Artifact(
        name=env("WANDB_MODEL_NAME"),
        type="model",
        description="A model trained to classify mushrooms",
        metadata=metrics,
    )

    artifact.add_file("models/mushroom_model.ckpt")
    logged_artifact = api.log_artifact(artifact, aliases=[wandb.run.name])

    logger.info("Artifact saved to run.")

    api.link_artifact(
        artifact=logged_artifact,
        target_path="radugrecu97-dtu-org/wandb-registry-model/model_collection"
    )
    logger.info("Artifact saved to registry.")


def stage_best_model_to_registry(metric_name: str, higher_is_better: bool = True) -> None:
    """
    Stage the best model to the model registry.

    Args:
        collection_name: Name of the artifact collection.
        collection_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.

    """
    api = wandb.Api(
        api_key=env("WANDB_API_KEY"),
        overrides={"entity": env("WANDB_ENTITY"), "project": env("WANDB_PROJECT")},
    )

    artifact_collection = api.artifact_collection(type_name="model", name=f"{env('WANDB_MODEL_NAME')}")

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
        target_path="radugrecu97-dtu-org/wandb-registry-model/model_collection",
        aliases=["best"],
    )

    best_artifact.save()
    logger.info("Model staged to registry.")

class MushroomClassifier(L.LightningModule):
    """My awesome model."""

    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.train_losses = []
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Added to fix shape issue
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 128),  # Fixed incorrect input size
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=4)  # For logging accuracy
        self.correct_classifications = [0] * 4  # Assuming 4 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        return self.classifier(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = self.accuracy(preds, target)
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"train_loss": loss.item(), "train_accuracy": acc})
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)

        # Calculate and log accuracy using self.accuracy
        acc = self.accuracy(preds, target)

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"validation_loss": loss.item(), "validation_accuracy": acc})

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate and log accuracy using self.accuracy
        acc = self.accuracy(logits, y)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"test_loss": loss.item(), "test_accuracy": acc})

        # Track correct classifications per class
        _, predicted = torch.max(logits, 1)
        for t, p in zip(y.view(-1), predicted.view(-1)):
            if t == p:
                self.correct_classifications[t.item()] += 1

        return loss

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        """Save train_losses and correct_classifications in the checkpoint."""
        checkpoint["train_losses"] = self.train_losses
        checkpoint["correct_classifications"] = self.correct_classifications
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load train_losses and correct_classifications from checkpoint."""
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        else:
            self.train_losses = []

        if "correct_classifications" in checkpoint:
            self.correct_classifications = checkpoint["correct_classifications"]
        else:
            self.correct_classifications = [0] * 4

    def on_test_end(self):
        total_test_samples = sum(len(batch[0]) for batch in self.trainer.datamodule.test_dataloader())
        print(f"Total test samples: {total_test_samples}")
        plot_training_loss(self.train_losses)
        plot_classification_per_class(self.correct_classifications,
                                      ["conditionally_edible", "deadly", "edible", "poisonous"],
                                      total_test_samples)

    def on_train_end(self):
        """At the end of training, log final validation accuracy."""
        final_val_accuracy = self.accuracy.compute()  # Compute the final validation accuracy
        logger.info(f"final_validation_accuracy {final_val_accuracy.item()}")

        # Save the model with its accuracy
        save_model_to_registry(run_name=wandb.run.name, metrics={"accuracy": final_val_accuracy.item()})

        # Stage the best model to the registry
        stage_best_model_to_registry(metric_name="accuracy")

        # Reset the accuracy metric for the next run
        self.accuracy.reset()



if __name__ == "__main__":
    model = MushroomClassifier()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with a dummy input of 224x224x3 (RGB image)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
