import lightning as L
import torch
from torch import nn, optim


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        return self.classifier(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        """Save train_losses in the checkpoint."""
        checkpoint["train_losses"] = self.train_losses
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load train_losses from checkpoint."""
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        else:
            self.train_losses = []

    # def validation_step(self, val_batch):
    #     x = val_batch['input_ids']
    #     y = val_batch['label']
    #     outputs = self(x)
    #     accuracy = Accuracy(task='binary').to(torch.device('cuda'))
    #     acc = accuracy(outputs, y)
    #     self.log('accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
    #     return


if __name__ == "__main__":
    model = MushroomClassifier()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with a dummy input of 224x224x3 (RGB image)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
