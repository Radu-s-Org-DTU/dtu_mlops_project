import torch
from torch import nn, optim
import lightning as L
from torch.optim import Optimizer

class MushroomClassifier(L.LightningModule):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
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
        return optim.Adam(self.parameters(), lr=1e-2)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        return loss

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
