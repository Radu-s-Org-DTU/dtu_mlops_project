
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy

from src.mushroomclassification.model import MushroomClassifier


def test_model_construction():
    model = MushroomClassifier(learning_rate=0.001)
    assert isinstance(model.backbone, nn.Sequential)
    assert isinstance(model.classifier, nn.Sequential)
    assert isinstance(model.criterion, nn.CrossEntropyLoss)
    assert isinstance(model.accuracy, MulticlassAccuracy)
    assert model.learning_rate == 0.001

def test_forward_pass():
    model = MushroomClassifier(learning_rate=0.001)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 4)

def test_configure_optimizers():
    model = MushroomClassifier(learning_rate=0.001)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, optim.AdamW)
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 0.001

def test_training_step():
    model = MushroomClassifier(learning_rate=0.001)
    dummy_data = torch.randn(2, 3, 224, 224)
    dummy_target = torch.tensor([0, 1])
    batch = (dummy_data, dummy_target)

    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    assert loss.item() > 0
