import matplotlib.pyplot as plt
import seaborn as sns
import torch
import typer
from model import MushroomClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.config_loader import load_config

from data import MushroomDatamodule


def plot_training_loss(model):
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label="Training Loss", color="blue")
    plt.title("Training Loss Over Iterations", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/train_losses.png")
    plt.show()

def compute_predictions(model, test_loader):
    embeddings, targets, predictions = [], [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            embeddings.append(outputs)
            targets.append(labels)

            # Get predictions
            preds = outputs.argmax(dim=1)
            predictions.append(preds)

    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()
    predictions = torch.cat(predictions).numpy()

    return embeddings, targets, predictions

def plot_classification_results(targets, predictions, class_names):
    correct_counts = {class_name: 0 for class_name in class_names}
    wrong_counts = {class_name: 0 for class_name in class_names}

    for target, pred in zip(targets, predictions):
        if target == pred:
            correct_counts[class_names[target]] += 1
        else:
            wrong_counts[class_names[target]] += 1

    labels = list(class_names)
    correct = [correct_counts[label] for label in labels]
    wrong = [wrong_counts[label] for label in labels]

    x_positions = range(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, correct, bar_width, label="Correct", color="blue", alpha=0.7)
    plt.bar([p + bar_width for p in x_positions], wrong, bar_width, label="Wrong", color="red", alpha=0.7)

    plt.xticks([p + bar_width / 2 for p in x_positions], labels, rotation=45, ha="right")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Correctly and Wrongly Classified Samples per Class", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/classification_results.png")
    plt.show()

def plot_tsne(embeddings, targets, class_names):
    if embeddings.shape[1] > 500:
        embeddings = PCA(n_components=100).fit_transform(embeddings)
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    tsne_colors = sns.color_palette("hsv", len(class_names))
    for i in range(len(class_names)):
        mask = targets == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_names[i],
            alpha=0.7,
            color=tsne_colors[i],
        )
    plt.legend()
    plt.title("TSNE Visualization of Embeddings", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/tsne.png")
    plt.show()

def visualize(
    data_path: str = typer.Option(..., "--data-path", help="Path to the data"),
    batch_size: int = typer.Option(..., "--batch-size", help="Batch size for data loading"),
    num_workers: int = typer.Option(..., "--num-workers", help="Number of data-loading workers"),
) -> None:
    model = MushroomClassifier.load_from_checkpoint('models/' + load_config()['model']['file_name'] + '.ckpt')

    plot_training_loss(model)

    model.eval()
    dm = MushroomDatamodule(data_path=data_path, batch_size=batch_size, num_workers=num_workers)
    dm.setup(stage="test")

    test_loader = dm.test_dataloader()
    embeddings, targets, predictions = compute_predictions(model, test_loader)

    plot_classification_results(targets, predictions, dm.data_train.classes)
    plot_tsne(embeddings, targets, dm.data_train.classes)

if __name__ == "__main__":
    typer.run(visualize)