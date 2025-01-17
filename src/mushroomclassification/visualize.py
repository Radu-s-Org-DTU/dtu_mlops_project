import matplotlib.pyplot as plt
import torch
import typer
from model import MushroomClassifier
from data import MushroomDatamodule
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path", help="Path to the model checkpoint"),
    data_path: str = typer.Option(..., "--data-path", help="Path to the data"),
    batch_size: int = typer.Option(..., "--batch-size", help="Batch size for data loading"),
    num_workers: int = typer.Option(..., "--num-workers", help="Number of data-loading workers"),
) -> None:
    model = MushroomClassifier.load_from_checkpoint(checkpoint_path)
    
    # plot training loss
    plt.plot(model.train_losses)
    plt.title("Training Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/train_losses.png")
    plt.show()
    
    model.eval()
    model.fc = torch.nn.Identity()

    dm = MushroomDatamodule(data_path=data_path, batch_size=batch_size, num_workers=num_workers)
    dm.setup(stage="test")

    embeddings, targets = [], []
    test_loader = dm.test_dataloader()
    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            embeddings.append(outputs)
            targets.append(labels)
    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:
        embeddings = PCA(n_components=100).fit_transform(embeddings)
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    class_names = dm.data_train.classes
    
    # plot TSNE
    plt.figure(figsize=(10, 10))
    for i in range(len(class_names)):
        mask = targets == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=class_names[i], alpha=0.5)
    plt.legend()
    plt.title("TSNE Visualization of Embeddings")
    plt.savefig("reports/figures/tsne.png")
    plt.show()

if __name__ == "__main__":
    typer.run(visualize)