import os
from invoke import Context, task
from src.mushroomclassification.utils.config_loader import load_config

WINDOWS = os.name == "nt"
PROJECT_NAME = "mushroomclassification"
PYTHON_VERSION = "3.11"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def train(ctx, config_path="model_config.yaml"):
    """Train model."""
    config = load_config(config_path)

    ctx.run(
        f"python src/{PROJECT_NAME}/train.py fit "
        f"--data.data_path={config['data']['data_path']} "
        f"--data.batch_size={config['data']['batch_size']} "
        f"--data.num_workers={config['data']['num_workers']} "
        f"--trainer.max_epochs={config['trainer']['max_epochs']}",
        echo=True,
        pty=not WINDOWS,
    )

@task
def visualize(ctx, config_path="model_config.yaml"):
    """Visualize model predictions."""
    config = load_config(config_path)

    ctx.run(
        f"python src/{PROJECT_NAME}/visualize.py "
        f"--data-path={config['data']['data_path']} "
        f"--batch-size={config['data']['batch_size']} "
        f"--num-workers={config['data']['num_workers']}",
        echo=True,
        pty=not WINDOWS,
    )
    
@task
def create_subset(ctx, source_dir, target_dir, classes, num_samples=10):
    """
    Creates a subset of the dataset using the Typer CLI.
    """
    ctx.run(
        f"python src/mushroomclassification/utils/create_raw_data_subset.py "
        f'"{source_dir}" "{target_dir}" '
        f"--classes \"{classes}\" "
        f"--num-samples {num_samples}",
        echo=True,
        pty=not WINDOWS,
    )
    
@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
