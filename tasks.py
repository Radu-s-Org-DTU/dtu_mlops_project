import os
import yaml
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mushroomclassification"
PYTHON_VERSION = "3.11"

def load_config(config_path="model_config.yaml"):
    """Utility to load and return the YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
    data_conf = config["data"]
    trainer_conf = config["trainer"]

    ctx.run(
        f"python src/{PROJECT_NAME}/train.py fit "
        f"--data.data_path={data_conf['data_path']} "
        f"--data.batch_size={data_conf['batch_size']} "
        f"--data.num_workers={data_conf['num_workers']} "
        f"--trainer.max_epochs={trainer_conf['max_epochs']}",
        echo=True,
        pty=not WINDOWS,
    )

@task
def visualize(ctx, config_path="model_config.yaml"):
    """Visualize model predictions."""
    config = load_config(config_path)
    data_conf = config["data"]
    model_conf = config["model"]

    ctx.run(
        f"python src/{PROJECT_NAME}/visualize.py "
        f"--checkpoint-path={model_conf['checkpoint_path']} "
        f"--data-path={data_conf['data_path']} "
        f"--batch-size={data_conf['batch_size']} "
        f"--num-workers={data_conf['num_workers']}",
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
