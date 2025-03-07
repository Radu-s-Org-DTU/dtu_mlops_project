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
def train(ctx, config_path="configs/model_config.yaml"):
    """Train model."""
    config = load_config(config_path)

    ctx.run(
        f"""python src/{PROJECT_NAME}/train.py fit \
            --seed_everything={config['seed']} \
            --data.data_path={config['data']['data_path']} \
            --data.batch_size={config['data']['batch_size']} \
            --data.percent_of_data={config['data']['percent_of_data']} \
            --data.train_pct={config['data']['train_pct']} \
            --data.test_pct={config['data']['test_pct']} \
            --data.val_pct={config['data']['val_pct']} \
            --data.num_workers={config['data']['num_workers']} \
            --model.learning_rate={config['trainer']['learning_rate']} \
            --trainer.max_epochs={config['trainer']['max_epochs']} \
            --trainer.deterministic=True \
            --trainer.precision=16-mixed \
            --trainer.accelerator=cpu \
            --trainer.devices=1""",
        echo=True,
        pty=not WINDOWS,
    )

@task
def visualize(ctx, config_path="configs/model_config.yaml"):
    config = load_config(config_path)

    ctx.run(
        f"""python src/{PROJECT_NAME}/train.py test \
            --seed_everything={config['seed']} \
            --ckpt_path='models/{config['model']['file_name']}.ckpt' \
            --data.data_path={config['data']['data_path']} \
            --data.batch_size={config['data']['batch_size']} \
            --data.percent_of_data={config['data']['percent_of_data']} \
            --data.train_pct={config['data']['train_pct']} \
            --data.test_pct={config['data']['test_pct']} \
            --data.val_pct={config['data']['val_pct']} \
            --data.num_workers={config['data']['num_workers']} \
            --model.learning_rate={config['trainer']['learning_rate']} \
            --trainer.max_epochs={config['trainer']['max_epochs']} \
            --trainer.deterministic=True \
            --trainer.precision=16-mixed \
            --trainer.accelerator=cpu \
            --trainer.devices=1""",
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

@task
def serve_api(ctx, host="0.0.0.0", port=8000, reload=True, log_level="debug"):
    """Start the FastAPI application."""
    reload_flag = "--reload" if reload else ""
    ctx.run(
        f"uvicorn --app-dir src/mushroomclassification "
        f"api:app --host {host} "
        f"--port {port} "
        f"{reload_flag} "
        f"--log-level {log_level}",
        echo=True,
        pty=not WINDOWS,
    )
