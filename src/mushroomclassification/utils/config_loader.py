import yaml


def load_config(config_path="configs/model_config.yaml"):
    """Utility to load and return the YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
