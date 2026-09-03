
from pydantic import BaseModel
from pathlib import Path

import yaml


class CoreConfig(BaseModel):
    filepath: str


def load_yaml_config(file: str = "config.yaml") -> CoreConfig:
    config_path = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "configs"
            / file
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    return CoreConfig(**config)

core_config = load_yaml_config()


