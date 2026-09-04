
from pydantic import BaseModel
from pathlib import Path


import yaml

class DataSourceConfig(BaseModel):
    filepath: str
    raw_file: str
    staged_file: str
    curated_file: str

class TrainingConfig(BaseModel):
    data_type: str
    features: list
    target: str
    test_size: float
    is_temporal_split: bool
    date_col: str

class ExperimentConfig(BaseModel):
    experiment_name: str
    run_name: str
    artifact_path: str
    registered_model_name: str


class CoreConfig(BaseModel):
    data_source: DataSourceConfig
    training: TrainingConfig
    experiment: ExperimentConfig


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


