



import os
from pathlib import Path
from typing import Optional

from dagster import ConfigurableResource

_PACKAGE_DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data")


class StorageResource(ConfigurableResource):
    """ Filesystem-backed storage for raw/staged/curated layers. """
    base_path: Optional[str] = None

    def _path(self, layer: str, filename: str) -> Path:
        base = Path(self.base_path) if self.base_path else Path(_PACKAGE_DATA_DIR)
        p = base / layer
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    def write_parquet(self, layer: str, filename: str, df) -> str:
        path = self._path(layer, filename)
        df.to_parquet(path, index=False)
        return str(path)

    def read_parquet(self, layer: str, filename: str):
        import pandas as pd

        return pd.read_parquet(self._path(layer, filename))

class MLflowResource(ConfigurableResource):
    """Logs dataset lineage under its own MLflow experiment, separate from
    forecast-agent's "forecast-agent" training-run experiment — so a data
    refresh and a model training run are each traceable independently,
    and a training run can cite which pipeline run's data it used.
    """

    tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = "sales-data-pipeline"

    def log_dataset_version(self, run_name: str, params: dict) -> None:
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)