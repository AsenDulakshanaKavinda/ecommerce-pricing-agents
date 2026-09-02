from dagster import Definitions, load_assets_from_modules

from sales_data_pipeline import assets
from sales_data_pipeline.resources import MLflowResource, StorageResource

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "storage": StorageResource(),  # defaults to this package's own data/ dir
        "mlflow": MLflowResource(),
    }
)