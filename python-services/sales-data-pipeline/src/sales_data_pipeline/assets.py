
import pandas as pd
from dagster import AssetExecutionContext, Config, MetadataValue, asset

from sales_data_pipeline.resources import MLflowResource, StorageResource
from sales_data_pipeline.schemas import CuratedSalesSchema, StagedSalesSchema


"""Pipeline stages: raw sales history -> staged -> curated (forecast-agent input).

- raw layer is immutable
- staged is the validation gate — bad rows never reach curated
- curated is what forecast_agent.train reads to build the demand model
"""


class RawDataConfig(Config):
    filepath: str 



@asset(group_name="ingestion")
def raw_data(context: AssetExecutionContext, storage: StorageResource) -> pd.DataFrame:

    # df = pd.read_csv(config.filepath)
    df = pd.read_csv("/home/viper/projects/ecommerce-pricing-agents/python-services/sales-data-pipeline/data/raw/train.csv")
    path = storage.write_parquet("raw", "sales.parquet", df)
    context.add_output_metadata({"row_count": len(df), "path": MetadataValue.path(path)})
    return df


@asset(group_name="validation")
def staged_data(context: AssetExecutionContext, raw_data: pd.DataFrame, storage: StorageResource) -> pd.DataFrame:

    original_row_count = len(raw_data)

    # 1. use only the data from store 01, 
    dataset = raw_data[raw_data['store'] == 1]

    # 2. conv data column type object -> datetime64
    # if "date" in dataset.columns():
    dataset["date"] = pd.to_datetime(dataset["date"])

    # 3. remove the 'store' column
    dataset = dataset.drop(columns=["store"])

    # 4.
    validated = StagedSalesSchema.validate(dataset)

    path = storage.write_parquet("staged", "sales.parquet", validated)
    context.add_output_metadata(
        {
            "row_count": len(validated),
            "dropped_rows": original_row_count - len(validated),
            "path": MetadataValue.path(path),
        }
    )
    return validated


@asset(group_name="features")
def curated_sales(
    context: AssetExecutionContext,
    staged_data: pd.DataFrame,
    storage: StorageResource,
    mlflow: MLflowResource,
) -> pd.DataFrame:
    original_row_count = len(staged_data)

    dataset =staged_data.sort_values(['item', 'date']).reset_index(drop=True)

    dataset['sales_lag_1'] = dataset.groupby('item')['sales'].shift(1)
    dataset['sales_lag_7'] = dataset.groupby('item')['sales'].shift(7)

    dataset = dataset.dropna()

    validated = CuratedSalesSchema.validate(dataset)

    path = storage.write_parquet("curated", "sales.parquet", validated)

    mlflow.log_dataset_version(
        run_name="curated_sales_build",
        params={"row_count": len(validated), "output_path": path},
    )

    context.add_output_metadata(
        {
            "row_count": len(validated),
            "dropped_rows": original_row_count - len(validated),
            "path": MetadataValue.path(path),
            "preview": MetadataValue.md(validated.head().to_markdown()),
        }
    )
    return validated
