
import pandas as pd
from dagster import AssetExecutionContext, Config, MetadataValue, asset

from sales_data_pipeline.resources import MLflowResource, StorageResource
from sales_data_pipeline.schemas import CuratedSalesSchema, StagedSalesSchema
from core import core_config


"""
Pipeline stages: raw sales history -> staged -> curated (forecast-agent input).
    - raw layer is immutable
    - staged is the validation gate — bad rows never reach curated
    - curated is what forecast_agent.train reads to build the demand model
"""

@asset(group_name="ingestion")
def raw_data(context: AssetExecutionContext, storage: StorageResource) -> pd.DataFrame:
    """
    read the raw dataset and store it in the raw layer of the data lake.
    """
    # 1. read dataset 
    df = pd.read_csv(core_config.filepath)

    # 2. store the dataset
    path = storage.write_parquet("raw", "sales.parquet", df)

    # 3. create the dataset
    context.add_output_metadata({"row_count": len(df), "path": MetadataValue.path(path)})

    return df


@asset(group_name="validation")
def staged_data(context: AssetExecutionContext, raw_data: pd.DataFrame, storage: StorageResource) -> pd.DataFrame:
    """
    Use the raw dataset to create a staged dataset that is validated and cleaned. 
    The staged dataset is stored in the data lake.
    """

    original_row_count = len(raw_data)

    # 1. use only the data from store 01, 
    dataset = raw_data[raw_data['store'] == 1]

    # 2. conv data column type object -> datetime64
    # if "date" in dataset.columns():
    dataset["date"] = pd.to_datetime(dataset["date"])

    # 3. remove the 'store' column
    dataset = dataset.drop(columns=["store"])

    # 4. validate the dataset
    validated = StagedSalesSchema.validate(dataset)

    # 5. save validated dataset
    path = storage.write_parquet("staged", "sales.parquet", validated)

    # 6. create metadata
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
    """
    Create a curated dataset that is ready for training the demand model. 
    The curated dataset is stored in the data lake.
    """

    # 1. count the original raw 
    original_row_count = len(staged_data)

    # 2. store by date and item
    dataset = staged_data.sort_values(['item', 'date']).reset_index(drop=True)

    # 3. calculate the lag for 1 and 7 days
    dataset['sales_lag_1'] = dataset.groupby('item')['sales'].shift(1)
    dataset['sales_lag_7'] = dataset.groupby('item')['sales'].shift(7)

    # 4. remove all the nan values in lag_n columns
    dataset = dataset.dropna()

    # 5. validate the dataset
    validated = CuratedSalesSchema.validate(dataset)

    # 6. store the validated dataset
    path = storage.write_parquet("curated", "sales.parquet", validated)

    # 7. log dataset version
    mlflow.log_dataset_version(
        run_name="curated_sales_build",
        params={"row_count": len(validated), "output_path": path},
    )

    # 8. create metadata
    context.add_output_metadata(
        {
            "row_count": len(validated),
            "dropped_rows": original_row_count - len(validated),
            "path": MetadataValue.path(path),
            "preview": MetadataValue.md(validated.head().to_markdown()),
        }
    )
    return validated
