import dagster as dg


@dg.asset
def main_data_source() -> str:
    return "data_pipeline/data_source/online_retail_II.csv"
