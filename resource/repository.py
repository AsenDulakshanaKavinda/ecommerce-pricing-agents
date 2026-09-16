import os
from dagster import asset, Definitions
from dagster_duckdb import DuckDBResource
import pandas as pd

@asset
def sample_data(duckdb: DuckDBResource):
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    
    with duckdb.get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users AS SELECT * FROM df")

@asset(deps=[sample_data])
def queried_data(duckdb: DuckDBResource):
    with duckdb.get_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        print(f"Total users: {result[0]}")

defs = Definitions(
    assets=[sample_data, queried_data],
    resources={
        "duckdb": DuckDBResource(
            database=os.getenv("DUCKDB_DATABASE_PATH", "/opt/dagster/app/duckdb_data/main.duckdb")
        )
    }
)