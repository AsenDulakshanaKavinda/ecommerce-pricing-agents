import os

import polars as pl
import argparse
from pathlib import Path

from dotenv import load_dotenv

# load db credentials from .env file
load_dotenv()

class DatabaseInsertError(Exception):
    """Base exception for database insertion failures."""


def read_and_write_to_DB():
    parser = argparse.ArgumentParser(description="Read path of the CSV file, load it and write it to DB")
    parser.add_argument("filename", type=str, help="The path to the csv file you want to read")
    args = parser.parse_args()

    # str -> Path
    file_path = Path(args.filename)

    if not file_path.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    print(f"Reading {file_path}...")

    db_url = os.getenv("SYNC_DATABASE_URL")
    if not db_url:
        print("Error: SYNC_DATABASE_URL environment variable is not set in the .env file.")
        ValueError("Error: SYNC_DATABASE_URL environment variable is not set in the .env file.")

    try:
        df = (
            pl.read_csv(file_path, schema_overrides={"Invoice": pl.String})
            .rename(lambda col: "customer_id" if col.lower() == "customer id" else col.lower())
        )

        df.write_database(
            table_name="raw_dataset",
            connection=db_url,
            if_table_exists="replace"
        )
        print("Database write complete!")
    except Exception as e:
        raise DatabaseInsertError(str(e))


if __name__ == "__main__":
    read_and_write_to_DB()




