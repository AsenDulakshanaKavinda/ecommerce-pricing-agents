from datetime import datetime

import pandas as pd


class DataManager:
    def __init__(self):
        ...


    def get_data(self, data_type: str) -> pd.DataFrame:
        """
        Get the data of the specified type.

        Args:
            data_type (str): The type of data to retrieve. Must be one of "raw", "staged", or "curated".

        Returns:
            pd.DataFrame: The requested data as a pandas DataFrame.

        Raises:
            ValueError: If the specified data_type is not recognized.
        """
        if data_type == "raw":
            return self._get_raw_data()
        elif data_type == "staged":
            return self._get_staged_data()
        elif data_type == "curated":
            return self._get_curated_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _get_raw_data(self, path: str) -> pd.DataFrame:
        # return raw data
        return pd.read_parquet(path)

    def _get_staged_data(self, path: str) -> pd.DataFrame:
        # return staged data
        return pd.read_parquet(path)

    def _get_curated_data(self, path: str) -> pd.DataFrame:
        # return curated data
        return pd.read_parquet(path)


    def train_test_split(self, dataset: pd.DataFrame, split_date: datetime, features: list, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets based on the specified split date.
        Args:
            dataset (pd.DataFrame): The dataset to split.
            split_date (datetime): The date to use for splitting the dataset. All data before this date will be used for training, and all data on or after this date will be used for testing.
            features (list): The list of feature column names to include in the training and testing sets.
            target (str): The name of the target column to include in the training and testing sets.
        Returns:
            tuple: A tuple containing four pandas DataFrames: (X_train, X_test, y_train, y_test).
        """

        train = dataset[dataset['date'] < split_date]
        test = dataset[dataset['date'] >= split_date]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        return X_train, X_test, y_train, y_test
