

import pandas as pd
from core import core_config
from sklearn.model_selection import train_test_split

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
            return self._get_raw_data(core_config.data_source.raw_file)
        elif data_type == "staged":
            return self._get_staged_data(core_config.data_source.staged_file)
        elif data_type == "curated":
            return self._get_curated_data(core_config.data_source.curated_file)
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


    def train_test_split(
            self,
            dataset: pd.DataFrame,
            features: list[str],
            target: str,
            test_size: float = 0.20,
            temporal_split: bool = True,
            date_col: str = "date"
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            """
            Splits dataset using test percentage ratio.
            
            Args:
                test_size (float): Proportion of dataset to include in the test split (e.g. 0.2 for 20%).
                temporal_split (bool): If True, preserves chronological ordering without shuffling.
            """
            if temporal_split:
                dataset = dataset.sort_values(by=date_col)
                split_idx = int(len(dataset) * (1 - test_size))

                train = dataset.iloc[:split_idx]
                test = dataset.iloc[split_idx:]

                X_train, y_train = train[features], train[target]
                X_test, y_test = test[features], test[target]
            else:
                X = dataset[features]
                y = dataset[target]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, shuffle=False
                )

            return X_train, X_test, y_train, y_test