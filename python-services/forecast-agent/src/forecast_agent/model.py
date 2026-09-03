from core import core_config
from sales_data_pipeline import DataManager

import mlflow

def train():
    # 1. set mlflow experiment
    mlflow.set_experiment(experiment_name="expe-name")

    # 2. load data
    data_mgr = DataManager()
    curated_dataset = data_mgr.get_data(data_type="curated")
    X_train, X_test, y_train, y_test = data_mgr.train_test_split(
        dataset=curated_dataset, 
        split_date="2000-10-10",
        features=[],
        target=str,
    )

    # 2. start model training
    with mlflow.start_run(run_name="run-name") as run:
        ...


def predict():
    pass
