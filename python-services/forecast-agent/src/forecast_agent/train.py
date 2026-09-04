import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
from mlflow.models import infer_signature
from core import core_config, get_logger
from sales_data_pipeline import DataManager

log = get_logger(__file__)

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Absolute Percentage Error (WAPE)."""
    actual_sum = np.sum(y_true)
    if actual_sum == 0:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / actual_sum

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculates MAPE safely using sklearn.
    Adds epsilon handling for zero values to avoid division by zero.
    """
    # Alternatively using sklearn directly:
    return mean_absolute_percentage_error(y_true, y_pred)


def train():
    """
    Script to train a LightGBM regression model using MLflow for experiment tracking.
    create mlflow experiment and log parameters, metrics, and model artifacts.
    Register the trained model in MLflow Model Registry for future inference and deployment.
    """


    # 1. Set MLflow experiment & enable autologging
    mlflow.set_experiment(experiment_name=core_config.experiment.experiment_name)
    mlflow.lightgbm.autolog(log_models=False) # Handled manually in step 5 for custom signatures

    # 2. Load data
    data_mgr = DataManager()
    curated_dataset = data_mgr.get_data(data_type=core_config.training.data_type)
    X_train, X_test, y_train, y_test = data_mgr.train_test_split(
        dataset=curated_dataset,
        features=core_config.training.features,
        target=core_config.training.target,
        test_size=core_config.training.test_size,
        temporal_split=core_config.training.is_temporal_split,
        date_col=core_config.training.date_col,
    )

    # 3. Parameters
    params = {
        "objective": core_config.params.objective,
        "metric": core_config.params.metric,
        "boosting_type": core_config.params.boosting_type,
        "n_estimators": core_config.params.n_estimators,
        "learning_rate": core_config.params.learning_rate,
        "num_leaves": core_config.params.num_leaves,
        "subsample": core_config.params.subsample,
        "colsample_bytree": core_config.params.colsample_bytree,
        "random_state": core_config.params.random_state,
        "verbose": core_config.params.verbose
    }

    # 4. Start MLflow Run & Train Model
    with mlflow.start_run(run_name=core_config.experiment.run_name) as run:
        # Log parameters explicitly
        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        # 5. Evaluate Model (Regression)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) 
        wape = calculate_wape(y_test.to_numpy(), y_pred)

        # Log metrics
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "mape": mape,
            "wape": wape
        })

        # 6. Model register
        signature = infer_signature(X_train, y_pred)

        # 7. Log Model Artifact
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path=core_config.experiment.artifact_path,
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=core_config.experiment.registered_model_name  
        )

        
        
        log.info(f"Run completed. Run ID: {run.info.run_id}")


def predict(run_id: str, sample_data):
    # Load logged model using PyFunc interface for inference
    model_uri = f"runs:/{run_id}/{core_config.experiment.artifact_path}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    predictions = loaded_model.predict(sample_data)
    return predictions


if __name__ == "__main__":
    train()