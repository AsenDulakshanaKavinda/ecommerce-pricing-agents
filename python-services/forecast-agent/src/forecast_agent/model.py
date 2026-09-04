import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core import core_config
from sales_data_pipeline import DataManager


def train():
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
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
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

        # Log metrics
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        })

        # 6. Log Model Artifact
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path=core_config.experiment.artifact_path,
            input_example=X_train[:5]
        )
        
        print(f"Run completed. Run ID: {run.info.run_id}")


def predict(run_id: str, sample_data):
    # Load logged model using PyFunc interface for inference
    model_uri = f"runs:/{run_id}/{core_config.experiment.artifact_path}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    predictions = loaded_model.predict(sample_data)
    return predictions


if __name__ == "__main__":
    train()