



import os

import mlflow
import pandas as pd
from sales_data_pipeline.data_manager import DataManager
from core import core_config
from dotenv import load_dotenv

load_dotenv()

def predict(data: pd.DataFrame):
    """
    Predicts using a trained LightGBM regression model loaded from MLflow Model Registry.
    The model is loaded using the model name and version number specified in environment variables.
    Args:
        data (pd.DataFrame): Input features for prediction.
    Returns:
        np.ndarray: Predicted values.
    """


    model_name = os.getenv("MODEL_NAME")
    version_number = int(os.getenv("VERSION_NUMBER"))

    # Load logged model using PyFunc interface for inference
    model_uri = f"models:/{model_name}/{version_number}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    predictions = loaded_model.predict(data)
    return predictions

if __name__ == "__main__":

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
    
    predictions = predict(X_test[:3])
    print(predictions)


