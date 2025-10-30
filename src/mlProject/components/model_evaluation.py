import setuptools, sys
sys.modules['distutils'] = setuptools._distutils
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Calculate standard regression evaluation metrics:
        - RMSE
        - MAE
        - R²
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """
        Evaluate model performance on test data and log:
        - Parameters
        - Metrics
        - Trained model
        to MLflow (via DagsHub)
        """

        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Split into features and target
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # ✅ Set the correct MLflow tracking URI
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Predict on test data
            predicted_qualities = model.predict(test_x)

            # Compute evaluation metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log params and metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # ✅ Log model to MLflow
            if tracking_url_type_store != "file":
                # Register model in remote MLflow (DagsHub)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    
                )
            else:
                # Local MLflow tracking — skip registry
                mlflow.sklearn.log_model(model, artifact_path="model")

