"""MLflow experiment tracking for model training and evaluation."""

import logging
import os
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from src.config_loader import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments, metrics, and models using MLflow."""

    def __init__(self):
        mlops_cfg = CONFIG.get("mlops", {})
        self.tracking_uri = mlops_cfg.get("mlflow_tracking_uri", "mlruns")
        self.experiment_name = mlops_cfg.get("experiment_name", "priceradar")

        # Set up MLflow (use SQLite backend for MLflow 3.x compatibility)
        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(self.experiment_name)

        self.forecasts_dir = PROJECT_ROOT / "data" / "forecasts"
        self.reports_dir = PROJECT_ROOT / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def log_forecast_run(self, model_name: str, source: str, forecast_df: pd.DataFrame, price_col: str = "y"):
        """Log a forecasting run to MLflow."""
        if forecast_df.empty:
            logger.warning(f"Empty forecast for {model_name}/{source}, skipping")
            return

        with mlflow.start_run(run_name=f"{model_name}_{source}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_param("model", model_name)
            mlflow.log_param("source", source)
            mlflow.log_param("n_commodities", forecast_df["commodity"].nunique())
            mlflow.log_param("commodities", ",".join(forecast_df["commodity"].unique()))

            # Calculate and log metrics per commodity
            all_metrics = []
            for commodity in forecast_df["commodity"].unique():
                cdf = forecast_df[forecast_df["commodity"] == commodity]
                actual = cdf[price_col].dropna()
                predicted = cdf.loc[actual.index, "yhat"]

                if len(actual) == 0:
                    continue

                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mape = mean_absolute_percentage_error(actual, predicted) * 100

                tag = commodity.lower().replace(" ", "_")
                mlflow.log_metric(f"mae_{tag}", round(mae, 4))
                mlflow.log_metric(f"rmse_{tag}", round(rmse, 4))
                mlflow.log_metric(f"mape_{tag}", round(mape, 4))

                all_metrics.append({
                    "commodity": commodity,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                })

            # Overall metrics
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                mlflow.log_metric("avg_mae", round(metrics_df["mae"].mean(), 4))
                mlflow.log_metric("avg_rmse", round(metrics_df["rmse"].mean(), 4))
                mlflow.log_metric("avg_mape", round(metrics_df["mape"].mean(), 4))

                # Save metrics report
                report_path = self.reports_dir / f"{model_name}_{source}_metrics.csv"
                metrics_df.to_csv(report_path, index=False)
                mlflow.log_artifact(str(report_path))

                logger.info(f"Logged {model_name}/{source}: avg MAPE={metrics_df['mape'].mean():.1f}%")

    def log_all_forecasts(self):
        """Log all available forecast results to MLflow."""
        forecast_files = {
            ("prophet", "yahoo"): "prophet_yahoo.csv",
            ("prophet", "fred"): "prophet_fred.csv",
            ("lightgbm", "yahoo"): "lgbm_yahoo.csv",
            ("lightgbm", "fred"): "lgbm_fred.csv",
        }

        for (model, source), filename in forecast_files.items():
            path = self.forecasts_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                self.log_forecast_run(model, source, df)
            else:
                logger.info(f"Skipping {filename} (not found)")

    def log_anomaly_summary(self):
        """Log anomaly detection summary to MLflow."""
        anomaly_dir = PROJECT_ROOT / "data" / "anomalies"

        with mlflow.start_run(run_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_param("method", "zscore+isolation_forest")

            for source in ["yahoo", "fred"]:
                path = anomaly_dir / f"{source}_anomalies.csv"
                if not path.exists():
                    continue

                df = pd.read_csv(path)
                total = len(df)
                anomalies = df["anomaly_combined"].sum()
                high = (df["anomaly_severity"] == "high").sum()

                mlflow.log_metric(f"{source}_total_rows", total)
                mlflow.log_metric(f"{source}_anomalies", int(anomalies))
                mlflow.log_metric(f"{source}_high_severity", int(high))
                mlflow.log_metric(f"{source}_anomaly_rate", round(anomalies / total * 100, 2))

                logger.info(f"Logged anomaly summary for {source}")