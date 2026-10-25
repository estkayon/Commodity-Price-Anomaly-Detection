"""Anomaly detection for commodity prices using Z-Score and Isolation Forest."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config_loader import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect price anomalies using multiple methods."""

    def __init__(self):
        self.features_dir = PROJECT_ROOT / "data" / "features"
        self.output_dir = PROJECT_ROOT / "data" / "anomalies"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        anomaly_cfg = CONFIG.get("anomaly_detection", {})
        self.zscore_threshold = anomaly_cfg.get("zscore_threshold", 2.5)
        self.isolation_contamination = anomaly_cfg.get("isolation_forest_contamination", 0.05)

    def zscore_detection(self, df: pd.DataFrame, price_col: str, window: int = 30) -> pd.DataFrame:
        """Detect anomalies using rolling Z-Score method."""
        result = df.copy()

        result["rolling_mean"] = result.groupby("commodity")[price_col].transform(
            lambda x: x.rolling(window, min_periods=5).mean()
        )
        result["rolling_std"] = result.groupby("commodity")[price_col].transform(
            lambda x: x.rolling(window, min_periods=5).std()
        )
        result["zscore"] = (result[price_col] - result["rolling_mean"]) / result["rolling_std"].replace(0, np.nan)

        result["anomaly_zscore"] = (result["zscore"].abs() > self.zscore_threshold).astype(int)

        count = result["anomaly_zscore"].sum()
        logger.info(f"Z-Score anomalies (threshold={self.zscore_threshold}): {count} detected")
        return result

    def isolation_forest_detection(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        result = df.copy()

        anomalies_all = []
        for commodity in result["commodity"].unique():
            cdf = result[result["commodity"] == commodity].copy()

            # Select features, drop NaN rows for model fitting
            features = cdf[feature_cols].copy()
            valid_mask = features.notna().all(axis=1)
            valid_features = features[valid_mask]

            if len(valid_features) < 10:
                logger.warning(f"{commodity}: not enough valid rows for Isolation Forest")
                cdf["anomaly_iforest"] = 0
                anomalies_all.append(cdf)
                continue

            # Scale features
            scaler = StandardScaler()
            scaled = scaler.fit_transform(valid_features)

            # Fit Isolation Forest
            model = IsolationForest(
                contamination=self.isolation_contamination,
                random_state=42,
                n_estimators=100,
                n_jobs=-1,
            )
            predictions = model.fit_predict(scaled)

            # -1 = anomaly, 1 = normal
            cdf.loc[valid_mask, "anomaly_iforest"] = (predictions == -1).astype(int)
            cdf["anomaly_iforest"] = cdf["anomaly_iforest"].fillna(0).astype(int)

            # Anomaly score (lower = more anomalous)
            cdf.loc[valid_mask, "iforest_score"] = model.decision_function(scaled)

            count = cdf["anomaly_iforest"].sum()
            logger.info(f"{commodity}: {count} Isolation Forest anomalies")
            anomalies_all.append(cdf)

        return pd.concat(anomalies_all, ignore_index=True)

    def combined_detection(self, df: pd.DataFrame, price_col: str, feature_cols: list) -> pd.DataFrame:
        """Run both methods and create combined anomaly flag."""
        result = self.zscore_detection(df, price_col)
        result = self.isolation_forest_detection(result, feature_cols)

        # Combined: anomaly if EITHER method flags it
        result["anomaly_combined"] = ((result["anomaly_zscore"] == 1) | (result["anomaly_iforest"] == 1)).astype(int)

        # Severity: both methods agree = high, one method = medium
        result["anomaly_severity"] = "normal"
        both_mask = (result["anomaly_zscore"] == 1) & (result["anomaly_iforest"] == 1)
        either_mask = result["anomaly_combined"] == 1
        result.loc[either_mask, "anomaly_severity"] = "medium"
        result.loc[both_mask, "anomaly_severity"] = "high"

        total = result["anomaly_combined"].sum()
        high = both_mask.sum()
        logger.info(f"Combined anomalies: {total} total ({high} high severity)")
        return result

    def detect_yahoo(self) -> pd.DataFrame:
        """Run anomaly detection on Yahoo features data."""
        path = self.features_dir / "yahoo_features.csv"
        if not path.exists():
            logger.warning("yahoo_features.csv not found. Run FeatureEngineer first.")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded Yahoo features: {len(df)} rows")

        feature_cols = [
            "return_1d", "return_5d", "volatility_7d", "volatility_30d",
            "bb_position", "rsi_14", "macd_histogram",
            "price_to_sma_7", "price_to_sma_30", "zscore_30d",
        ]

        result = self.combined_detection(df, price_col="close", feature_cols=feature_cols)

        output_path = self.output_dir / "yahoo_anomalies.csv"
        result.to_csv(output_path, index=False)
        logger.info(f"Saved Yahoo anomalies: {len(result)} rows -> {output_path}")
        return result

    def detect_fred(self) -> pd.DataFrame:
        """Run anomaly detection on FRED features data."""
        path = self.features_dir / "fred_features.csv"
        if not path.exists():
            logger.warning("fred_features.csv not found. Run FeatureEngineer first.")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded FRED features: {len(df)} rows")

        feature_cols = [
            "return_1m", "return_3m", "volatility_6m", "zscore_12m",
        ]

        result = self.combined_detection(df, price_col="price", feature_cols=feature_cols)

        output_path = self.output_dir / "fred_anomalies.csv"
        result.to_csv(output_path, index=False)
        logger.info(f"Saved FRED anomalies: {len(result)} rows -> {output_path}")
        return result

    def detect_all(self) -> dict:
        """Run anomaly detection on all datasets."""
        results = {}
        results["yahoo"] = self.detect_yahoo()
        results["fred"] = self.detect_fred()
        return results