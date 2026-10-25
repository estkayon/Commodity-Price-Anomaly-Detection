"""Feature engineering for commodity price data."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config_loader import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate features from cleaned commodity data."""

    def __init__(self):
        self.processed_dir = PROJECT_ROOT / CONFIG["data"]["processed_dir"]
        self.features_dir = PROJECT_ROOT / "data" / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def add_technical_features(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Add technical indicator features to price data."""
        result = df.copy()

        # Returns
        result["return_1d"] = result.groupby("commodity")[price_col].pct_change(1)
        result["return_5d"] = result.groupby("commodity")[price_col].pct_change(5)
        result["return_20d"] = result.groupby("commodity")[price_col].pct_change(20)

        # Log returns
        result["log_return"] = result.groupby("commodity")[price_col].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # Moving averages
        for window in [7, 14, 30, 60]:
            result[f"sma_{window}"] = result.groupby("commodity")[price_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            result[f"ema_{window}"] = result.groupby("commodity")[price_col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean()
            )

        # Volatility (rolling std of returns)
        result["volatility_7d"] = result.groupby("commodity")["log_return"].transform(
            lambda x: x.rolling(7, min_periods=2).std()
        )
        result["volatility_30d"] = result.groupby("commodity")["log_return"].transform(
            lambda x: x.rolling(30, min_periods=5).std()
        )

        # Bollinger Bands (20-day)
        result["bb_middle"] = result["sma_30"]
        result["bb_std"] = result.groupby("commodity")[price_col].transform(
            lambda x: x.rolling(30, min_periods=5).std()
        )
        result["bb_upper"] = result["bb_middle"] + 2 * result["bb_std"]
        result["bb_lower"] = result["bb_middle"] - 2 * result["bb_std"]
        result["bb_position"] = (result[price_col] - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        )

        # RSI (14-day)
        result["rsi_14"] = result.groupby("commodity")[price_col].transform(
            lambda x: _compute_rsi(x, 14)
        )

        # MACD
        result["macd"] = result["ema_14"] - result["ema_30"]
        result["macd_signal"] = result.groupby("commodity")["macd"].transform(
            lambda x: x.ewm(span=9, min_periods=1).mean()
        )
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        # Price relative to moving averages
        result["price_to_sma_7"] = result[price_col] / result["sma_7"]
        result["price_to_sma_30"] = result[price_col] / result["sma_30"]

        # Z-score (30-day rolling)
        result["zscore_30d"] = result.groupby("commodity")[price_col].transform(
            lambda x: (x - x.rolling(30, min_periods=5).mean()) / x.rolling(30, min_periods=5).std()
        )

        logger.info(f"Added technical features: {result.shape[1] - df.shape[1]} new columns")
        return result

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        result = df.copy()

        result["day_of_week"] = result["date"].dt.dayofweek
        result["month"] = result["date"].dt.month
        result["quarter"] = result["date"].dt.quarter
        result["year"] = result["date"].dt.year
        result["day_of_year"] = result["date"].dt.dayofyear
        result["is_month_start"] = result["date"].dt.is_month_start.astype(int)
        result["is_month_end"] = result["date"].dt.is_month_end.astype(int)

        # Cyclical encoding for month and day_of_week
        result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
        result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
        result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

        logger.info("Added temporal features")
        return result

    def add_lag_features(self, df: pd.DataFrame, price_col: str = "close", lags: list = None) -> pd.DataFrame:
        """Add lagged price features."""
        if lags is None:
            lags = [1, 2, 3, 5, 7, 14, 30]

        result = df.copy()
        for lag in lags:
            result[f"lag_{lag}"] = result.groupby("commodity")[price_col].shift(lag)

        logger.info(f"Added {len(lags)} lag features")
        return result

    def engineer_yahoo(self) -> pd.DataFrame:
        """Full feature engineering pipeline for Yahoo data."""
        path = self.processed_dir / "yahoo_cleaned.csv"
        if not path.exists():
            logger.warning("yahoo_cleaned.csv not found. Run DataCleaner first.")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded cleaned Yahoo data: {len(df)} rows")

        df = self.add_technical_features(df, price_col="close")
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df, price_col="close")

        output_path = self.features_dir / "yahoo_features.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved Yahoo features: {len(df)} rows, {df.shape[1]} columns -> {output_path}")
        return df

    def engineer_fred(self) -> pd.DataFrame:
        """Full feature engineering pipeline for FRED data."""
        path = self.processed_dir / "fred_cleaned.csv"
        if not path.exists():
            logger.warning("fred_cleaned.csv not found. Run DataCleaner first.")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded cleaned FRED data: {len(df)} rows")

        # FRED is monthly, so use different windows
        result = df.copy()

        # Returns
        result["return_1m"] = result.groupby("commodity")["price"].pct_change(1)
        result["return_3m"] = result.groupby("commodity")["price"].pct_change(3)
        result["return_6m"] = result.groupby("commodity")["price"].pct_change(6)
        result["return_12m"] = result.groupby("commodity")["price"].pct_change(12)

        # Moving averages (monthly)
        for window in [3, 6, 12]:
            result[f"sma_{window}m"] = result.groupby("commodity")["price"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # Volatility
        result["volatility_6m"] = result.groupby("commodity")["return_1m"].transform(
            lambda x: x.rolling(6, min_periods=2).std()
        )

        # Z-score
        result["zscore_12m"] = result.groupby("commodity")["price"].transform(
            lambda x: (x - x.rolling(12, min_periods=3).mean()) / x.rolling(12, min_periods=3).std()
        )

        # Lags
        for lag in [1, 2, 3, 6, 12]:
            result[f"lag_{lag}m"] = result.groupby("commodity")["price"].shift(lag)

        # Temporal
        result = self.add_temporal_features(result)

        output_path = self.features_dir / "fred_features.csv"
        result.to_csv(output_path, index=False)
        logger.info(f"Saved FRED features: {len(result)} rows, {result.shape[1]} columns -> {output_path}")
        return result

    def engineer_all(self) -> dict:
        """Run feature engineering for all datasets."""
        results = {}
        results["yahoo"] = self.engineer_yahoo()
        results["fred"] = self.engineer_fred()
        return results


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)