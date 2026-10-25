"""Price forecasting using Prophet and LightGBM."""

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from src.config_loader import CONFIG, PROJECT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Forecast commodity prices using Facebook Prophet."""

    def __init__(self, forecast_horizon: int = None):
        cfg = CONFIG.get("forecasting", {})
        self.horizon = forecast_horizon or cfg.get("horizon_days", 30)
        self.output_dir = PROJECT_ROOT / "data" / "forecasts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def forecast_single(self, df: pd.DataFrame, commodity_name: str, price_col: str = "close") -> pd.DataFrame:
        """Forecast a single commodity using Prophet."""
        from prophet import Prophet

        cdf = df[["date", price_col]].dropna().copy()
        cdf.columns = ["ds", "y"]
        cdf["ds"] = pd.to_datetime(cdf["ds"])

        if len(cdf) < 30:
            logger.warning(f"{commodity_name}: not enough data for Prophet (need 30+, got {len(cdf)})")
            return pd.DataFrame()

        # Train/test split (last horizon days as test)
        train = cdf.iloc[:-self.horizon]
        test = cdf.iloc[-self.horizon:]

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )

        # Suppress Prophet stdout
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        model.fit(train)
        sys.stdout = old_stdout

        # Predict on test period + future
        future = model.make_future_dataframe(periods=self.horizon + self.horizon)
        forecast = model.predict(future)

        # Merge actuals for evaluation
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.merge(cdf, on="ds", how="left")
        result["commodity"] = commodity_name
        result["model"] = "prophet"

        # Calculate metrics on test set
        test_pred = result[result["ds"].isin(test["ds"])]
        if len(test_pred) > 0 and test_pred["y"].notna().any():
            actual = test_pred["y"].dropna()
            predicted = test_pred.loc[actual.index, "yhat"]
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = mean_absolute_percentage_error(actual, predicted) * 100
            logger.info(f"{commodity_name} Prophet - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")

        return result

    def forecast_yahoo(self) -> pd.DataFrame:
        """Forecast all Yahoo commodities."""
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        if not path.exists():
            path = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"
        if not path.exists():
            logger.warning("No Yahoo data found for forecasting")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        all_forecasts = []

        for commodity in df["commodity"].unique():
            cdf = df[df["commodity"] == commodity].sort_values("date")
            forecast = self.forecast_single(cdf, commodity, price_col="close")
            if not forecast.empty:
                all_forecasts.append(forecast)

        if not all_forecasts:
            return pd.DataFrame()

        combined = pd.concat(all_forecasts, ignore_index=True)
        output_path = self.output_dir / "prophet_yahoo.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved Prophet Yahoo forecasts -> {output_path}")
        return combined

    def forecast_fred(self) -> pd.DataFrame:
        """Forecast all FRED commodities."""
        path = PROJECT_ROOT / "data" / "anomalies" / "fred_anomalies.csv"
        if not path.exists():
            path = PROJECT_ROOT / "data" / "features" / "fred_features.csv"
        if not path.exists():
            logger.warning("No FRED data found for forecasting")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        all_forecasts = []

        for commodity in df["commodity"].unique():
            cdf = df[df["commodity"] == commodity].sort_values("date")
            # FRED is monthly, shorter horizon
            self_horizon_backup = self.horizon
            self.horizon = min(self.horizon, len(cdf) // 4)
            if self.horizon < 3:
                self.horizon = 3
            forecast = self.forecast_single(cdf, commodity, price_col="price")
            self.horizon = self_horizon_backup
            if not forecast.empty:
                all_forecasts.append(forecast)

        if not all_forecasts:
            return pd.DataFrame()

        combined = pd.concat(all_forecasts, ignore_index=True)
        output_path = self.output_dir / "prophet_fred.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved Prophet FRED forecasts -> {output_path}")
        return combined


class LightGBMForecaster:
    """Forecast commodity prices using LightGBM."""

    def __init__(self, forecast_horizon: int = None):
        cfg = CONFIG.get("forecasting", {})
        self.horizon = forecast_horizon or cfg.get("horizon_days", 30)
        self.output_dir = PROJECT_ROOT / "data" / "forecasts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_features(self, df: pd.DataFrame, price_col: str, target_horizon: int = 1) -> tuple:
        """Prepare features and target for LightGBM."""
        feature_candidates = [
            "return_1d", "return_5d", "return_20d", "log_return",
            "sma_7", "sma_14", "sma_30", "sma_60",
            "ema_7", "ema_14", "ema_30", "ema_60",
            "volatility_7d", "volatility_30d",
            "bb_position", "rsi_14", "macd", "macd_histogram",
            "price_to_sma_7", "price_to_sma_30", "zscore_30d",
            "day_of_week", "month", "quarter", "day_of_year",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "lag_1", "lag_2", "lag_3", "lag_5", "lag_7", "lag_14", "lag_30",
            # FRED-specific
            "return_1m", "return_3m", "return_6m", "return_12m",
            "sma_3m", "sma_6m", "sma_12m", "volatility_6m", "zscore_12m",
            "lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
        ]

        available = [c for c in feature_candidates if c in df.columns]
        if not available:
            return None, None, []

        # Target: future price change
        target = df[price_col].shift(-target_horizon)

        features = df[available].copy()
        valid = features.notna().all(axis=1) & target.notna()

        return features[valid], target[valid], available

    def forecast_single(self, df: pd.DataFrame, commodity_name: str, price_col: str = "close") -> pd.DataFrame:
        """Forecast a single commodity using LightGBM."""
        cdf = df.sort_values("date").reset_index(drop=True)

        X, y, feature_names = self._prepare_features(cdf, price_col)
        if X is None or len(X) < 50:
            logger.warning(f"{commodity_name}: not enough data for LightGBM")
            return pd.DataFrame()

        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        logger.info(f"{commodity_name} LightGBM - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        logger.info(f"Top 5 features: {importance.head(5)['feature'].tolist()}")

        # Build result
        dates = cdf.loc[X_test.index, "date"].values
        result = pd.DataFrame({
            "ds": dates,
            "y": y_test.values,
            "yhat": y_pred,
            "commodity": commodity_name,
            "model": "lightgbm",
        })

        return result

    def forecast_yahoo(self) -> pd.DataFrame:
        """Forecast all Yahoo commodities."""
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        if not path.exists():
            path = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"
        if not path.exists():
            logger.warning("No Yahoo data found")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        all_forecasts = []

        for commodity in df["commodity"].unique():
            cdf = df[df["commodity"] == commodity].sort_values("date")
            forecast = self.forecast_single(cdf, commodity, price_col="close")
            if not forecast.empty:
                all_forecasts.append(forecast)

        if not all_forecasts:
            return pd.DataFrame()

        combined = pd.concat(all_forecasts, ignore_index=True)
        output_path = self.output_dir / "lgbm_yahoo.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved LightGBM Yahoo forecasts -> {output_path}")
        return combined

    def forecast_fred(self) -> pd.DataFrame:
        """Forecast all FRED commodities."""
        path = PROJECT_ROOT / "data" / "anomalies" / "fred_anomalies.csv"
        if not path.exists():
            path = PROJECT_ROOT / "data" / "features" / "fred_features.csv"
        if not path.exists():
            logger.warning("No FRED data found")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        all_forecasts = []

        for commodity in df["commodity"].unique():
            cdf = df[df["commodity"] == commodity].sort_values("date")
            forecast = self.forecast_single(cdf, commodity, price_col="price")
            if not forecast.empty:
                all_forecasts.append(forecast)

        if not all_forecasts:
            return pd.DataFrame()

        combined = pd.concat(all_forecasts, ignore_index=True)
        output_path = self.output_dir / "lgbm_fred.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved LightGBM FRED forecasts -> {output_path}")
        return combined