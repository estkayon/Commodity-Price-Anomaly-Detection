"""PriceRadar - Pipeline Tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config_loader import PROJECT_ROOT, load_config
from src.data.yahoo_collector import YahooCollector
from src.data.fred_collector import FredCollector
from src.news.news_collector import NewsCollector
from src.features.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.anomaly.detector import AnomalyDetector
from src.models.forecaster import ProphetForecaster, LightGBMForecaster


# ========== Config Tests ==========

class TestConfig:
    def test_config_loads(self):
        config = load_config()
        assert config is not None
        assert "commodities" in config

    def test_config_has_yahoo(self):
        config = load_config()
        assert "yahoo" in config["commodities"]

    def test_config_has_fred(self):
        config = load_config()
        assert "fred" in config["commodities"]

    def test_config_has_anomaly_settings(self):
        config = load_config()
        assert "anomaly" in config

    def test_config_has_forecast_settings(self):
        config = load_config()
        assert "forecasting" in config

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()


# ========== Data File Tests ==========

class TestDataFiles:
    def test_yahoo_raw_exists(self):
        path = PROJECT_ROOT / "data" / "raw" / "yahoo_commodities.csv"
        assert path.exists(), "Yahoo raw data not found. Run collect_data.py first."

    def test_fred_raw_exists(self):
        path = PROJECT_ROOT / "data" / "raw" / "fred_commodities.csv"
        assert path.exists(), "FRED raw data not found. Run collect_data.py first."

    def test_news_raw_exists(self):
        path = PROJECT_ROOT / "data" / "raw" / "news_headlines.csv"
        assert path.exists(), "News raw data not found. Run collect_data.py first."

    def test_yahoo_cleaned_exists(self):
        path = PROJECT_ROOT / "data" / "processed" / "yahoo_cleaned.csv"
        assert path.exists(), "Yahoo cleaned data not found. Run process_data.py first."

    def test_fred_cleaned_exists(self):
        path = PROJECT_ROOT / "data" / "processed" / "fred_cleaned.csv"
        assert path.exists(), "FRED cleaned data not found. Run process_data.py first."

    def test_yahoo_features_exists(self):
        path = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"
        assert path.exists(), "Yahoo features not found. Run process_data.py --step features first."

    def test_yahoo_anomalies_exists(self):
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        assert path.exists(), "Yahoo anomalies not found. Run detect_anomalies.py first."

    def test_forecast_prophet_exists(self):
        path = PROJECT_ROOT / "data" / "forecasts" / "prophet_yahoo.csv"
        assert path.exists(), "Prophet forecast not found. Run run_forecast.py first."


# ========== Data Quality Tests ==========

class TestDataQuality:
    @pytest.fixture
    def yahoo_raw(self):
        path = PROJECT_ROOT / "data" / "raw" / "yahoo_commodities.csv"
        if not path.exists():
            pytest.skip("Yahoo raw data not available")
        return pd.read_csv(path, parse_dates=["date"])

    @pytest.fixture
    def fred_raw(self):
        path = PROJECT_ROOT / "data" / "raw" / "fred_commodities.csv"
        if not path.exists():
            pytest.skip("FRED raw data not available")
        return pd.read_csv(path, parse_dates=["date"])

    @pytest.fixture
    def yahoo_cleaned(self):
        path = PROJECT_ROOT / "data" / "processed" / "yahoo_cleaned.csv"
        if not path.exists():
            pytest.skip("Yahoo cleaned data not available")
        return pd.read_csv(path, parse_dates=["date"])

    def test_yahoo_has_required_columns(self, yahoo_raw):
        required = ["date", "commodity", "close"]
        for col in required:
            assert col in yahoo_raw.columns, f"Missing column: {col}"

    def test_fred_has_required_columns(self, fred_raw):
        required = ["date", "commodity", "price"]
        for col in required:
            assert col in fred_raw.columns, f"Missing column: {col}"

    def test_yahoo_no_negative_prices(self, yahoo_raw):
        assert (yahoo_raw["close"] >= 0).all(), "Negative prices found"

    def test_fred_no_negative_prices(self, fred_raw):
        assert (fred_raw["price"] >= 0).all(), "Negative prices found"

    def test_yahoo_date_range(self, yahoo_raw):
        min_date = yahoo_raw["date"].min()
        max_date = yahoo_raw["date"].max()
        date_span = (max_date - min_date).days
        assert date_span > 365, f"Less than 1 year of data: {date_span} days"

    def test_yahoo_has_multiple_commodities(self, yahoo_raw):
        assert yahoo_raw["commodity"].nunique() >= 2, "Need at least 2 commodities"

    def test_cleaned_no_nulls_in_price(self, yahoo_cleaned):
        assert yahoo_cleaned["close"].notna().all(), "Null prices after cleaning"

    def test_cleaned_dates_sorted(self, yahoo_cleaned):
        for commodity in yahoo_cleaned["commodity"].unique():
            cdf = yahoo_cleaned[yahoo_cleaned["commodity"] == commodity]
            dates = cdf["date"].tolist()
            assert dates == sorted(dates), f"Dates not sorted for {commodity}"


# ========== Collector Tests ==========

class TestCollectors:
    def test_yahoo_collector_init(self):
        collector = YahooCollector()
        assert collector is not None

    def test_fred_collector_init(self):
        collector = FredCollector()
        assert collector is not None

    def test_news_collector_init(self):
        collector = NewsCollector()
        assert collector is not None

    def test_yahoo_collector_works(self):
        collector = YahooCollector()
        assert collector is not None


# ========== Cleaner Tests ==========

class TestCleaner:
    def test_cleaner_init(self):
        cleaner = DataCleaner()
        assert cleaner is not None

    def test_clean_yahoo_returns_dataframe(self):
        cleaner = DataCleaner()
        path = PROJECT_ROOT / "data" / "raw" / "yahoo_commodities.csv"
        if not path.exists():
            pytest.skip("Yahoo raw data not available")
        result = cleaner.clean_yahoo()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ========== Feature Engineering Tests ==========

class TestFeatureEngineer:
    def test_engineer_init(self):
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_technical_features_added(self):
        path = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"
        if not path.exists():
            pytest.skip("Yahoo features not available")
        df = pd.read_csv(path)
        expected_features = ["return_1d", "sma_7", "ema_7", "volatility_7d", "rsi_14"]
        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_features_no_inf(self):
        path = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"
        if not path.exists():
            pytest.skip("Yahoo features not available")
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Inf values found in features"


# ========== Anomaly Detection Tests ==========

class TestAnomalyDetector:
    def test_detector_init(self):
        detector = AnomalyDetector()
        assert detector is not None

    def test_anomaly_columns_exist(self):
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        if not path.exists():
            pytest.skip("Yahoo anomalies not available")
        df = pd.read_csv(path)
        expected = ["anomaly_zscore", "anomaly_iforest", "anomaly_combined", "anomaly_severity"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_anomaly_values_binary(self):
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        if not path.exists():
            pytest.skip("Yahoo anomalies not available")
        df = pd.read_csv(path)
        assert set(df["anomaly_combined"].unique()).issubset({0, 1})

    def test_severity_values_valid(self):
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        if not path.exists():
            pytest.skip("Yahoo anomalies not available")
        df = pd.read_csv(path)
        anomalies = df[df["anomaly_combined"] == 1]
        valid = {"high", "medium"}
        assert set(anomalies["anomaly_severity"].unique()).issubset(valid)


# ========== Forecaster Tests ==========

class TestForecaster:
    def test_prophet_init(self):
        config = load_config()
        forecaster = ProphetForecaster(config)
        assert forecaster is not None

    def test_lgbm_init(self):
        config = load_config()
        forecaster = LightGBMForecaster(config)
        assert forecaster is not None

    def test_prophet_forecast_has_columns(self):
        path = PROJECT_ROOT / "data" / "forecasts" / "prophet_yahoo.csv"
        if not path.exists():
            pytest.skip("Prophet forecast not available")
        df = pd.read_csv(path)
        expected = ["ds", "yhat", "commodity"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_forecast_no_negative_predictions(self):
        path = PROJECT_ROOT / "data" / "forecasts" / "prophet_yahoo.csv"
        if not path.exists():
            pytest.skip("Prophet forecast not available")
        df = pd.read_csv(path)
        negative_ratio = (df["yhat"] < 0).mean()
        assert negative_ratio < 0.05, f"Too many negative predictions: {negative_ratio:.1%}"


# ========== Integration Tests ==========

class TestIntegration:
    def test_full_pipeline_data_flow(self):
        stages = [
            ("Raw Yahoo", "data/raw/yahoo_commodities.csv"),
            ("Cleaned Yahoo", "data/processed/yahoo_cleaned.csv"),
            ("Features Yahoo", "data/features/yahoo_features.csv"),
            ("Anomalies Yahoo", "data/anomalies/yahoo_anomalies.csv"),
            ("Forecast Prophet", "data/forecasts/prophet_yahoo.csv"),
        ]
        for name, rel_path in stages:
            path = PROJECT_ROOT / rel_path
            assert path.exists(), f"Pipeline break at {name}: {rel_path} not found"
            df = pd.read_csv(path)
            assert len(df) > 0, f"Empty file at {name}"

    def test_commodity_consistency(self):
        features_path = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"
        anomalies_path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"

        if not features_path.exists() or not anomalies_path.exists():
            pytest.skip("Pipeline data not available")

        features_commodities = set(pd.read_csv(features_path)["commodity"].unique())
        anomalies_commodities = set(pd.read_csv(anomalies_path)["commodity"].unique())

        assert features_commodities == anomalies_commodities, \
            f"Commodity mismatch: features={features_commodities}, anomalies={anomalies_commodities}"

    def test_anomaly_count_reasonable(self):
        path = PROJECT_ROOT / "data" / "anomalies" / "yahoo_anomalies.csv"
        if not path.exists():
            pytest.skip("Anomalies not available")
        df = pd.read_csv(path)
        anomaly_rate = df["anomaly_combined"].mean()
        assert anomaly_rate < 0.20, f"Anomaly rate too high: {anomaly_rate:.1%}"
        assert anomaly_rate > 0.001, f"Anomaly rate suspiciously low: {anomaly_rate:.1%}"