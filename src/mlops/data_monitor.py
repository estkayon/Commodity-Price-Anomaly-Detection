"""Data quality monitoring using Evidently."""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from src.config_loader import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class DataMonitor:
    """Monitor data quality and drift using Evidently."""

    def __init__(self):
        self.reports_dir = PROJECT_ROOT / "reports" / "monitoring"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check_data_quality(self, df: pd.DataFrame, name: str) -> dict:
        """Run data quality checks on a dataset."""
        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=None, current_data=df)

        # Save HTML report
        report_path = self.reports_dir / f"quality_{name}_{datetime.now().strftime('%Y%m%d')}.html"
        report.save_html(str(report_path))
        logger.info(f"Data quality report saved: {report_path}")

        # Extract summary
        result = report.as_dict()
        summary = {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "report_path": str(report_path),
        }
        return summary

    def check_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, name: str) -> dict:
        """Check for data drift between reference and current data."""
        # Use only numeric columns both datasets share
        common_cols = [c for c in reference_df.select_dtypes(include="number").columns
                       if c in current_df.columns]

        if not common_cols:
            logger.warning(f"No common numeric columns for drift check: {name}")
            return {"name": name, "drift_detected": False, "error": "no common columns"}

        ref = reference_df[common_cols].dropna()
        cur = current_df[common_cols].dropna()

        if len(ref) < 10 or len(cur) < 10:
            logger.warning(f"Not enough data for drift check: {name}")
            return {"name": name, "drift_detected": False, "error": "insufficient data"}

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        report_path = self.reports_dir / f"drift_{name}_{datetime.now().strftime('%Y%m%d')}.html"
        report.save_html(str(report_path))
        logger.info(f"Data drift report saved: {report_path}")

        result = report.as_dict()
        summary = {
            "name": name,
            "report_path": str(report_path),
        }

        # Try to extract drift info
        try:
            metrics = result.get("metrics", [])
            for m in metrics:
                mr = m.get("result", {})
                if "drift_share" in mr:
                    summary["drift_share"] = mr["drift_share"]
                    summary["drift_detected"] = mr.get("dataset_drift", False)
                    summary["n_drifted"] = mr.get("number_of_drifted_columns", 0)
                    break
        except Exception:
            summary["drift_detected"] = False

        return summary

    def monitor_yahoo(self) -> list:
        """Run monitoring on Yahoo data."""
        processed = PROJECT_ROOT / "data" / "processed" / "yahoo_cleaned.csv"
        features = PROJECT_ROOT / "data" / "features" / "yahoo_features.csv"

        results = []

        if processed.exists():
            df = pd.read_csv(processed, parse_dates=["date"])
            results.append(self.check_data_quality(df, "yahoo_cleaned"))

        if features.exists():
            df = pd.read_csv(features, parse_dates=["date"])
            results.append(self.check_data_quality(df, "yahoo_features"))

            # Drift check: first half vs second half
            mid = len(df) // 2
            ref = df.iloc[:mid]
            cur = df.iloc[mid:]
            results.append(self.check_data_drift(ref, cur, "yahoo_time_drift"))

        return results

    def monitor_fred(self) -> list:
        """Run monitoring on FRED data."""
        processed = PROJECT_ROOT / "data" / "processed" / "fred_cleaned.csv"
        features = PROJECT_ROOT / "data" / "features" / "fred_features.csv"

        results = []

        if processed.exists():
            df = pd.read_csv(processed, parse_dates=["date"])
            results.append(self.check_data_quality(df, "fred_cleaned"))

        if features.exists():
            df = pd.read_csv(features, parse_dates=["date"])
            results.append(self.check_data_quality(df, "fred_features"))

        return results

    def monitor_all(self) -> list:
        """Run all monitoring checks."""
        results = []
        results.extend(self.monitor_yahoo())
        results.extend(self.monitor_fred())
        return results