"""FRED API data collector for agricultural commodity prices."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

from src.config_loader import CONFIG, PROJECT_ROOT

load_dotenv()
logger = logging.getLogger(__name__)


class FredCollector:
    """Collect monthly agricultural commodity prices from FRED."""

    def __init__(self):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY not set in .env")
        self.fred = Fred(api_key=api_key)
        self.commodities = CONFIG["commodities"]["fred"]
        self.history_years = CONFIG["data"]["history_years"]
        self.raw_dir = PROJECT_ROOT / CONFIG["data"]["raw_dir"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def collect_single(
        self, series_id: str, name: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Collect data for a single FRED series."""
        if start is None:
            start = (datetime.now() - timedelta(days=365 * self.history_years)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting {name} ({series_id}) from {start} to {end}")

        try:
            series = self.fred.get_series(series_id, observation_start=start, observation_end=end)

            if series.empty:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame()

            df = pd.DataFrame({
                "date": series.index,
                "price": series.values,
                "commodity": name,
                "series_id": series_id,
                "source": "fred",
            })

            df = df.dropna(subset=["price"])
            df["date"] = pd.to_datetime(df["date"])

            logger.info(f"Collected {len(df)} rows for {name}")
            return df

        except Exception as e:
            logger.error(f"Error collecting {series_id}: {e}")
            return pd.DataFrame()

    def collect_all(self) -> pd.DataFrame:
        """Collect data for all configured FRED commodities."""
        all_data = []
        for commodity in self.commodities:
            df = self.collect_single(commodity["series_id"], commodity["name"])
            if not df.empty:
                all_data.append(df)

        if not all_data:
            logger.warning("No data collected from FRED")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        output_path = self.raw_dir / "fred_commodities.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} rows to {output_path}")
        return combined
