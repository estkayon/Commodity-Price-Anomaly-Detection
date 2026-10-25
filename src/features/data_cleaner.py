"""Data cleaning and preprocessing for raw commodity data."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config_loader import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess raw commodity price data."""

    def __init__(self):
        self.raw_dir = PROJECT_ROOT / CONFIG["data"]["raw_dir"]
        self.processed_dir = PROJECT_ROOT / CONFIG["data"]["processed_dir"]
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_yahoo(self) -> pd.DataFrame:
        """Clean Yahoo Finance commodity data."""
        path = self.raw_dir / "yahoo_commodities.csv"
        if not path.exists():
            logger.warning("yahoo_commodities.csv not found")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded Yahoo data: {len(df)} rows")

        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["date", "commodity"], keep="last")
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} duplicate rows")

        # Sort by commodity and date
        df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

        # Remove rows where close price is zero or negative
        df = df[df["close"] > 0]

        # Handle missing values per commodity
        cleaned = []
        for commodity in df["commodity"].unique():
            cdf = df[df["commodity"] == commodity].copy()

            # Forward fill small gaps (up to 5 days - weekends/holidays)
            cdf = cdf.set_index("date")
            cdf = cdf.asfreq("D")  # fill missing calendar days
            cdf["commodity"] = cdf["commodity"].fillna(commodity)
            cdf["ticker"] = cdf["ticker"].ffill()
            cdf["source"] = cdf["source"].fillna("yahoo_finance")

            # Forward fill price columns (max 5 days gap)
            price_cols = ["open", "high", "low", "close", "volume"]
            cdf[price_cols] = cdf[price_cols].ffill(limit=5)

            # Drop remaining NaN rows (larger gaps)
            cdf = cdf.dropna(subset=["close"])
            cdf = cdf.reset_index()

            # Remove extreme outliers using IQR method per commodity
            q1 = cdf["close"].quantile(0.01)
            q99 = cdf["close"].quantile(0.99)
            outlier_mask = (cdf["close"] >= q1) & (cdf["close"] <= q99)
            removed = (~outlier_mask).sum()
            if removed > 0:
                logger.info(f"{commodity}: removed {removed} extreme outliers")
            cdf = cdf[outlier_mask]

            cleaned.append(cdf)

        result = pd.concat(cleaned, ignore_index=True)

        output_path = self.processed_dir / "yahoo_cleaned.csv"
        result.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned Yahoo data: {len(result)} rows -> {output_path}")
        return result

    def clean_fred(self) -> pd.DataFrame:
        """Clean FRED commodity data."""
        path = self.raw_dir / "fred_commodities.csv"
        if not path.exists():
            logger.warning("fred_commodities.csv not found")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded FRED data: {len(df)} rows")

        # Remove duplicates
        df = df.drop_duplicates(subset=["date", "commodity"], keep="last")

        # Sort
        df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

        # Remove zero/negative prices
        df = df[df["price"] > 0]

        # Forward fill small gaps (up to 2 months for monthly data)
        cleaned = []
        for commodity in df["commodity"].unique():
            cdf = df[df["commodity"] == commodity].copy()
            cdf = cdf.set_index("date")
            cdf = cdf.asfreq("MS")  # month start frequency
            cdf["commodity"] = cdf["commodity"].fillna(commodity)
            cdf["series_id"] = cdf["series_id"].ffill()
            cdf["source"] = cdf["source"].fillna("fred")
            cdf["price"] = cdf["price"].ffill(limit=2)
            cdf = cdf.dropna(subset=["price"])
            cdf = cdf.reset_index()
            cleaned.append(cdf)

        result = pd.concat(cleaned, ignore_index=True)

        output_path = self.processed_dir / "fred_cleaned.csv"
        result.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned FRED data: {len(result)} rows -> {output_path}")
        return result

    def clean_news(self) -> pd.DataFrame:
        """Clean news headlines data."""
        path = self.raw_dir / "news_headlines.csv"
        if not path.exists():
            logger.warning("news_headlines.csv not found")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded News data: {len(df)} rows")

        # Remove duplicates by title
        before = len(df)
        df = df.drop_duplicates(subset=["title"], keep="first")
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} duplicate articles")

        # Remove articles with empty/null titles
        df = df.dropna(subset=["title"])
        df = df[df["title"].str.strip().str.len() > 10]  # min 10 chars

        # Remove [Removed] articles (NewsAPI placeholder)
        df = df[~df["title"].str.contains(r"\[Removed\]", case=False, na=False)]

        # Clean description
        df["description"] = df["description"].fillna("")

        # Sort by date
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        output_path = self.processed_dir / "news_cleaned.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned News data: {len(df)} rows -> {output_path}")
        return df

    def clean_all(self) -> dict:
        """Clean all raw datasets."""
        results = {}
        results["yahoo"] = self.clean_yahoo()
        results["fred"] = self.clean_fred()
        results["news"] = self.clean_news()
        return results