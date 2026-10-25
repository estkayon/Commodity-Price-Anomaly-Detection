"""NewsAPI collector for commodity-related news headlines."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config_loader import CONFIG, PROJECT_ROOT

load_dotenv()
logger = logging.getLogger(__name__)

COMMODITY_KEYWORDS = [
    "crude oil price",
    "natural gas price",
    "gold price",
    "wheat price",
    "rice price",
    "sugar price",
    "commodity market",
    "oil futures",
    "gold futures",
]


class NewsCollector:
    """Collect commodity-related news from NewsAPI."""

    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not set in .env")
        self.base_url = "https://newsapi.org/v2/everything"
        self.raw_dir = PROJECT_ROOT / CONFIG["data"]["raw_dir"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def collect_single(
        self, query: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Collect news for a single query keyword."""
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting news for '{query}' from {from_date} to {to_date}")

        try:
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": self.api_key,
            }
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok" or not data.get("articles"):
                logger.warning(f"No articles for '{query}'")
                return pd.DataFrame()

            articles = data["articles"]
            df = pd.DataFrame({
                "date": [a["publishedAt"] for a in articles],
                "title": [a["title"] for a in articles],
                "description": [a.get("description", "") for a in articles],
                "source": [a["source"]["name"] for a in articles],
                "url": [a["url"] for a in articles],
                "query": query,
            })

            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.dropna(subset=["title"])
            logger.info(f"Collected {len(df)} articles for '{query}'")
            return df

        except Exception as e:
            logger.error(f"Error collecting news for '{query}': {e}")
            return pd.DataFrame()

    def collect_all(self) -> pd.DataFrame:
        """Collect news for all commodity keywords."""
        all_data = []
        for keyword in COMMODITY_KEYWORDS:
            df = self.collect_single(keyword)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            logger.warning("No news collected from NewsAPI")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=["title", "url"], keep="first")
        combined = combined.sort_values("date", ascending=False).reset_index(drop=True)

        output_path = self.raw_dir / "news_headlines.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} articles to {output_path}")
        return combined
