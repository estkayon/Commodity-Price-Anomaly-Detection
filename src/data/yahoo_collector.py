"""Yahoo Finance data collector for commodity prices."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config_loader import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

# yfinance 0.2.31 sometimes fails with futures (=F) tickers.
# Fallback: use ETF proxies that track the same commodities.
TICKER_FALLBACKS = {
    "CL=F": ["USO", "BNO"],      # Crude Oil ETFs
    "NG=F": ["UNG", "BOIL"],     # Natural Gas ETFs
    "GC=F": ["GLD", "IAU"],      # Gold ETFs
}


class YahooCollector:
    """Collect daily commodity prices from Yahoo Finance."""

    def __init__(self):
        self.commodities = CONFIG["commodities"]["yahoo"]
        self.history_years = CONFIG["data"]["history_years"]
        self.raw_dir = PROJECT_ROOT / CONFIG["data"]["raw_dir"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _try_download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Try downloading a single ticker, return empty DataFrame on failure."""
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
        except Exception as e:
            logger.warning(f"Download failed for {ticker}: {e}")
            return pd.DataFrame()

    def collect_single(
        self, ticker: str, name: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Collect data for a single commodity, with ETF fallback."""
        if start is None:
            start = (datetime.now() - timedelta(days=365 * self.history_years)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting {name} ({ticker}) from {start} to {end}")

        # Try primary ticker first
        data = self._try_download(ticker, start, end)

        # If failed, try fallbacks
        used_ticker = ticker
        if data.empty and ticker in TICKER_FALLBACKS:
            for fallback in TICKER_FALLBACKS[ticker]:
                logger.info(f"Primary {ticker} failed, trying fallback {fallback}...")
                data = self._try_download(fallback, start, end)
                if not data.empty:
                    used_ticker = fallback
                    logger.info(f"Fallback {fallback} succeeded for {name}")
                    break

        if data.empty:
            logger.warning(f"No data for {name} (tried {ticker} + fallbacks)")
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": data.index,
            "open": data["Open"].values,
            "high": data["High"].values,
            "low": data["Low"].values,
            "close": data["Close"].values,
            "volume": data["Volume"].values,
            "commodity": name,
            "ticker": used_ticker,
            "original_ticker": ticker,
            "source": "yahoo_finance",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        logger.info(f"Collected {len(df)} rows for {name} (ticker: {used_ticker})")
        return df

    def collect_all(self) -> pd.DataFrame:
        """Collect data for all configured Yahoo commodities."""
        all_data = []
        for commodity in self.commodities:
            df = self.collect_single(commodity["ticker"], commodity["name"])
            if not df.empty:
                all_data.append(df)

        if not all_data:
            logger.warning("No data collected from Yahoo Finance")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        output_path = self.raw_dir / "yahoo_commodities.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} rows to {output_path}")
        return combined