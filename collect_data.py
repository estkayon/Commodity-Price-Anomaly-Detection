"""
PriceRadar - Data Collection Script

Usage:
    python collect_data.py
    python collect_data.py --source yahoo
    python collect_data.py --source fred
    python collect_data.py --source news
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_yahoo():
    from src.data.yahoo_collector import YahooCollector
    logger.info("=" * 50)
    logger.info("Collecting Yahoo Finance data...")
    yc = YahooCollector()
    df = yc.collect_all()
    if not df.empty:
        logger.info(f"Yahoo: {len(df)} rows | Commodities: {df['commodity'].unique().tolist()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        logger.warning("Yahoo: No data collected")
    return df


def collect_fred():
    from src.data.fred_collector import FredCollector
    logger.info("=" * 50)
    logger.info("Collecting FRED data...")
    fc = FredCollector()
    df = fc.collect_all()
    if not df.empty:
        logger.info(f"FRED: {len(df)} rows | Commodities: {df['commodity'].unique().tolist()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        logger.warning("FRED: No data collected")
    return df


def collect_news():
    from src.news.news_collector import NewsCollector
    logger.info("=" * 50)
    logger.info("Collecting NewsAPI data...")
    nc = NewsCollector()
    df = nc.collect_all()
    if not df.empty:
        logger.info(f"News: {len(df)} articles | Sources: {df['source'].nunique()}")
    else:
        logger.warning("News: No data collected")
    return df


def main():
    parser = argparse.ArgumentParser(description="PriceRadar Data Collection")
    parser.add_argument(
        "--source",
        choices=["yahoo", "fred", "news", "all"],
        default="all",
        help="Which source to collect (default: all)",
    )
    args = parser.parse_args()

    logger.info("PriceRadar - Data Collection")
    logger.info("=" * 50)

    results = {}

    if args.source in ("yahoo", "all"):
        results["yahoo"] = collect_yahoo()

    if args.source in ("fred", "all"):
        results["fred"] = collect_fred()

    if args.source in ("news", "all"):
        results["news"] = collect_news()

    logger.info("=" * 50)
    logger.info("SUMMARY")
    for name, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"  {name}: {len(df)} rows -> data/raw/")
        else:
            logger.info(f"  {name}: FAILED")
    logger.info("Done! Check data/raw/ for output files.")


if __name__ == "__main__":
    main()
