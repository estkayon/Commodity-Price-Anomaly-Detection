"""
PriceRadar - Data Processing & Feature Engineering

Usage:
    python process_data.py
    python process_data.py --step clean
    python process_data.py --step features
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_cleaning():
    from src.features.data_cleaner import DataCleaner
    logger.info("=" * 50)
    logger.info("STEP 1: Cleaning raw data...")
    cleaner = DataCleaner()
    results = cleaner.clean_all()
    for name, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"  {name}: {len(df)} cleaned rows")
        else:
            logger.info(f"  {name}: no data")
    return results


def run_features():
    from src.features.feature_engineer import FeatureEngineer
    logger.info("=" * 50)
    logger.info("STEP 2: Engineering features...")
    fe = FeatureEngineer()
    results = fe.engineer_all()
    for name, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"  {name}: {len(df)} rows x {df.shape[1]} features")
        else:
            logger.info(f"  {name}: no data")
    return results


def main():
    parser = argparse.ArgumentParser(description="PriceRadar Data Processing")
    parser.add_argument(
        "--step",
        choices=["clean", "features", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("PriceRadar - Data Processing")
    logger.info("=" * 50)

    if args.step in ("clean", "all"):
        run_cleaning()

    if args.step in ("features", "all"):
        run_features()

    logger.info("=" * 50)
    logger.info("Done! Check data/processed/ and data/features/")


if __name__ == "__main__":
    main()