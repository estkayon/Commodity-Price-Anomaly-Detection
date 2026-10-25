"""
PriceRadar - Anomaly Detection

Usage:
    python detect_anomalies.py
    python detect_anomalies.py --source yahoo
    python detect_anomalies.py --source fred
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


def main():
    parser = argparse.ArgumentParser(description="PriceRadar Anomaly Detection")
    parser.add_argument(
        "--source",
        choices=["yahoo", "fred", "all"],
        default="all",
        help="Which dataset to analyze (default: all)",
    )
    args = parser.parse_args()

    from src.anomaly.detector import AnomalyDetector
    detector = AnomalyDetector()

    logger.info("PriceRadar - Anomaly Detection")
    logger.info("=" * 50)

    results = {}

    if args.source in ("yahoo", "all"):
        results["yahoo"] = detector.detect_yahoo()

    if args.source in ("fred", "all"):
        results["fred"] = detector.detect_fred()

    logger.info("=" * 50)
    logger.info("SUMMARY")
    for name, df in results.items():
        if df is not None and not df.empty:
            total = df["anomaly_combined"].sum()
            high = (df["anomaly_severity"] == "high").sum()
            logger.info(f"  {name}: {total} anomalies ({high} high severity) out of {len(df)} rows")
        else:
            logger.info(f"  {name}: FAILED")

    logger.info("Done! Check data/anomalies/")


if __name__ == "__main__":
    main()