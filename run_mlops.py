"""
PriceRadar - MLOps Pipeline

Usage:
    python run_mlops.py
    python run_mlops.py --step track
    python run_mlops.py --step monitor
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


def run_tracking():
    from src.mlops.experiment_tracker import ExperimentTracker
    logger.info("=" * 50)
    logger.info("Logging experiments to MLflow...")
    tracker = ExperimentTracker()
    tracker.log_all_forecasts()
    tracker.log_anomaly_summary()
    logger.info("MLflow tracking complete. Run 'mlflow ui' to view.")


def run_monitoring():
    from src.mlops.data_monitor import DataMonitor
    logger.info("=" * 50)
    logger.info("Running data monitoring (Evidently)...")
    monitor = DataMonitor()
    results = monitor.monitor_all()
    for r in results:
        logger.info(f"  {r.get('name', 'unknown')}: {r}")
    logger.info("Monitoring reports saved to reports/monitoring/")


def main():
    parser = argparse.ArgumentParser(description="PriceRadar MLOps Pipeline")
    parser.add_argument(
        "--step",
        choices=["track", "monitor", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("PriceRadar - MLOps Pipeline")
    logger.info("=" * 50)

    if args.step in ("track", "all"):
        run_tracking()

    if args.step in ("monitor", "all"):
        run_monitoring()

    logger.info("=" * 50)
    logger.info("Done!")


if __name__ == "__main__":
    main()