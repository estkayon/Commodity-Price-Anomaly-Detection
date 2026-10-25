"""
PriceRadar - Forecasting Pipeline

Usage:
    python run_forecast.py
    python run_forecast.py --model prophet
    python run_forecast.py --model lgbm
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


def run_prophet():
    from src.models.forecaster import ProphetForecaster
    logger.info("=" * 50)
    logger.info("Running Prophet forecasts...")
    pf = ProphetForecaster()
    yahoo = pf.forecast_yahoo()
    fred = pf.forecast_fred()
    return {"prophet_yahoo": yahoo, "prophet_fred": fred}


def run_lgbm():
    from src.models.forecaster import LightGBMForecaster
    logger.info("=" * 50)
    logger.info("Running LightGBM forecasts...")
    lf = LightGBMForecaster()
    yahoo = lf.forecast_yahoo()
    fred = lf.forecast_fred()
    return {"lgbm_yahoo": yahoo, "lgbm_fred": fred}


def main():
    parser = argparse.ArgumentParser(description="PriceRadar Forecasting")
    parser.add_argument(
        "--model",
        choices=["prophet", "lgbm", "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("PriceRadar - Forecasting Pipeline")
    logger.info("=" * 50)

    results = {}

    if args.model in ("prophet", "all"):
        results.update(run_prophet())

    if args.model in ("lgbm", "all"):
        results.update(run_lgbm())

    logger.info("=" * 50)
    logger.info("SUMMARY")
    for name, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"  {name}: {len(df)} forecast rows")
        else:
            logger.info(f"  {name}: FAILED or empty")

    logger.info("Done! Check data/forecasts/")


if __name__ == "__main__":
    main()