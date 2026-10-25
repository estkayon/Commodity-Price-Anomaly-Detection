"""Configuration loader for PriceRadar."""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY not set in .env")
    return key


def get_news_api_key() -> str:
    key = os.getenv("NEWS_API_KEY")
    if not key:
        raise ValueError("NEWS_API_KEY not set in .env")
    return key


CONFIG = load_config()
