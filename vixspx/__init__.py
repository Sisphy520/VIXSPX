"""vixspx: analyse the relationship between the VIX and S&P 500 returns."""
from .config import DATA_FILE, OUTPUT_DIR, HORIZONS
from .data import download_data, load_data
from .features import add_features
from .prediction import (
    predictive_regression,
    regression_with_realized_vol,
    bucket_forward_returns,
    out_of_sample_r2,
    subsample_stability,
)
from .events import find_events, summarize_events, bootstrap_event_mean

__all__ = [
    "DATA_FILE", "OUTPUT_DIR", "HORIZONS",
    "download_data", "load_data", "add_features",
    "predictive_regression", "regression_with_realized_vol",
    "bucket_forward_returns", "out_of_sample_r2", "subsample_stability",
    "find_events", "summarize_events", "bootstrap_event_mean",
]
