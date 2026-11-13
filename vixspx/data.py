"""Download and load SPX / VIX daily closing prices."""
import pandas as pd

from .config import DATA_FILE, START_DATE, TICKERS


def download_data(start: str = START_DATE, path=DATA_FILE) -> pd.DataFrame:
    """Download daily closes from Yahoo Finance, save to CSV, return the frame."""
    import yfinance as yf

    raw = yf.download(list(TICKERS.values()), start=start,
                      auto_adjust=True, progress=False)
    close = raw["Close"][list(TICKERS.values())].copy()
    close.columns = list(TICKERS.keys())

    print("rows:", len(close), "| range:", close.index.min().date(),
          "to", close.index.max().date())
    print("missing values:\n", close.isna().sum(), "\n")

    close = close.dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(path)
    return close


def load_data(path=DATA_FILE) -> pd.DataFrame:
    """Load the saved CSV (columns: SPX, VIX; DatetimeIndex)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(path, index_col=0, parse_dates=True).dropna()
