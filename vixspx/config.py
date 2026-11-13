"""Project-wide paths and constants (independent of the working directory)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_FILE = DATA_DIR / "data.csv"

START_DATE = "1990-01-01"      # the current VIX methodology starts in 1990
TICKERS = {"SPX": "^GSPC", "VIX": "^VIX"}
HORIZONS = [1, 5, 21, 63]      # forward-return horizons in trading days
EVENT_HORIZONS = [1, 5, 21, 63, 126, 252]
