"""Download SPX and VIX daily closes to data/data.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vixspx import download_data

if __name__ == "__main__":
    df = download_data()
    print(df.describe().round(2))
