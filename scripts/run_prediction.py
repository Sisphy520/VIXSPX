"""Can the VIX predict future S&P 500 returns? Prints tables, saves outputs/predict.png."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vixspx import (HORIZONS, load_data, add_features, predictive_regression,
                    regression_with_realized_vol, bucket_forward_returns,
                    out_of_sample_r2, subsample_stability)
from vixspx.plotting import plot_prediction


def main():
    d = add_features(load_data(), HORIZONS)

    print("=== 1. forward return ~ lnVIX (Newey-West HAC) ===")
    print(predictive_regression(d, HORIZONS).round(3).to_string(index=False))

    print("\n=== 2. adding 21d realized vol ===")
    print(regression_with_realized_vol(d, HORIZONS).round(3).to_string(index=False))

    print("\n=== 3. mean forward return (%) by VIX bucket ===")
    buckets = bucket_forward_returns(d, HORIZONS)
    print(buckets.round(2).to_string())

    print("\n=== 4. out-of-sample R2 (expanding window vs historical mean) ===")
    print(out_of_sample_r2(d, HORIZONS).round(4).to_string(index=False))

    print("\n=== 5. subsample stability (h=21) ===")
    print(subsample_stability(d).round(3).to_string(index=False))

    plot_prediction(d, buckets, HORIZONS)


if __name__ == "__main__":
    main()
