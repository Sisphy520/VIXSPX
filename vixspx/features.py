"""Feature construction: returns, realized vol, forward returns."""
import numpy as np
import pandas as pd


def add_features(d: pd.DataFrame, horizons, rv_window: int = 21) -> pd.DataFrame:
    """Add log VIX, daily log return, realized vol, VRP proxy and forward returns.

    Forward return ``f{h}`` is the log return (in %) from the close of day t
    to the close of day t+h, so it is not known at time t (no look-ahead if
    VIX_t is used as the predictor).
    """
    d = d.copy()
    lp = np.log(d.SPX)
    d["lnVIX"] = np.log(d.VIX)
    d["r"] = lp.diff()
    d["rv21"] = d.r.rolling(rv_window).std() * np.sqrt(252) * 100  # annualised, %
    d["vrp"] = d.VIX - d.rv21                                     # simple VRP proxy
    for h in horizons:
        d[f"f{h}"] = (lp.shift(-h) - lp) * 100
    return d
