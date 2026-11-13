"""Event study around extreme VIX readings."""
import numpy as np
import pandas as pd


def add_drawdown(d: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Add ``mdd{window}``: worst close over the next ``window`` days vs today (%)."""
    d = d.copy()
    spx = d.SPX.values
    out = np.full(len(d), np.nan)
    for i in range(len(d) - window):
        out[i] = (spx[i + 1:i + window + 1].min() / spx[i] - 1) * 100
    d[f"mdd{window}"] = out
    return d


def find_events(d: pd.DataFrame, threshold: float, cooldown: int = 60) -> pd.DataFrame:
    """De-clustered events: the first day VIX >= threshold.

    A new event is allowed only after VIX has stayed below the threshold for
    more than ``cooldown`` trading days since the last day above it.
    """
    idx, last_above, active = [], -10**9, False
    for i, v in enumerate(d.VIX.values):
        if v >= threshold:
            if not active and i - last_above > cooldown:
                idx.append(i)
            active, last_above = True, i
        else:
            active = False
    return d.iloc[idx]


def summarize_events(events: pd.DataFrame, unconditional: pd.Series, horizons) -> pd.DataFrame:
    """Mean / median / hit-rate of forward returns after events vs unconditional mean."""
    cols = [f"f{h}" for h in horizons]
    return pd.DataFrame({
        "mean": events[cols].mean(),
        "median": events[cols].median(),
        "win%": (events[cols] > 0).mean() * 100,
        "uncond": unconditional[cols].values,
    })


def bootstrap_event_mean(events: pd.DataFrame, h: int, n: int = 10000, seed: int = 0):
    """Event-level bootstrap of the mean forward return. Returns (mean, lo, hi, n_events)."""
    v = events[f"f{h}"].dropna().values
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)
