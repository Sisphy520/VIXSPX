"""Predictive regressions: can the VIX forecast future S&P 500 returns?"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _ols_hac(y: pd.Series, X: pd.DataFrame, h: int):
    """OLS with Newey-West (HAC) errors; overlapping returns need maxlags ~ 2h."""
    m = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(m.iloc[:, 0], sm.add_constant(m.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(h, 1) * 2})


def predictive_regression(d, horizons, x: str = "lnVIX") -> pd.DataFrame:
    """f{h} ~ x for each horizon."""
    rows = []
    for h in horizons:
        r = _ols_hac(d[f"f{h}"], d[[x]], h)
        rows.append([h, r.params[x], r.tvalues[x], r.rsquared])
    return pd.DataFrame(rows, columns=["h", "b", "t_HAC", "R2"])


def regression_with_realized_vol(d, horizons) -> pd.DataFrame:
    """f{h} ~ VIX + realized vol: which one carries the information?"""
    rows = []
    for h in horizons:
        r = _ols_hac(d[f"f{h}"], d[["VIX", "rv21"]], h)
        rows.append([h, r.params["VIX"], r.tvalues["VIX"],
                     r.params["rv21"], r.tvalues["rv21"], r.rsquared])
    return pd.DataFrame(rows, columns=["h", "b_VIX", "t_VIX", "b_RV", "t_RV", "R2"])


def bucket_forward_returns(d, horizons, edges=(0, 15, 20, 25, 35, 100),
                           labels=("<15", "15-20", "20-25", "25-35", ">35")) -> pd.DataFrame:
    """Mean forward return (%) by VIX level bucket."""
    b = pd.cut(d.VIX, list(edges), labels=list(labels))
    return d.groupby(b, observed=True).agg(
        days=("VIX", "size"), **{f"f{h}": (f"f{h}", "mean") for h in horizons})


def out_of_sample_r2(d, horizons, x: str = "lnVIX", start: int = 2520,
                     step: int = 5) -> pd.DataFrame:
    """Expanding-window OOS R2 versus the historical-mean forecast.

    The training window drops the last ``h`` observations so that no
    not-yet-realised return leaks into the fit.
    """
    rows = []
    for h in horizons:
        m = d[[x, f"f{h}"]].dropna()
        y, xv = m[f"f{h}"].values, m[x].values
        se_model = se_mean = 0.0
        for i in range(start, len(m), step):
            tr = slice(0, i - h)
            b = np.polyfit(xv[tr], y[tr], 1)
            j = slice(i, min(i + step, len(m)))
            se_model += ((y[j] - np.polyval(b, xv[j])) ** 2).sum()
            se_mean += ((y[j] - y[tr].mean()) ** 2).sum()
        rows.append([h, 1 - se_model / se_mean])
    return pd.DataFrame(rows, columns=["h", "R2_OS"])


def subsample_stability(d, h: int = 21, x: str = "lnVIX",
                        periods=(("1990", "1999"), ("2000", "2009"),
                                 ("2010", "2019"), ("2020", "2026"))) -> pd.DataFrame:
    rows = []
    for a, b in periods:
        s = d.loc[a:b]
        r = _ols_hac(s[f"f{h}"], s[[x]], h)
        rows.append([f"{a}-{b}", r.params[x], r.tvalues[x], r.rsquared])
    return pd.DataFrame(rows, columns=["period", "b", "t_HAC", "R2"])
