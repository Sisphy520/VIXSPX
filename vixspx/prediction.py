"""Predictive regressions: can the VIX forecast future S&P 500 returns?
Can S&P 500 momentum moderates the relationship between VIX and future returns?
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

# I changed the title at the beginning to add the one additional question
# I removed the return values in each function definition to keep the format consistent
# I also changed the doc string to incldue brief descriptions of inputs and what it returns

# I added this new function for examining the S&P 500 momentum as a moderator
def vix_momentum_interaction(d, h=21):
    """Fit an OLS regression testing whether S&P 500 momentum moderates
    the relationship between VIX and future S&P 500 returns.

    Calculate past 21-day S&P 500 log returns as momentum, mean-center
    log VIX and momentum, and create their interaction term. Fit the
    regression using Newey-West HAC standard errors to account for
    heteroskedasticity and autocorrelation in overlapping forward returns.

    Args:
        d (pd.DataFrame): DataFrame containing 'SPX', 'lnVIX', and
            forward-return columns named 'f{h}'.
        h (int): Forward-return horizon in trading days. Defaults to 21.

    Returns:
        RegressionResultsWrapper: Fitted OLS regression results,
        including coefficient estimates, HAC standard errors,
        t-statistics, p-values, and R-squared.
    """
    df = d.copy() # I don't wanto accidentally modify the df
    
    # Past 21-day S&P 500 momentum
    df["mom21"] = (np.log(df["SPX"]) - np.log(df["SPX"].shift(21))) * 100
    # the mathematical equation: M₂₁,ₜ = 100 ln(SPXₜ / SPXₜ₋₂₁), which measures
    # cumulative S&P 500 returns over the preceding 21 trading days

    # Mean-center predictors, created two new columns for that
    df["vix_c"] = df["lnVIX"] - df["lnVIX"].mean()
    df["mom_c"] = df["mom21"] - df["mom21"].mean()
    # Reason: centering reduces the artificial correlation between the 
    # individual terms (vix_c, mom_c) and their interaction term (vix_c * mom_c)

    # Interaction term, created a new column for that
    df["interaction"] = df["vix_c"] * df["mom_c"]

    # the matrix X that includes the predictor variables and the interaction term,
    # created a new column for that
    X = df[["vix_c", "mom_c", "interaction"]] 
    # f21 represents 
    y = df[f"f{h}"] # equivalent to f21

    return _ols_hac(y, X, h)

def _ols_hac(y: pd.Series, X: pd.DataFrame, h: int):
    """Fit an OLS regression with Newey-West HAC standard errors.

    Align the dependent variable and predictors, drop rows with missing
    values, and fit the regression with a constant. Set the HAC maximum
    lag to twice the forward-return horizon to account for overlapping
    returns.

    Args:
        y (pd.Series): Dependent variable, such as a forward-return series.
        X (pd.DataFrame): DataFrame of predictor variables.
        h (int): Forward-return horizon in trading days.

    Returns:
        RegressionResultsWrapper: Fitted OLS regression results,
        including coefficient estimates, HAC standard errors,
        t-statistics, p-values, and R-squared.
    """
    m = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(m.iloc[:, 0], sm.add_constant(m.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(h, 1) * 2})


def predictive_regression(d, horizons, x: str = "lnVIX"):
    """Fit predictive regressions for each requested return horizon.

    Regress each forward-return series on a selected predictor using
    Newey-West HAC standard errors, and collect the predictor coefficient,
    t-statistic, and regression R-squared for each horizon.

    Args:
        d (pd.DataFrame): DataFrame containing the predictor column and
            forward-return columns named 'f{h}'.
        horizons (Iterable[int]): Forward-return horizons in trading days.
        x (str): Predictor column to use. Defaults to "lnVIX".

    Returns:
        pd.DataFrame: Results with columns "h", "b", "t_HAC", and "R2",
        containing the horizon, coefficient, HAC t-statistic, and
        R-squared.
    """
    rows = []
    for h in horizons:
        r = _ols_hac(d[f"f{h}"], d[[x]], h)
        rows.append([h, r.params[x], r.tvalues[x], r.rsquared])
    return pd.DataFrame(rows, columns=["h", "b", "t_HAC", "R2"])


def regression_with_realized_vol(d, horizons):
    """Compare VIX and realized volatility as predictors of future returns.

    For each requested horizon, regress the forward return on VIX and
    21-day realized volatility using Newey-West HAC standard errors.

    Args:
        d (pd.DataFrame): DataFrame containing "VIX", "rv21", and
            forward-return columns named 'f{h}'.
        horizons (Iterable[int]): Forward-return horizons in trading days.

    Returns:
        pd.DataFrame: Results with columns "h", "b_VIX", "t_VIX", "b_RV",
        "t_RV", and "R2", containing each horizon's coefficients,
        HAC t-statistics, and R-squared.
    """
    rows = []
    for h in horizons:
        r = _ols_hac(d[f"f{h}"], d[["VIX", "rv21"]], h)
        rows.append([h, r.params["VIX"], r.tvalues["VIX"],
                     r.params["rv21"], r.tvalues["rv21"], r.rsquared])
    return pd.DataFrame(rows, columns=["h", "b_VIX", "t_VIX", "b_RV", "t_RV", "R2"])


def bucket_forward_returns(d, horizons, edges=(0, 15, 20, 25, 35, 100),
                           labels=("<15", "15-20", "20-25", "25-35", ">35")):
    """Summarize forward returns by VIX level bucket.

    Divide observations into VIX intervals and calculate the number of
    observations and mean forward return for each requested horizon.

    Args:
        d (pd.DataFrame): DataFrame containing a "VIX" column and
            forward-return columns named 'f{h}'.
        horizons (Iterable[int]): Forward-return horizons in trading days.
        edges (Iterable[float]): VIX interval boundaries. Defaults to
            (0, 15, 20, 25, 35, 100).
        labels (Iterable[str]): Labels for the VIX intervals. Defaults to
            ("<15", "15-20", "20-25", "25-35", ">35").

    Returns:
        pd.DataFrame: Summary indexed by VIX bucket, with a "days" count
        column and one mean forward-return column named 'f{h}' per horizon.
    """
    b = pd.cut(d.VIX, list(edges), labels=list(labels))
    return d.groupby(b, observed=True).agg(
        days=("VIX", "size"), **{f"f{h}": (f"f{h}", "mean") for h in horizons})


def out_of_sample_r2(d, horizons, x: str = "lnVIX", start: int = 2520,
                     step: int = 5):
    """Expanding-window OOS R2 versus the historical-mean forecast.

    At each evaluation point, fit a linear predictor and historical-mean
    forecast on prior observations, excluding the most recent ``h`` rows
    to avoid using not-yet-realised forward returns. Compare their
    cumulative squared errors over the evaluation periods.

    Args:
        d (pd.DataFrame): DataFrame containing the predictor column and
            forward-return columns named 'f{h}'.
        horizons (Iterable[int]): Forward-return horizons in trading days.
        x (str): Predictor column. Defaults to "lnVIX".
        start (int): First observation index used for out-of-sample
            evaluation. Defaults to 2520.
        step (int): Number of observations per evaluation period.
            Defaults to 5.

    Returns:
        pd.DataFrame: Results with columns "h" and "R2_OS", containing
        each horizon and its out-of-sample R-squared versus the
        historical-mean forecast.
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
                                 ("2010", "2019"), ("2020", "2026"))):
    """Compare predictive-regression estimates across time periods.

    Fit a univariate OLS regression with Newey-West HAC standard errors
    for each requested period, using the selected predictor and return
    horizon.

    Args:
        d (pd.DataFrame): DataFrame indexed by date and containing the
            predictor column and forward-return column 'f{h}'.
        h (int): Forward-return horizon in trading days. Defaults to 21.
        x (str): Predictor column. Defaults to "lnVIX".
        periods (Iterable[tuple[str, str]]): Inclusive start and end
            years for each subsample. Defaults to the periods 1990-1999,
            2000-2009, 2010-2019, and 2020-2026.

    Returns:
        pd.DataFrame: Results with columns "period", "b", "t_HAC", and
        "R2", containing each period's coefficient, HAC t-statistic,
        and R-squared.
    """
    rows = []
    for a, b in periods:
        s = d.loc[a:b]
        r = _ols_hac(s[f"f{h}"], s[[x]], h)
        rows.append([f"{a}-{b}", r.params[x], r.tvalues[x], r.rsquared])
    return pd.DataFrame(rows, columns=["period", "b", "t_HAC", "R2"])
