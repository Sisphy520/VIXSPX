# VIX vs. S&P 500: Can the VIX Predict Returns?

A small, reproducible research project on the relationship between the CBOE Volatility Index (VIX) and S&P 500 returns, focused on predictive power: does a high VIX today tell us anything about future returns, and what happens after extreme VIX spikes?

A second question: does S&P 500 price momentum moderate the relationship between VIX and future returns (a VIX × momentum interaction regression)?

Data: daily closes of `^GSPC` and `^VIX` from Yahoo Finance, 1990 to present (~9,300 trading days).

## Project layout

```
VIXSPX/
├── vixspx/                 # the package
│   ├── config.py           # paths and constants (cwd-independent)
│   ├── data.py             # download_data(), load_data()
│   ├── features.py         # add_features(): log VIX, realized vol, forward returns
│   ├── prediction.py       # predictive regressions, buckets, out-of-sample R²,
│   │                       #   vix_momentum_interaction()
│   ├── events.py           # de-clustered event study, bootstrap
│   └── plotting.py         # figures, incl. plot_interaction()
├── scripts/
│   ├── download_data.py    # fetch data -> data/data.csv
│   ├── run_prediction.py   # predictive analysis -> outputs/predict.png
│   └── run_events.py       # extreme-VIX event study -> outputs/events.png
├── notebook/
│   └── run_regression.ipynb  # VIX x momentum interaction -> outputs/interaction.png
├── data/data.csv           # saved SPX / VIX closes
├── outputs/                # generated figures
└── requirements.txt
```

## Setup and usage

```bash
pip install -r requirements.txt
python scripts/download_data.py     # optional: refresh data (a CSV is already included)
python scripts/run_prediction.py
python scripts/run_events.py
jupyter notebook notebook/run_regression.ipynb   # interaction regression and figure
```

Scripts resolve paths relative to the project, so they can be run from any directory.

Use the package directly:

```python
from vixspx import load_data, add_features, predictive_regression, find_events

d = add_features(load_data(), horizons=[1, 5, 21, 63])
print(predictive_regression(d, [1, 5, 21, 63]))
events = find_events(d, threshold=30, cooldown=60)

# VIX x momentum interaction (not re-exported at package level)
from vixspx.prediction import vix_momentum_interaction
from vixspx.plotting import plot_interaction
print(vix_momentum_interaction(d, h=21).summary())
plot_interaction(d, [1, 5, 21, 63])
```

The notebook adds the project folder to `sys.path` (it assumes `~/Desktop/PSY 46050 Project/VIXSPX`), so edit `INPUTDIR` in its first cell if the project lives elsewhere.

## Methodology

- **Forward returns:** log return (%) from the close of day *t* to the close of day *t+h*, for h = 1, 5, 21, 63 (and 126, 252 in the event study).
- **Predictive regression:** `f_h ~ ln(VIX)`, with Newey-West (HAC) standard errors, `maxlags = 2h`, because overlapping windows make residuals autocorrelated.
- **VIX vs. realized volatility:** adds 21-day realized volatility as a second regressor to see which carries the information (a proxy for the variance risk premium).
- **Out-of-sample test:** expanding window (first 10 years as training), compared with the historical-mean forecast. The last `h` observations are dropped from training to avoid look-ahead.
- **Event study:** an event is the first day VIX ≥ threshold (30 or 40). A new event requires VIX to stay below the threshold for more than 60 trading days (de-clustering). Significance uses an event-level bootstrap.
- **Interaction regression:** `f_h ~ ln(VIX) + M21 + ln(VIX) × M21`, where `M21 = 100 · ln(SPX_t / SPX_{t-21})` is past 21-day momentum. Predictors are mean-centered, and standard errors are Newey-West (HAC) with `maxlags = 2h`. The interaction term tests whether momentum moderates the VIX-return relationship.

  R(t,t+h) = β₀ + β₁ ln(VIX_t) + β₂ M21_t + β₃ (ln(VIX_t) × M21_t) + ε_t

- **Interaction figure:** the left panel shows predicted 21-day returns across VIX levels (5th to 95th percentile) at low (25th pct) and high (75th pct) momentum. The right panel shows the interaction coefficient with a 95% HAC CI for each horizon.

## Main findings

| Question | Result |
|---|---|
| Does `ln(VIX)` predict future returns linearly? | Coefficients are positive but insignificant (HAC t ≈ 1.0 to 1.3, R² < 1%). |
| Does it hold out of sample? | No. OOS R² is negative at every horizon (−2.5% at h = 63), so the historical mean is a better forecast. |
| Is the relationship stable over time? | No. Positive in the 1990s, 2010s, 2020s; about zero in 2000 to 2009. |
| What about extreme levels? | Returns are higher after VIX > 25 to 35, but this relies on few, clustered crisis periods. |
| Buy on the first VIX ≥ 30? | No clear edge. 22 events; mean 63-day return +2.3% vs +2.1% unconditional; bootstrap 95% CI [-2.5, 6.5] contains the unconditional mean. Average 63-day max drawdown after the event is −7.4%. |
| Buy on the first VIX ≥ 40? | Better (63-day mean +6.9%, 90% positive), but only 10 events. |
| Does momentum moderate the VIX-return link? | No. At h = 21 the interaction is near zero and insignificant (β = −0.029, p = .815); the model R² is about 0.6%. The low- and high-momentum lines do not cross. |
| Why do daily and event results differ? | Daily samples are dominated by mid/late-crisis high-VIX days, after much of the fall has happened. The first breach is often still early in the decline. |

Takeaway: there is a weak "high VIX is followed by higher returns" effect, but it appears late in stress episodes rather than at the first spike, is not stable, and is not statistically reliable out of sample.

## Limitations

- Few independent extreme-VIX episodes; confidence intervals are wide.
- The 60-day cooldown and the 30/40 thresholds are choices, and no sensitivity analysis has been done.
- No transaction costs, and no tradable strategy. The VIX itself cannot be traded directly (futures/ETNs have roll costs).
- SPX closes at 16:00 ET while the VIX closes at 16:15 ET, a small timing mismatch in daily data.
- Yahoo Finance data is not guaranteed to be revision-free.

## Possible extensions

- Enter after the VIX falls a given percentage from its peak, rather than at the first breach.
- Contemporaneous and asymmetric (leverage effect) analysis of ΔVIX vs returns.
- Regime-switching models and rolling-window betas.
- Tradable backtests with costs.

This project is for research and education only and is not investment advice.
