# VIX vs. S&P 500: Can the VIX Predict Returns?

A small, reproducible research project on the relationship between the CBOE Volatility Index (VIX) and S&P 500 returns, focused on **predictive** power: does a high VIX today tell us anything about future returns, and what happens after extreme VIX spikes?

# (NEW) S&P 500 momentum as a moderator:

Does S&P 500 price momentum moderate the relationship between VIX and future S&P 500 returns?

Data: daily closes of `^GSPC` and `^VIX` from Yahoo Finance, 1990 to present (~9,300 trading days).

## Project layout

```
Finance/
├── vixspx/                 # the package
│   ├── config.py           # paths and constants (cwd-independent)
│   ├── data.py             # download_data(), load_data()
│   ├── features.py         # add_features(): log VIX, realized vol, forward returns
│   ├── prediction.py       # predictive regressions, buckets, out-of-sample R²
│   ├── events.py           # de-clustered event study, bootstrap
│   ├── montecarlo.py       # Monte Carlo null test (simulated "VIX is useless" worlds)
│   └── plotting.py         # figures
├── scripts/
│   ├── download_data.py    # fetch data -> data/data.csv
│   ├── run_prediction.py   # predictive analysis -> outputs/predict.png
│   ├── run_events.py       # extreme-VIX event study -> outputs/events.png
│   └── run_montecarlo.py   # Monte Carlo null test -> outputs/montecarlo.png
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
python scripts/run_montecarlo.py    # 5,000 simulated paths; options: --sims, --ar, --block, --seed
```

Scripts resolve paths relative to the project, so they can be run from any directory.

Use the package directly:

```python
from vixspx import load_data, add_features, predictive_regression, find_events

d = add_features(load_data(), horizons=[1, 5, 21, 63])
print(predictive_regression(d, [1, 5, 21, 63]))
events = find_events(d, threshold=30, cooldown=60)
```

## Figure conventions

All figures are written to `outputs/` and follow the same rules (see `vixspx/plotting.py`): English only, a descriptive title, both axes labelled with units, and a legend explaining every series and reference line. New plots should follow the same rules.

## Methodology

- **Forward returns:** log return (%) from the close of day *t* to the close of day *t+h*, for h = 1, 5, 21, 63 (and 126, 252 in the event study).
- **Predictive regression:** `f_h ~ ln(VIX)`, with Newey-West (HAC) standard errors, `maxlags = 2h`, because overlapping windows make residuals autocorrelated.
- **VIX vs. realized volatility:** adds 21-day realized volatility as a second regressor to see which carries the information (a proxy for the variance risk premium).
- **Out-of-sample test:** expanding window (first 10 years as training), compared with the historical-mean forecast. The last `h` observations are dropped from training to avoid look-ahead.
- **Event study:** an event is the first day VIX ≥ threshold (30 or 40). A new event requires VIX to stay below the threshold for more than 60 trading days (de-clustering). Significance uses an event-level bootstrap.
- **(New)Interaction regression:** f_h ~ ln(VIX) + Momentum_21 + ln(VIX) × Momentum_21, using mean-centered predictors and Newey–West (HAC) standard errors (maxlags = 2h) to account for autocorrelation from overlapping forward-return windows. The interaction term tests whether past 21-day S&P 500 momentum moderates the relationship between VIX and future S&P 500 returns.
- **(New)Regression Model:** 
Rₜ,ₜ₊ₕ = β₀ + β₁ ln(VIXₜ) + β₂ M₂₁,ₜ + β₃ (ln(VIXₜ) × M₂₁,ₜ) + εₜ

## Monte Carlo null test

**Question:** if the VIX (and past momentum) had *no* predictive power, how often would we still see results as strong as the ones we observed?

**Idea (coin-flip analogy):** to judge whether 58 heads in 100 flips is unusual, simulate thousands of fair-coin experiments and count how often 58 or more heads appear. Here the "fair coin" is a world in which the VIX does not predict returns.

**How the "VIX is useless" world is built** (`vixspx/montecarlo.py`):

1. Estimate from the real data: the mean daily log return `mu`; an AR(p) model for ln(VIX) (default p = 1, persistence about 0.98); and the leftover daily shocks, kept as pairs `(e_t, u_t)` for the return and the VIX.
2. Resample the shock pairs with replacement (i.i.d. pairs). Keeping the pairs together preserves the strong negative link between returns and VIX shocks (correlation about -0.72), which causes a small-sample bias in predictive regressions (Stambaugh bias).
3. Build a fake history of the same length: `return_t = mu + e_t` (no dependence on lagged VIX) and `ln VIX_t = c + phi * ln VIX_{t-1} + u_t`.
4. On each fake history, run exactly the same regressions as on the real data (same horizons, same momentum windows, same Newey-West errors).
5. Repeat 5,000 times and see where the real statistic falls in the simulated distribution (Monte Carlo p-value). For the momentum grid, record the largest absolute t-statistic across all 28 tests in each simulated history, which corrects for trying many combinations.

**Block length matters.** With a block bootstrap (`--block 21`), each block also carries the real lead-lag relationship between return and VIX shocks, so genuine predictability leaks into the "null" (null slope at h = 1 was 0.068 vs. 0.071 observed, and the test rejected 23% of the time at the nominal 5%). The default is therefore i.i.d. pairs (`--block 1`). `outputs/montecarlo_block21.png` is kept only as a diagnostic; do not use its p-values.

**Momentum grid:** 7 momentum windows (5, 10, 21, 42, 63, 126, 252 trading days) x 4 horizons (1, 5, 21, 63 days) = 28 tests of the `ln(VIX) x momentum` interaction.

**Results** (`outputs/montecarlo.png`, 5,000 paths):

| Horizon h | Observed t (ln VIX) | Simulated 95% range under "no predictability" | Monte Carlo p |
|---|---|---|---|
| 1 day | 1.30 | [-1.86, 2.03] | 0.20 |
| 5 days | 1.21 | [-2.02, 2.24] | 0.27 |
| 21 days | 1.26 | [-2.03, 2.30] | 0.26 |
| 63 days | 1.01 | [-2.02, 2.43] | 0.38 |

- The observed t-statistics sit well inside the simulated null range, so there is no evidence that the VIX predicts returns.
- Newey-West tests are somewhat over-sized at longer horizons in these simulations (nominal 5%, actual about 7 to 9% at h = 5, 21, 63; 4.9% at h = 1).
- Momentum interaction: only 2 of 28 tests have |t| > 1.96 (both with 5-day momentum), about what noise alone produces (2.1 on average). The largest observed |t| is 2.14 vs. a simulated 95th percentile of 3.28; family-wise p = 0.41. For the 21-day momentum and 21-day horizon cell, t = -0.23 and Monte Carlo p = 0.84.
- Sensitivity: replacing the AR(1) model for ln(VIX) with AR(2), AR(3) or AR(5) changes the p-values by less than 0.02. AIC/BIC prefer higher orders (the real VIX has long memory), but the conclusions do not change.

## Main findings

| Question | Result |
|---|---|
| Does `ln(VIX)` predict future returns linearly? | Coefficients are positive but insignificant (HAC t ≈ 1.0 to 1.3, R² < 1%). |
| Does it hold out of sample? | No. OOS R² is negative at every horizon (−2.5% at h = 63), so the historical mean is a better forecast. |
| Is the relationship stable over time? | No. Positive in the 1990s, 2010s, 2020s; about zero in 2000 to 2009. |
| What about extreme levels? | Returns are higher after VIX > 25 to 35, but this relies on few, clustered crisis periods. |
| Buy on the first VIX ≥ 30? | No clear edge. 22 events; mean 63-day return +2.3% vs +2.1% unconditional; bootstrap 95% CI [-2.5, 6.5] contains the unconditional mean. Average 63-day max drawdown after the event is −7.4%. |
| Buy on the first VIX ≥ 40? | Better (63-day mean +6.9%, 90% positive), but only 10 events. |
| Does a Monte Carlo null test change the picture? | No. Monte Carlo p-values are 0.20 to 0.38 across horizons; the "significant" momentum interactions are what noise produces (family-wise p = 0.41). |
| Why do daily and event results differ? | Daily samples are dominated by mid/late-crisis high-VIX days, after much of the fall has happened. The first breach is often still early in the decline. |

Takeaway: there is a weak "high VIX is followed by higher returns" effect, but it appears late in stress episodes rather than at the first spike, is not stable, and is not statistically reliable out of sample.

## Limitations

- Few independent extreme-VIX episodes; confidence intervals are wide.
- The 60-day cooldown and the 30/40 thresholds are choices, and no sensitivity analysis has been done.
- No transaction costs, and no tradable strategy. The VIX itself cannot be traded directly (futures/ETNs have roll costs).
- SPX closes at 16:00 ET while the VIX closes at 16:15 ET, a small timing mismatch in daily data.
- Yahoo Finance data is not guaranteed to be revision-free.
- Monte Carlo: i.i.d. resampling of shock pairs drops volatility clustering, and ln(VIX) is modelled as a low-order AR process while the real VIX has longer memory (conclusions were unchanged for AR(1), (2), (3), (5)).
- Monte Carlo: only the null "no predictability" is tested. No power analysis has been done, so "not significant" does not prove "no effect".
- The extreme-VIX event study has not been re-tested with Monte Carlo (it still uses an event-level bootstrap).

## Possible extensions

- Enter after the VIX falls a given percentage from its peak, rather than at the first breach.
- Contemporaneous and asymmetric (leverage effect) analysis of ΔVIX vs returns.
- Regime-switching models and rolling-window betas.
- Tradable backtests with costs.
- Power analysis: add a small true predictability to the simulated histories and measure how often the tests detect it.
- Placebo event study: compare the post-event returns with randomly chosen dates (same spacing rules).

This project is for research and education only and is not investment advice.
