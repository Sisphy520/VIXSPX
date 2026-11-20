"""Monte Carlo (bootstrap) null distributions for the predictive regressions.

Question answered: if the VIX (and past momentum) truly had NO predictive power,
how often would we see test statistics as large as the ones observed?

Null model (Stambaugh / Mark-style residual bootstrap)
------------------------------------------------------
    r_t     = mu + e_t                         (returns: no dependence on lagged VIX)
    lnVIX_t = c + phi * lnVIX_{t-1} + u_t      (VIX: persistent AR(1))

The residual pairs (e_t, u_t) are resampled together, so the simulated data keep
  * the strong negative contemporaneous correlation between returns and VIX shocks
    (about -0.72, the source of the Stambaugh small-sample bias),
  * the high persistence of the VIX (phi estimated from the data),
while the link "VIX today -> returns tomorrow" is set to zero by construction.

Block length: the default is mean_block=1, i.e. i.i.d. resampling of the pairs, which is
the strict null. With mean_block > 1 (stationary bootstrap) each block also carries the
real lead-lag dependence between return and VIX shocks, which leaks genuine predictability
into the "null": in a test with 21-day blocks the null slope at h=1 was 0.068, almost
equal to the observed 0.071, and the HAC test rejected 23% of the time at the nominal 5%.
Larger blocks are therefore kept only as a diagnostic.

On every simulated path we re-run exactly the same regressions as on the real
data (same horizons, same momentum windows, same Newey-West errors).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import lfilter, lfiltic

HORIZONS = (1, 5, 21, 63)
WINDOWS = (5, 10, 21, 42, 63, 126, 252)


# --------------------------------------------------------------------------
# Fast OLS with Newey-West (Bartlett) standard errors, maxlags = 2h
# --------------------------------------------------------------------------
def _ols_nw(X, y, L):
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    s = X * (y - X @ b)[:, None]
    S = s.T @ s
    for lag in range(1, L + 1):
        g = s[lag:].T @ s[:-lag]
        S += (1 - lag / (L + 1)) * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ S @ xtx_inv))
    return b, b / se


def simple_stats(lp, lnv, horizons=HORIZONS):
    """f_h ~ const + lnVIX. Returns array (len(horizons), 2) of [slope, HAC t]."""
    out = np.empty((len(horizons), 2))
    for i, h in enumerate(horizons):
        y = (lp[h:] - lp[:-h]) * 100
        X = np.column_stack([np.ones(len(y)), lnv[:-h]])
        b, t = _ols_nw(X, y, 2 * h)
        out[i] = b[1], t[1]
    return out


def grid_stats(lp, lnv, windows=WINDOWS, horizons=HORIZONS):
    """Momentum-interaction regression for every (window, horizon).

    f_h ~ const + lnVIX_c + mom_w_c + lnVIX_c * mom_w_c.
    Returns the HAC t-statistic of the interaction term, shape (len(windows), len(horizons)).
    (The interaction t-statistic is invariant to how the variables are centred.)
    """
    n = len(lp)
    out = np.empty((len(windows), len(horizons)))
    for i, w in enumerate(windows):
        mom = np.full(n, np.nan)
        mom[w:] = (lp[w:] - lp[:-w]) * 100
        for j, h in enumerate(horizons):
            t0, t1 = w, n - h                      # rows with both momentum and f_h
            y = (lp[t0 + h:t1 + h] - lp[t0:t1]) * 100
            x, m = lnv[t0:t1], mom[t0:t1]
            xc, mc = x - x.mean(), m - m.mean()
            X = np.column_stack([np.ones(len(y)), xc, mc, xc * mc])
            _, t = _ols_nw(X, y, 2 * h)
            out[i, j] = t[3]
    return out


# --------------------------------------------------------------------------
# Null model and simulation
# --------------------------------------------------------------------------
def fit_null_model(lp, lnv, ar_order=1):
    """Estimate mu, the AR(ar_order) model for ln VIX and the paired residuals (e, u).

    ``phi`` is the sum of the AR coefficients (overall persistence); ``phis`` holds them all.
    """
    p = ar_order
    r = np.diff(lp)
    mu = r.mean()
    e = (r - mu)[p - 1:]                                   # return shocks for t = p..N-1
    y = lnv[p:]
    X = np.column_stack([np.ones(len(y))] + [lnv[p - i:len(lnv) - i] for i in range(1, p + 1)])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phis = coef[0], coef[1:]
    u = y - X @ coef
    return dict(mu=mu, c=c, phi=phis.sum(), phis=phis, ar_order=p, e=e, u=u,
                lnv_init=lnv[:p].copy(), lp_init=lp[:p].copy(), n=len(lnv))


def stationary_bootstrap_idx(n, mean_block, rng):
    """Politis-Romano stationary bootstrap indices (vectorised)."""
    jump = rng.random(n) < 1.0 / mean_block
    jump[0] = True
    seg = np.cumsum(jump) - 1
    starts = rng.integers(0, n, size=int(seg[-1]) + 1)
    pos = np.arange(n)
    first = pos[jump]
    return (starts[seg] + (pos - first[seg])) % n


def simulate_path(model, mean_block, rng):
    """One simulated (log price, log VIX) path under the null of no predictability."""
    idx = stationary_bootstrap_idx(len(model["e"]), mean_block, rng)
    e, u = model["e"][idx], model["u"][idx]
    p = model["ar_order"]
    a = np.concatenate([[1.0], -model["phis"]])
    zi = lfiltic([1.0], a, model["lnv_init"][::-1])      # most recent value first
    lnv, _ = lfilter([1.0], a, model["c"] + u, zi=zi)
    lp = model["lp_init"][-1] + np.cumsum(model["mu"] + e)
    return (np.concatenate([model["lp_init"], lp]),
            np.concatenate([model["lnv_init"], lnv]))


def _worker(args):
    model, mean_block, seed, n_sims, horizons, windows = args
    rng = np.random.default_rng(seed)
    simple = np.empty((n_sims, len(horizons), 2))
    grid = np.empty((n_sims, len(windows), len(horizons)))
    for k in range(n_sims):
        lp, lnv = simulate_path(model, mean_block, rng)
        simple[k] = simple_stats(lp, lnv, horizons)
        grid[k] = grid_stats(lp, lnv, windows, horizons)
    return simple, grid


def run_monte_carlo(lp, lnv, n_sims=2000, mean_block=1, seed=0,
                    horizons=HORIZONS, windows=WINDOWS, n_jobs=8, ar_order=1):
    """Run the null simulation. Returns dict with real statistics and simulated ones."""
    model = fit_null_model(lp, lnv, ar_order)
    real_simple = simple_stats(lp, lnv, horizons)
    real_grid = grid_stats(lp, lnv, windows, horizons)

    n_jobs = max(1, min(n_jobs, n_sims))
    counts = [n_sims // n_jobs + (1 if i < n_sims % n_jobs else 0) for i in range(n_jobs)]
    seeds = np.random.SeedSequence(seed).spawn(n_jobs)
    jobs = [(model, mean_block, s, c, horizons, windows) for s, c in zip(seeds, counts)]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_worker, jobs))
    return dict(
        model=model, horizons=horizons, windows=windows, mean_block=mean_block,
        real_simple=real_simple, real_grid=real_grid,
        sim_simple=np.concatenate([p[0] for p in parts]),
        sim_grid=np.concatenate([p[1] for p in parts]),
    )
