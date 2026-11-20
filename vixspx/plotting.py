"""Figures (headless-safe matplotlib backend).

Figure conventions (apply to every new plot added here):
  * English only.
  * Descriptive title that says what is plotted and over what period.
  * Both axes labelled, with units, e.g. "(%)" or "(index points)".
  * A legend that explains every series, marker and reference line.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import * # I changed it to * so that it imports all the constants we need
from prediction import vix_momentum_interaction

# I added a new plot function, plot_interaction()
# I refined the fucntions by adding doc strings


def plot_prediction(d, buckets, horizons, path=OUTPUT_DIR / "predict.png"):
    """Plot forward returns by VIX bucket and VIX against 21-day returns.

    Create a two-panel figure using the bucketed return summary and
    observations from the input data, then save it to the requested path.

    Args:
        d (pd.DataFrame): DataFrame containing "VIX" and "f21" columns.
        buckets (pd.DataFrame): Mean forward-return summary by VIX bucket,
            with a column named 'f{h}' for each requested horizon.
        horizons (Iterable[int]): Forward-return horizons represented in
            the bucketed return summary.
        path (pathlib.Path): Output file path. Defaults to
            OUTPUT_DIR / "predict.png".

    Returns:
        None: The figure is saved to `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # Left: average forward return by VIX level bucket
    plot_df = buckets[[f"f{h}" for h in horizons]].copy()
    plot_df.columns = [f"Next {h} trading day{'s' if h > 1 else ''}" for h in horizons]
    plot_df.plot.bar(ax=ax[0], rot=0)
    ax[0].set_title("Average S&P 500 return after the VIX reading,\ngrouped by VIX level")
    ax[0].set_xlabel("VIX level on the starting day (index points, annualized vol %)")
    ax[0].set_ylabel("Average cumulative return over the following period (%)")
    ax[0].legend(title="Holding period after the VIX reading", loc="upper left")
    ax[0].grid(axis="y", alpha=.3)

    # Right: one dot per day (random sample of 2,500 days)
    m = d[["VIX", "f21"]].dropna().sample(2500, random_state=0)
    ax[1].scatter(m.VIX, m.f21, s=4, alpha=.4, label="One trading day (random sample of 2,500)")
    ax[1].axhline(0, c="k", lw=.8, label="0% (no gain / no loss)")
    ax[1].set_title("VIX level vs. S&P 500 return over the next 21 trading days\n(about 1 month)")
    ax[1].set_xlabel("VIX level on the starting day (index points, annualized vol %)")
    ax[1].set_ylabel("Cumulative return over the next 21 trading days (%)")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_monte_carlo(mc, path=OUTPUT_DIR / "montecarlo.png", h_show: int = 21):
    """Left: null distribution of the ln(VIX) t-statistic. Right: null max |t| of the interaction grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hs = list(mc["horizons"])
    k = hs.index(h_show)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    n = len(mc["sim_simple"])

    t_null = mc["sim_simple"][:, k, 1]
    t_real = mc["real_simple"][k, 1]
    ax[0].hist(t_null, bins=60, color="tab:blue", alpha=.6,
               label=f"Simulated t-statistics under 'no predictability' ({n:,} paths)")
    lo, hi = np.percentile(t_null, [2.5, 97.5])
    ax[0].axvline(lo, c="k", ls=":", lw=1.2, label=f"Simulated 2.5% / 97.5% cut-offs ({lo:.2f} / {hi:.2f})")
    ax[0].axvline(hi, c="k", ls=":", lw=1.2)
    ax[0].axvline(t_real, c="tab:red", lw=2.2, label=f"Observed t-statistic ({t_real:.2f})")
    ax[0].set_title(f"Does ln(VIX) predict the next {h_show} trading days' S&P 500 return?\n"
                    "Observed t-statistic vs. simulated null distribution")
    ax[0].set_xlabel("Newey-West t-statistic of the ln(VIX) coefficient (unitless)")
    ax[0].set_ylabel("Number of simulated paths")
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 1.3)   # headroom so the legend does not hide the data
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].grid(alpha=.3)

    max_null = np.abs(mc["sim_grid"]).reshape(n, -1).max(axis=1)
    max_real = np.abs(mc["real_grid"]).max()
    n_cells = mc["real_grid"].size
    ax[1].hist(max_null, bins=60, color="tab:blue", alpha=.6,
               label=f"Simulated largest |t| across all {n_cells} tests ({n:,} paths)")
    ax[1].axvline(np.percentile(max_null, 95), c="k", ls=":", lw=1.2,
                  label=f"Simulated 95th percentile ({np.percentile(max_null, 95):.2f})")
    ax[1].axvline(max_real, c="tab:red", lw=2.2,
                  label=f"Observed largest |t| ({max_real:.2f})")
    ax[1].axvline(1.96, c="tab:green", ls="--", lw=1.2,
                  label="Conventional single-test cut-off (1.96)")
    ax[1].set_title("Momentum x VIX interaction: all windows and horizons together\n"
                    "Largest |t| among the tests vs. simulated null distribution")
    ax[1].set_xlabel("Largest absolute Newey-West t-statistic of the interaction term (unitless)")
    ax[1].set_ylabel("Number of simulated paths")
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 1.3)
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_event_paths(d, events, threshold, path=OUTPUT_DIR / "events.png",
                     before: int = 20, after: int = 126):
    """Plot S&P 500 paths around selected high-VIX events.

    Args:
        d (pd.DataFrame): DataFrame indexed by date with an "SPX" column.
        events (pd.DataFrame): Event rows indexed by dates in `d`, each
            containing the event-day "SPX" value.
        threshold (float): VIX threshold used to select events and label
            the plot.
        path (pathlib.Path): Output file path. Defaults to
            OUTPUT_DIR / "events.png".
        before (int): Number of trading days before each event to plot.
            Defaults to 20.
        after (int): Number of trading days after each event to plot.
            Defaults to 126.

    Returns:
        None: The figure is saved to `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    paths = {}
    for dt, row in events.iterrows():
        i = d.index.get_loc(dt)
        seg = d.SPX.iloc[max(i - before, 0):i + after + 1]
        rel = np.arange(len(seg)) - min(before, i)
        level = seg.values / row.SPX * 100
        ax.plot(rel, level, lw=.8, alpha=.5, color="tab:blue")
        for k, v in zip(rel, level):
            paths.setdefault(k, []).append(v)
    days = sorted(paths)
    ax.plot(days, [np.median(paths[k]) for k in days], c="tab:red", lw=2.2,
            label="Median across events")
    ax.plot([], [], c="tab:blue", lw=.8, alpha=.5,
            label=f"One line = one event ({len(events)} events)")
    ax.axvline(0, c="k", lw=.8, ls="--", label="Event day (first day VIX reaches the threshold)")
    ax.axhline(100, c="k", lw=.8, label="100 = S&P 500 close on the event day")
    ax.set_title(f"S&P 500 around the first day the VIX reaches {threshold} or more\n"
                 "(each path rescaled so the event-day close = 100)")
    ax.set_xlabel("Trading days relative to the event day (0 = event day)")
    ax.set_ylabel("S&P 500 level (event-day close = 100)")
    ax.legend(loc="lower left")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_interaction(d, horizons=HORIZONS, path=OUTPUT_DIR / "interaction.png"):
    """Plot predicted returns and VIX-momentum interaction estimates.

    Create a two-panel figure showing predicted 21-day returns across
    VIX levels at low and high momentum, and interaction coefficients
    with approximate 95% Newey-West HAC confidence intervals across
    forecasting horizons.

    Args:
        d (pd.DataFrame): DataFrame containing "VIX", "lnVIX", "SPX",
            "f21", and the forward-return columns 'f{h}' for the requested
            horizons.
        horizons (Iterable[int]): Forward-return horizons for the
            interaction-coefficient panel. Defaults to HORIZONS.
        path (pathlib.Path): Output file path. Defaults to
            OUTPUT_DIR / "interaction.png".

    Returns:
        pathlib.Path: The path where the figure was saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: Predicted 21-day returns across VIX levels and past momentum
    h = 21
    r = vix_momentum_interaction(d, h)  # Fit interaction regression
    m = d[["VIX", "lnVIX", "SPX", f"f{h}"]].copy()
    m["mom21"] = (np.log(m.SPX) - np.log(m.SPX.shift(21))) * 100  # Past 21-day return
    m = m.dropna()  # Keep observations with complete data

    # Calculate predictor means to match the regression's mean-centering
    mean_vix, mean_mom = m.lnVIX.mean(), m.mom21.mean()

    # Generate VIX values between the 5th and 95th percentiles
    vix = np.linspace(m.VIX.quantile(.05), m.VIX.quantile(.95), 100)
    vix_c = np.log(vix) - mean_vix  # Center log VIX

    # Compare predicted returns at low vs. high historical momentum
    momentum = {"Low momentum (25th pct)": m.mom21.quantile(.25),
                "High momentum (75th pct)": m.mom21.quantile(.75)}

    for label, mom in momentum.items():
        mom_c = mom - mean_mom  # Center the selected momentum level

        # Calculate predicted returns using fitted regression coefficients
        pred = (r.params["const"] + r.params["vix_c"] * vix_c
                + r.params["mom_c"] * mom_c + r.params["interaction"] * vix_c * mom_c)
        ax[0].plot(vix, pred, label=label)

    # Format the interaction plot
    ax[0].axhline(0, c="k", lw=.5)
    ax[0].set(title="VIX x momentum: 21d forward return",
              xlabel="VIX", ylabel="Predicted return (%)")
    ax[0].legend(fontsize=8)

    # Panel 2: Estimate interaction coefficients across forecasting horizons
    rows = []
    for h in horizons:
        r_h = vix_momentum_interaction(d, h)  # Fit a separate model per horizon
        rows.append([h, r_h.params["interaction"], r_h.bse["interaction"]])
    coef = pd.DataFrame(rows, columns=["h", "b", "se"])

    # Plot interaction coefficients with approximate 95% HAC confidence intervals
    ax[1].errorbar(coef.h, coef.b, yerr=1.96 * coef.se, fmt="o-", capsize=4)
    ax[1].axhline(0, c="k", lw=.5)  # Reference line for no interaction
    ax[1].set(title="Interaction coefficient (95% HAC CI)",
              xlabel="Forward horizon (trading days)", ylabel="Interaction coefficient")
    ax[1].set_xticks(list(horizons))

    # Save the figure and release plotting resources
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path
