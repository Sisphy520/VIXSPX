"""Figures (headless-safe matplotlib backend)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    buckets[[f"f{h}" for h in horizons]].plot.bar(ax=ax[0])
    ax[0].set_title("Mean forward return (%) by VIX bucket")
    m = d[["VIX", "f21"]].dropna().sample(2500, random_state=0)
    ax[1].scatter(m.VIX, m.f21, s=3, alpha=.4)
    ax[1].set_title("VIX vs 21d forward return")
    ax[1].axhline(0, c="k", lw=.5)
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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for dt, row in events.iterrows():
        i = d.index.get_loc(dt)
        seg = d.SPX.iloc[max(i - before, 0):i + after + 1]
        ax.plot(np.arange(len(seg)) - min(before, i), seg.values / row.SPX * 100,
                lw=.8, alpha=.6)
    ax.axvline(0, c="k", lw=.5)
    ax.axhline(100, c="k", lw=.5)
    ax.set_title(f"SPX path around first VIX>={threshold} events (event day=100)")
    ax.set_xlabel("trading days")
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
