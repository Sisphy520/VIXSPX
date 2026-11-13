"""Figures (headless-safe matplotlib backend)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR


def plot_prediction(d, buckets, horizons, path=OUTPUT_DIR / "predict.png"):
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
