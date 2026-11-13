"""Event study around extreme VIX readings. Prints tables, saves outputs/events.png."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from vixspx import (load_data, add_features, find_events, summarize_events,
                    bootstrap_event_mean)
from vixspx.config import EVENT_HORIZONS
from vixspx.events import add_drawdown
from vixspx.plotting import plot_event_paths

THRESHOLDS = (30, 40)
COOLDOWN = 60


def main():
    d = add_drawdown(add_features(load_data(), EVENT_HORIZONS), 63)
    fcols = [f"f{h}" for h in EVENT_HORIZONS]
    base = d[fcols].mean()
    print("unconditional mean (%):\n", base.round(2).to_string(), "\n")

    for th in THRESHOLDS:
        e = find_events(d, th, COOLDOWN)
        print(f"=== first VIX >= {th} (cooldown {COOLDOWN}d): {len(e)} events ===")
        t = e[["VIX", "SPX"] + fcols + ["mdd63"]].copy()
        t.index = t.index.date
        print(t.round(1).to_string())
        print(summarize_events(e, base, EVENT_HORIZONS).round(2).to_string())
        print("63d max drawdown after event, mean/median: %.1f / %.1f\n"
              % (e.mdd63.mean(), e.mdd63.median()))

    print("=== daily basis (clustered, descriptive only) ===")
    for th in (30, 35, 40):
        x = d[d.VIX >= th]
        print(f"VIX>={th}: {len(x)} days, 21d {x.f21.mean():.2f}%, "
              f"63d {x.f63.mean():.2f}%, 252d {x.f252.mean():.2f}%")

    print("\n=== robustness: daily VIX>=30 excluding 2008-09..2009-06 and 2020-02..2020-06 ===")
    crisis = (((d.index >= "2008-09-01") & (d.index <= "2009-06-30")) |
              ((d.index >= "2020-02-01") & (d.index <= "2020-06-30")))
    ex = d[(d.VIX >= 30) & ~crisis]
    print(f"{len(ex)} days:", ex[["f21", "f63", "f126", "f252"]].mean().round(2).to_dict())

    e30 = find_events(d, 30, COOLDOWN)
    for h in (21, 63, 252):
        m, lo, hi, n = bootstrap_event_mean(e30, h)
        print(f"event bootstrap h={h}: mean {m:.2f}%  95% CI [{lo:.2f}, {hi:.2f}]  "
              f"uncond {base[f'f{h}']:.2f}%  n={n}")

    plot_event_paths(d, e30, 30)


if __name__ == "__main__":
    main()
