"""Monte Carlo null test of the VIX predictive regressions and the momentum-interaction grid.

Usage:  python scripts/run_montecarlo.py [--sims 5000] [--block 1] [--ar 1] [--seed 0]
Saves outputs/montecarlo.png (block 1) or outputs/montecarlo_block{N}.png.

Note: block > 1 keeps the lead-lag dependence between return and VIX shocks inside each
block, which leaks real predictability into the "null" (see vixspx/montecarlo.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vixspx import load_data
from vixspx.config import OUTPUT_DIR
from vixspx.montecarlo import run_monte_carlo
from vixspx.plotting import plot_monte_carlo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--block", type=int, default=1,
                    help="mean bootstrap block length in days; 1 = i.i.d. pairs (strict null, default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ar", type=int, default=1, help="AR order of the ln VIX null model (default 1)")
    a = ap.parse_args()

    d = load_data()
    lp, lnv = np.log(d.SPX.values), np.log(d.VIX.values)
    mc = run_monte_carlo(lp, lnv, n_sims=a.sims, mean_block=a.block, seed=a.seed,
                         ar_order=a.ar)
    m, hs, ws = mc["model"], list(mc["horizons"]), list(mc["windows"])
    n = len(mc["sim_simple"])

    print(f"Null model: lnVIX AR(1) phi={m['phi']:.4f}, corr(return shock, VIX shock)="
          f"{np.corrcoef(m['e'], m['u'])[0, 1]:.3f}; AR order {m['ar_order']}; {n} simulated paths, mean block {a.block} days\n")

    print("=== 1. f_h ~ ln(VIX): observed vs simulated null ===")
    rows = []
    for k, h in enumerate(hs):
        b_s, t_s = mc["sim_simple"][:, k, 0], mc["sim_simple"][:, k, 1]
        b, t = mc["real_simple"][k]
        rows.append(dict(
            h=h, slope=b, t_HAC=t,
            null_mean_slope=b_s.mean(),                       # finite-sample (Stambaugh) bias
            null_t_2p5=np.percentile(t_s, 2.5), null_t_97p5=np.percentile(t_s, 97.5),
            size_at_1p96=(np.abs(t_s) > 1.96).mean(),          # should be 0.05 if HAC were exact
            p_MC=(np.abs(t_s) >= abs(t)).mean(),
            p_slope_MC=(np.abs(b_s - b_s.mean()) >= abs(b - b_s.mean())).mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 2. Momentum x ln(VIX) interaction: all windows x horizons ===")
    real, sim = mc["real_grid"], mc["sim_grid"]
    cell_p = np.array([[(np.abs(sim[:, i, j]) >= abs(real[i, j])).mean()
                        for j in range(len(hs))] for i in range(len(ws))])
    print("Observed t (rows: momentum window; cols: horizon)")
    print(pd.DataFrame(real, index=ws, columns=hs).round(2).to_string())
    print("\nSingle-test Monte Carlo p-values")
    print(pd.DataFrame(cell_p, index=ws, columns=hs).round(3).to_string())

    max_null = np.abs(sim).reshape(n, -1).max(axis=1)
    max_real = np.abs(real).max()
    cnt_null = (np.abs(sim) > 1.96).reshape(n, -1).sum(axis=1)
    cnt_real = int((np.abs(real) > 1.96).sum())
    print(f"\nLargest observed |t| = {max_real:.2f}; simulated 95th percentile of the largest |t| = "
          f"{np.percentile(max_null, 95):.2f}")
    print(f"Family-wise p-value (P[max|t| >= observed]) = {(max_null >= max_real).mean():.3f}")
    print(f"Tests with |t|>1.96: observed {cnt_real} of {real.size}; simulated mean {cnt_null.mean():.2f}; "
          f"P[count >= observed] = {(cnt_null >= cnt_real).mean():.3f}")

    name = "montecarlo.png" if a.block == 1 else f"montecarlo_block{a.block}.png"
    plot_monte_carlo(mc, OUTPUT_DIR / name)
    print("\nSaved", OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
