#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from loguru import logger

from python_inferno.mcmc import plot_pairwise_grid
from python_inferno.plotting import use_style
from python_inferno.spotpy_mcmc import plot_spotpy_results_df, spotpy_dream

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, help="method index", default=3)
    parser.add_argument(
        "--no-chains-plot", action="store_true", help="do not plot chains"
    )
    parser.add_argument(
        "--pair-plot", action="store_true", help="plot pairwise correlations"
    )
    args = parser.parse_args()
    method_index = args.n

    use_style()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    root_dir = Path("~/tmp/sinferno-mcmc").expanduser()
    root_dir.mkdir(parents=False, exist_ok=True)

    param_prod = list()

    beta = 1.0
    nChains = 50
    c = 0.1

    save_dir = root_dir / f"{method_index}"
    save_dir.mkdir(parents=False, exist_ok=True)

    mcmc_kwargs = dict(
        iter_opt_index=method_index,
        N=int(2e5),
        beta=beta,
        nChains=nChains,
        c=c,
        maxTime=5 * 60 * 60,
        acc_eps_delta=0.002,
    )

    dream_results = spotpy_dream(**mcmc_kwargs)
    results_df = dream_results["results_df"]
    space = dream_results["space"]

    if not args.no_chains_plot:
        chain_save_dir = save_dir / f"chains_{method_index}"
        chain_save_dir.mkdir(exist_ok=True)
        plot_spotpy_results_df(results_df=results_df, save_dir=chain_save_dir)

    # Analysis of results.
    names = space.continuous_param_names

    # Generate array of chain values.
    chains = np.hstack(
        [np.asarray(results_df[f"par{name}"]).reshape(-1, 1) for name in names]
    )

    if args.pair_plot:
        plot_pairwise_grid(chains=chains, names=names, save_dir=save_dir)
