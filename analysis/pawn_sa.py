#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger

from python_inferno.configuration import land_pts
from python_inferno.model_params import get_model_params
from python_inferno.pawn_sa import analyse_sis
from python_inferno.pawn_sa import pawn_sis_calc as sis_calc
from python_inferno.sensitivity_analysis import SAMetric
from python_inferno.spotpy_mcmc import spotpy_dream

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, help="method index", default=0)
    args = parser.parse_args()

    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    save_dir = Path("~/tmp/pawn-sa").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )

    (
        method_index,
        (
            dryness_method,
            fuel_build_up_method,
            df_sel,
            min_index,
            min_loss,
            params,
            exp_name,
            exp_key,
        ),
    ) = next(islice(enumerate(method_iter()), args.n, args.n + 1))
    assert int(params["include_temperature"]) == 1

    mcmc_kwargs = dict(
        iter_opt_index=method_index,
        # 1e5 - 15 mins with beta=0.05
        # 2e5 - 50 mins with beta=0.05 - due to decreasing acceptance rate over time!
        N=int(2e5),
        beta=0.05,
    )
    assert spotpy_dream.check_in_store(**mcmc_kwargs), str(mcmc_kwargs)
    dream_results = spotpy_dream(**mcmc_kwargs)
    results_df = dream_results["results_df"]
    space = dream_results["space"]

    # Analysis of results.
    names = space.continuous_param_names

    # Generate array of chain values, transform back to original ranges.
    chains = np.hstack(
        [
            space.inv_map_float_to_0_1({name: np.asarray(results_df[f"par{name}"])})[
                name
            ].reshape(-1, 1)
            for name in names
        ]
    )

    all_sis = sis_calc(
        params=params,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        N=int(1e4),  # 1e3, 1e4
        chain_data=chains,
        chain_names=names,
        land_points=list(range(land_pts)),
        verbose=1,
    )

    valid_sis = {land_index: sis for land_index, sis in all_sis.items() if sis}

    for metric in SAMetric:
        metric_sis = {
            land_i: sis[metric] for land_i, sis in valid_sis.items() if metric in sis
        }
        sis_save_dir = save_dir / exp_key / metric.name
        sis_save_dir.mkdir(parents=True, exist_ok=True)
        analyse_sis(
            sis=metric_sis,
            save_dir=sis_save_dir,
            exp_name=exp_name,
            exp_key=exp_key,
            metric_name=metric.name,
        )
