#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from itertools import islice
from operator import itemgetter
from pathlib import Path

import matplotlib as mpl
from loguru import logger

from python_inferno.configuration import land_pts
from python_inferno.delta_sa import analyse_sis
from python_inferno.delta_sa import delta_sis_calc as sis_calc
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import SAMetric
from python_inferno.spotpy_mcmc import get_cached_mcmc_chains

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, help="method index", default=0)
    args = parser.parse_args()

    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    save_dir = Path("~/tmp/delta-sa").expanduser()
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

    names, chains = itemgetter("names", "chains")(
        get_cached_mcmc_chains(method_index=method_index)
    )

    all_sis = sis_calc(
        params=params,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        N=int(1e4),  # 1e3
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
