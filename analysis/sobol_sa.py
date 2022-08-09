#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path

from loguru import logger

from python_inferno.configuration import land_pts
from python_inferno.hyperopt import get_space_template
from python_inferno.iter_opt import ALWAYS_OPTIMISED, IGNORED
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import SAMetric
from python_inferno.sobol_sa import analyse_sis
from python_inferno.sobol_sa import sobol_sis_calc as sis_calc

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, help="method index", default=0)
    parser.add_argument(
        "-b", "--batches", type=int, help="number of batches", default=1
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    save_dir = Path("~/tmp/sa").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )

    (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) = next(islice(method_iter(), args.n, args.n + 1))
    assert int(params["include_temperature"]) == 1

    logger.info(exp_name)

    space_template = get_space_template(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=int(params["include_temperature"]),
    )

    param_names = [
        key for key in space_template if key not in ALWAYS_OPTIMISED.union(IGNORED)
    ]
    if "crop_f" in param_names:
        param_names.remove("crop_f")

    for (data_variables, analysis_type) in [
        (param_names, "Parameters"),
        (None, "Data"),
    ]:
        all_sis = sis_calc(
            n_batches=args.batches,
            land_points=list(range(land_pts)),
            exponent=12,
            params=params,
            data_variables=data_variables,
            # NOTE Derive parameter uncertainty ranges from the existing set of runs.
            fuel_build_up_method=fuel_build_up_method,
            dryness_method=dryness_method,
        )

        valid_sis = {land_index: sis for land_index, sis in all_sis.items() if sis}

        logger.info(f"Analysing {analysis_type}: {len(all_sis)}, {len(valid_sis)}.")

        for metric in SAMetric:
            metric_sis = {
                land_i: sis[metric]
                for land_i, sis in valid_sis.items()
                if metric in sis
            }

            logger.info(f"Metric {metric.name}: {len(metric_sis)}.")

            sis_save_dir = save_dir / exp_key / analysis_type.lower() / metric.name
            sis_save_dir.mkdir(parents=True, exist_ok=True)
            analyse_sis(
                sis=metric_sis,
                save_dir=sis_save_dir,
                exp_name=exp_name,
                exp_key=exp_key,
                metric_name=metric.name,
                analysis_type=analysis_type,
            )