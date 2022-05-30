#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product

import numpy as np
from hyperopt import fmin, hp, tpe
from hyperopt.mongoexp import MongoTrials
from loguru import logger
from tqdm import tqdm

from python_inferno.hyperopt import HyperoptSpace, main_opt, mod_quniform
from python_inferno.space import generate_space

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    exp_base = "exp201"

    dryness_methods = (1, 2)
    fuel_build_up_methods = (1, 2)
    include_temperatures = (1,)  # 0 - False, 1 - True

    experiment_args = dict()

    for dryness_method, fuel_build_up_method, include_temperature in product(
        dryness_methods, fuel_build_up_methods, include_temperatures
    ):
        space_template = dict(
            overall_scale=(1, [(1e-3, 1e3)], hp.uniform),
            fapar_factor=(3, [(-50, -1)], hp.uniform),
            fapar_centre=(3, [(-0.1, 1.1)], hp.uniform),
            fapar_shape=(3, [(0.1, 20.0)], hp.uniform),
            # NOTE All weights should be in [0, 1], otherwise unintended -ve values
            # may occur!
            fapar_weight=(3, [(0.01, 1.0)], hp.uniform),
            dryness_weight=(3, [(0.01, 1.0)], hp.uniform),
            fuel_weight=(3, [(0.01, 1.0)], hp.uniform),
            # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
            average_samples=(1, [(40, 160, 60)], mod_quniform),
            # `crop_f` suppresses BA in cropland areas.
            crop_f=(1, [(0.0, 1.0)], hp.uniform),
        )
        if dryness_method == 1:
            space_template.update(
                dict(
                    dry_day_factor=(3, [(0.0, 0.2)], hp.uniform),
                    dry_day_centre=(3, [(100, 200)], hp.uniform),
                    dry_day_shape=(3, [(0.1, 20.0)], hp.uniform),
                )
            )
        elif dryness_method == 2:
            space_template.update(
                dict(
                    rain_f=(3, [(0.1, 0.6, 0.25)], mod_quniform),
                    vpd_f=(3, [(50, 200, 75)], mod_quniform),
                    dry_bal_factor=(3, [(-100, -1)], hp.uniform),
                    dry_bal_centre=(3, [(-3, 3)], hp.uniform),
                    dry_bal_shape=(3, [(0.1, 20.0)], hp.uniform),
                )
            )
        else:
            raise ValueError(f"Unknown 'dryness_method' {dryness_method}.")

        if fuel_build_up_method == 1:
            space_template.update(
                dict(
                    fuel_build_up_n_samples=(3, [(100, 1300, 400)], mod_quniform),
                    fuel_build_up_factor=(3, [(0.5, 40)], hp.uniform),
                    fuel_build_up_centre=(3, [(-1.0, 1.0)], hp.uniform),
                    fuel_build_up_shape=(3, [(0.1, 20.0)], hp.uniform),
                )
            )
        elif fuel_build_up_method == 2:
            space_template.update(
                dict(
                    litter_tc=(3, [(1e-10, 1e-9, 4.5e-10)], mod_quniform),
                    leaf_f=(3, [(1e-4, 1e-3, 4.5e-4)], mod_quniform),
                    litter_pool_factor=(3, [(0.001, 0.1)], hp.uniform),
                    litter_pool_centre=(3, [(10, 5000)], hp.uniform),
                    litter_pool_shape=(3, [(0.1, 20.0)], hp.uniform),
                )
            )
        else:
            raise ValueError(f"Unknown 'fuel_build_up_method' {fuel_build_up_method}.")

        if include_temperature == 1:
            space_template.update(
                dict(
                    temperature_factor=(3, [(0.19, 0.3)], hp.uniform),
                    temperature_centre=(3, [(280, 320)], hp.uniform),
                    temperature_shape=(3, [(0.1, 20.0)], hp.uniform),
                    temperature_weight=(3, [(0.01, 1.0)], hp.uniform),
                )
            )
        elif include_temperature == 0:
            pass
        else:
            raise ValueError(f"Unknown 'include_temperature' {include_temperature}.")

        methods_str = "_".join(
            map(str, (dryness_method, fuel_build_up_method, include_temperature))
        )

        experiment_args[methods_str] = dict(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            space=HyperoptSpace(generate_space(space_template)),
        )

    with ThreadPoolExecutor() as executor:
        futures = []
        for (i, (methods_str, experiment_arg)) in enumerate(experiment_args.items()):
            exp_space = experiment_arg["space"]
            trials = MongoTrials(
                "mongo://localhost:1234/ba/jobs",
                exp_key=f"{exp_base}_{methods_str}",
            )

            futures.append(
                executor.submit(
                    fmin,
                    fn=partial(
                        main_opt,
                        **experiment_arg,
                    ),
                    algo=tpe.suggest,
                    trials=trials,
                    rstate=np.random.default_rng(0),
                    space=exp_space.render_discrete(),
                    # NOTE: Sometimes the same parameters are sampled repeatedly.
                    max_evals=min(10000, round(2 * exp_space.n_discrete_product)),
                    # NOTE: `leave=True` causes strange duplication of rows after
                    # completion of a set of trials, making the output hard to read.
                    # With `leave=False`, the progress bar of completed trials simply
                    # disappears.
                    show_progressbar=partial(
                        tqdm, desc=methods_str, leave=False, position=i
                    ),
                    max_queue_len=10,
                    pass_expr_memo_ctrl=True,
                )
            )
    for f in futures:
        print(f.result())
