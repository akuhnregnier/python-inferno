#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product

import numpy as np
from hyperopt import fmin, tpe
from hyperopt.mongoexp import MongoTrials
from loguru import logger
from tqdm import tqdm

from python_inferno.hyperopt import HyperoptSpace, get_space_template, main_opt
from python_inferno.space import generate_space_spec

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    exp_base = "newrun"

    dryness_methods = (1, 2)
    fuel_build_up_methods = (1, 2)
    include_temperatures = (1,)  # 0 - False, 1 - True

    experiment_args = dict()

    for dryness_method, fuel_build_up_method, include_temperature in product(
        dryness_methods, fuel_build_up_methods, include_temperatures
    ):
        methods_str = "_".join(
            map(str, (dryness_method, fuel_build_up_method, include_temperature))
        )

        experiment_args[methods_str] = dict(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            space=HyperoptSpace(
                generate_space_spec(
                    get_space_template(
                        dryness_method=dryness_method,
                        fuel_build_up_method=fuel_build_up_method,
                        include_temperature=include_temperature,
                    )
                )
            ),
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
