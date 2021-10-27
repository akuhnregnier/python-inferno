#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path
from pprint import pformat

import hyperopt
import numpy as np
from hyperopt import fmin, hp, tpe
from hyperopt.mongoexp import MongoTrials
from loguru import logger
from scipy.optimize import basinhopping

from python_inferno.ba_model import gen_to_optimise
from python_inferno.basinhopping import BoundedSteps, Recorder
from python_inferno.hyperopt import HyperoptSpace
from python_inferno.space import generate_space


def fail_func(*args, **kwargs):
    return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}


def success_func(loss, *args, **kwargs):
    return {"loss": loss, "status": hyperopt.STATUS_OK}


to_optimise = gen_to_optimise(
    fail_func=fail_func,
    success_func=success_func,
)


def main(
    discrete_params,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
):
    logger.info(f"Discrete parameters:\n{pformat(discrete_params)}")

    def to_optimise_with_discrete(x):
        opt_kwargs = {
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **discrete_params,
        }
        return to_optimise(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            **opt_kwargs,
        )

    recorder = Recorder(record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record")

    def basinhopping_callback(x, f, accept):
        values = space.inv_map_float_to_0_1(
            {**dict(zip(space.continuous_param_names, x)), **discrete_params}
        )
        values["dryness_method"] = dryness_method
        values["fuel_build_up_method"] = fuel_build_up_method
        values["include_temperature"] = include_temperature

        logger.info(f"Minimum found | loss: {f:0.6f}")

        for name, val in values.items():
            logger.info(f" - {name}: {val}")

        if recorder is not None:
            recorder.record(values, f)

            # Update record in file.
            recorder.dump()

    return (
        basinhopping(
            to_optimise_with_discrete,
            x0=space.continuous_x0_mid,
            disp=True,
            minimizer_kwargs=dict(
                method="L-BFGS-B",
                jac=None,
                bounds=[(0, 1)] * len(space.continuous_param_names),
                options=dict(maxiter=50, ftol=1e-5, eps=1e-3),
            ),
            seed=0,
            niter_success=5,
            callback=basinhopping_callback,
            take_step=BoundedSteps(stepsize=0.5, rng=np.random.default_rng(0)),
        ),
        space,
        discrete_params,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    exp_base = "exp100"

    dryness_methods = (1, 2)
    fuel_build_up_methods = (1, 2)
    include_temperatures = (1,)  # 0 - False, 1 - True

    # NOTE: hp.quniform (and qloguniform) will not respect 'start', instead only
    # returning i*step (even true for qloguniform, where values are still linear
    # spaced, but not all equally likely as in quniform).

    experiment_args = dict()

    for dryness_method, fuel_build_up_method, include_temperature in product(
        dryness_methods, fuel_build_up_methods, include_temperatures
    ):
        space_template = dict(
            fapar_factor=(3, [(-50, -1)], hp.uniform),
            fapar_centre=(3, [(-0.1, 1.1)], hp.uniform),
            fapar_shape=(3, [(0.1, 20.0)], hp.uniform),
            # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
            average_samples=(1, [(40, 161, 60)], hp.quniform),
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
                    rain_f=(3, [(0.1, 0.61, 0.25)], hp.quniform),
                    vpd_f=(3, [(50, 201, 75)], hp.quniform),
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
                    fuel_build_up_n_samples=(3, [(100, 1301, 400)], hp.quniform),
                    fuel_build_up_factor=(3, [(0.5, 40)], hp.uniform),
                    fuel_build_up_centre=(3, [(-1.0, 1.0)], hp.uniform),
                    fuel_build_up_shape=(3, [(0.1, 20.0)], hp.uniform),
                )
            )
        elif fuel_build_up_method == 2:
            space_template.update(
                dict(
                    litter_tc=(3, [(1e-10, 1.1e-9, 4.5e-10)], hp.quniform),
                    leaf_f=(3, [(1e-4, 1.1e-3, 4.5e-4)], hp.quniform),
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
                )
            )
        elif include_temperature == 0:
            pass
        else:
            raise ValueError(f"Unknown 'include_temperature' {include_temperature}.")

        space = HyperoptSpace(generate_space(space_template))

        methods_str = "_".join(
            map(str, (dryness_method, fuel_build_up_method, include_temperature))
        )

        experiment_args[methods_str] = dict(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            space=space,
        )

    with ThreadPoolExecutor() as executor:
        futures = []
        for methods_str, experiment_arg in experiment_args.items():
            trials = MongoTrials(
                "mongo://maritimus.webredirect.org:1234/ba/jobs",
                exp_key=f"{exp_base}_{methods_str}",
            )

            futures.append(
                executor.submit(
                    fmin,
                    fn=partial(
                        main,
                        space=space,
                        dryness_method=dryness_method,
                        fuel_build_up_method=fuel_build_up_method,
                        include_temperature=include_temperature,
                    ),
                    algo=tpe.suggest,
                    trials=trials,
                    rstate=np.random.RandomState(0),
                    space=space.render_discrete(),
                    max_evals=10000,
                    verbose=False,
                    max_queue_len=100,
                )
            )

        for methods_str, f in zip(experiment_args, futures):
            print(f"{methods_str}\n{f.result()}")
