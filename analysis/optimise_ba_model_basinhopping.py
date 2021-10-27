#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import basinhopping
from tqdm import tqdm

from python_inferno.ba_model import gen_to_optimise
from python_inferno.basinhopping import (
    ArgType,
    BasinHoppingSpace,
    BoundedSteps,
    Recorder,
)
from python_inferno.cx1 import get_parsers, run
from python_inferno.space import generate_space
from python_inferno.utils import memoize


def fail_func(*args, **kwargs):
    return 10000.0


def success_func(loss, *args, **kwargs):
    return loss


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
    *args,
    pre_calculate=False,
    **kwargs,
):
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

    if pre_calculate:
        # Only call the function once to pre-calculate cached results.
        return to_optimise_with_discrete(space.continuous_x0_mid)

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
            niter_success=10,
            callback=basinhopping_callback,
            take_step=BoundedSteps(stepsize=0.5, rng=np.random.default_rng(0)),
        ),
        space,
        discrete_params,
    )


def mod_get_parsers():
    parser_dict = get_parsers()
    parser_dict["parser"].add_argument(
        "--pre-calculate", action="store_true", help="pre-calculate cached results"
    )
    parser_dict["parser"].add_argument(
        "--use-pre-calculate",
        action="store_true",
        help="actually use pre-calculated cached results (e.g. different CX1 specs)",
    )
    return parser_dict


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    dryness_methods = (1, 2)
    fuel_build_up_methods = (1, 2)
    include_temperatures = (1,)  # 0 - False, 1 - True

    args = []

    for dryness_method, fuel_build_up_method, include_temperature in product(
        dryness_methods, fuel_build_up_methods, include_temperatures
    ):
        space_template = dict(
            fapar_factor=(3, [(-50, -1)], ArgType.FLOAT),
            fapar_centre=(3, [(-0.1, 1.1)], ArgType.FLOAT),
            fapar_shape=(3, [(0.1, 20.0)], ArgType.FLOAT),
            # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
            average_samples=(1, [(*range(40, 161, 60),)], ArgType.CHOICE),
            # `crop_f` suppresses BA in cropland areas.
            crop_f=(1, [(0.0, 1.0)], ArgType.FLOAT),
        )
        if dryness_method == 1:
            space_template.update(
                dict(
                    dry_day_factor=(3, [(0.0, 0.2)], ArgType.FLOAT),
                    dry_day_centre=(3, [(100, 200)], ArgType.FLOAT),
                    dry_day_shape=(3, [(0.1, 20.0)], ArgType.FLOAT),
                )
            )
        elif dryness_method == 2:
            space_template.update(
                dict(
                    rain_f=(3, [np.linspace(0.1, 0.6, 3)], ArgType.CHOICE),
                    vpd_f=(3, [np.linspace(50, 200, 3)], ArgType.CHOICE),
                    dry_bal_factor=(3, [(-100, -1)], ArgType.FLOAT),
                    dry_bal_centre=(3, [(-3, 3)], ArgType.FLOAT),
                    dry_bal_shape=(3, [(0.1, 20.0)], ArgType.FLOAT),
                )
            )
        else:
            raise ValueError(f"Unknown 'dryness_method' {dryness_method}.")

        if fuel_build_up_method == 1:
            space_template.update(
                dict(
                    fuel_build_up_n_samples=(
                        3,
                        [(*range(100, 1301, 400),)],
                        ArgType.CHOICE,
                    ),
                    fuel_build_up_factor=(3, [(0.5, 40)], ArgType.FLOAT),
                    fuel_build_up_centre=(3, [(-1.0, 1.0)], ArgType.FLOAT),
                    fuel_build_up_shape=(3, [(0.1, 20.0)], ArgType.FLOAT),
                )
            )
        elif fuel_build_up_method == 2:
            space_template.update(
                dict(
                    litter_tc=(3, [np.geomspace(1e-10, 1e-9, 3)], ArgType.CHOICE),
                    leaf_f=(3, [np.geomspace(1e-4, 1e-3, 3)], ArgType.CHOICE),
                    litter_pool_factor=(3, [(0.001, 0.1)], ArgType.FLOAT),
                    litter_pool_centre=(3, [(10, 5000)], ArgType.FLOAT),
                    litter_pool_shape=(3, [(0.1, 20.0)], ArgType.FLOAT),
                )
            )
        else:
            raise ValueError(f"Unknown 'fuel_build_up_method' {fuel_build_up_method}.")

        if include_temperature == 1:
            space_template.update(
                dict(
                    temperature_factor=(3, [(0.19, 0.3)], ArgType.FLOAT),
                    temperature_centre=(3, [(280, 320)], ArgType.FLOAT),
                    temperature_shape=(3, [(0.1, 20.0)], ArgType.FLOAT),
                )
            )
        elif include_temperature == 0:
            pass
        else:
            raise ValueError(f"Unknown 'include_temperature' {include_temperature}.")

        space = BasinHoppingSpace(generate_space(space_template))

        exp_desc = (
            f"dry{dryness_method}fuel{fuel_build_up_method}"
            f"temp{include_temperature} - nDiscrete"
        )

        logger.info(f"Discrete param names: {space.discrete_param_names}")

        for discrete_params in tqdm(list(space.discrete_param_product), desc=exp_desc):
            args.append(
                (
                    discrete_params,
                    space,
                    dryness_method,
                    fuel_build_up_method,
                    include_temperature,
                )
            )

    pre_calculate = mod_get_parsers()["parser"].parse_args().pre_calculate
    use_pre_calculate = mod_get_parsers()["parser"].parse_args().use_pre_calculate

    discrete_param_examples = []

    if pre_calculate:
        # CHOICE parameters cause re-calculation of cached results when their value
        # changes.
        # For now, duplicate calculation only occurs for `include_temperature` 1 or 0
        # pairs.
        for arg in tqdm(args, desc="Choosing pre-calc args"):
            for p in discrete_param_examples:
                if arg[0] == p[0]:
                    break
            else:
                # If there are no matches, add to the list.
                discrete_param_examples.append(arg)

        memoize.active = False

    run(
        main,
        *zip(*(discrete_param_examples if pre_calculate else args)),
        pre_calculate=pre_calculate,
        cx1_kwargs=dict(
            walltime="24:00:00", ncpus=1, mem=("2GB" if use_pre_calculate else "25GB")
        ),
        get_parsers=mod_get_parsers,
    )
