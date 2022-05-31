# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import basinhopping

from .ba_model import gen_to_optimise
from .basinhopping import BoundedSteps, Recorder
from .cache import cache


def fail_func(*args, **kwargs):
    return 10000.0


def success_func(loss, *args, **kwargs):
    return loss


@cache(ignore=["verbose", "_uncached_data"])
def space_opt(
    *,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    discrete_params,
    opt_record_dir="opt_record",
    defaults=None,
    basinhopping_options=None,
    minimizer_options=None,
    verbose=True,
    _uncached_data=True,
):
    """Optimisation of the continuous (float) part of a given `space`."""
    to_optimise = gen_to_optimise(
        fail_func=fail_func,
        success_func=success_func,
        # Init (data) params.
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        _uncached_data=_uncached_data,
        **discrete_params,
    )

    defaults_dict = defaults if defaults is not None else {}

    def to_optimise_with_discrete(x):
        return to_optimise(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **defaults_dict,
        )

    recorder = Recorder(record_dir=Path(os.environ["EPHEMERAL"]) / opt_record_dir)

    def basinhopping_callback(x, f, accept):
        # NOTE: Parameters recorded here are authoritative, since hyperopt will not
        # properly report values modified as in e.g. `mod_quniform`.
        values = {
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **discrete_params,
            **defaults_dict,
        }
        values["dryness_method"] = dryness_method
        values["fuel_build_up_method"] = fuel_build_up_method
        values["include_temperature"] = include_temperature

        if verbose:
            logger.info(f"Minimum found | loss: {f:0.6f}")

        for name, val in values.items():
            if verbose:
                logger.info(f" - {name}: {val}")

        if recorder is not None:
            recorder.record(values, f)

            # Update record in file.
            recorder.dump()

    minimizer_options_dict = minimizer_options if minimizer_options is not None else {}
    basinhopping_options_dict = (
        basinhopping_options if basinhopping_options is not None else {}
    )

    res = basinhopping(
        to_optimise_with_discrete,
        x0=space.continuous_x0_mid,
        seed=0,
        callback=basinhopping_callback,
        take_step=BoundedSteps(
            stepsize=0.3, rng=np.random.default_rng(0), verbose=verbose
        ),
        **{
            "disp": verbose,
            "minimizer_kwargs": dict(
                method="L-BFGS-B",
                jac=None,
                bounds=[(0, 1)] * len(space.continuous_param_names),
                options={
                    "maxiter": 60,
                    "ftol": 1e-5,
                    "eps": 1e-3,
                    **minimizer_options_dict,
                },
            ),
            "T": 0.05,
            "niter": 100,
            "niter_success": 15,
            **basinhopping_options_dict,
        },
    )

    loss = res.fun
    return loss
