#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import basinhopping

from python_inferno.basinhopping import BasinHoppingSpace, BoundedSteps, Recorder
from python_inferno.cx1 import run
from python_inferno.optimisation import gen_to_optimise
from python_inferno.space import generate_space


def fail_func(*args, **kwargs):
    return 10000.0


def success_func(loss, *args, **kwargs):
    return loss


to_optimise = gen_to_optimise(
    fail_func=fail_func,
    success_func=success_func,
)


space_template = dict(
    fapar_factor=(1, [(-50, -1)], float),
    fapar_centre=(1, [(-0.1, 1.1)], float),
    fuel_build_up_n_samples=(1, [(100, 1301, 400)], int),
    fuel_build_up_factor=(1, [(0.5, 30)], float),
    fuel_build_up_centre=(1, [(0.0, 0.5)], float),
    temperature_factor=(1, [(0.07, 0.2)], float),
    temperature_centre=(1, [(260, 295)], float),
    # rain_f=(1, [(0.1, 2.0)], float),
    # vpd_f=(1, [(5, 4000)], float),
    dry_bal_factor=(1, [(-100, -1)], float),
    dry_bal_centre=(1, [(-3, 3)], float),
    # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
    average_samples=(1, [(40, 161, 60)], int),
)

space = BasinHoppingSpace(generate_space(space_template))

record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"

if record_dir is not None:
    recorder = Recorder(record_dir=record_dir)
else:
    recorder = None


def main(integer_params, *args, **kwargs):
    def to_optimise_with_int(x):
        opt_kwargs = {
            **space.inv_map_float_to_0_1(dict(zip(space.float_param_names, x))),
            **integer_params,
        }
        return to_optimise(opt_kwargs)

    def basinhopping_callback(x, f, accept):
        values = space.inv_map_float_to_0_1(
            {**dict(zip(space.float_param_names, x)), **integer_params}
        )
        logger.info(f"Minimum found | loss: {f:0.6f}")

        for name, val in values.items():
            logger.info(f" - {name}: {val}")

        if recorder is not None:
            recorder.record(values, f)

            # Update record in file.
            recorder.dump()

    return (
        basinhopping(
            to_optimise_with_int,
            x0=space.float_x0_mid,
            disp=True,
            minimizer_kwargs=dict(
                method="L-BFGS-B",
                jac=None,
                bounds=[(0, 1)] * len(space.float_param_names),
                options=dict(maxiter=50, ftol=1e-5, eps=1e-3),
            ),
            seed=0,
            niter_success=10,
            callback=basinhopping_callback,
            take_step=BoundedSteps(stepsize=0.5, rng=np.random.default_rng(0)),
        ),
        space,
        integer_params,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    run(
        main,
        list(space.int_param_product),
        cx1_kwargs=dict(walltime="24:00:00", ncpus=2, mem="10GB"),
    )
