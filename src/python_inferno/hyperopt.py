# -*- coding: utf-8 -*-
import math
import os
from pathlib import Path
from pprint import pformat
from time import time

import hyperopt
import numpy as np
from hyperopt import hp
from hyperopt.pyll import rec_eval
from loguru import logger
from scipy.optimize import basinhopping

from python_inferno.space import OptSpace

from .ba_model import gen_to_optimise
from .basinhopping import BoundedSteps, Recorder


def mod_quniform(name, low, high, q):
    """Adapted `hp.quniform`. Returns values in [low, high]."""
    # NOTE: see https://github.com/hyperopt/hyperopt/issues/328
    # The parameter value returned by hyperopt `fmin` will not match the value given
    # to the optimised function as `low` is added on here afterwards.
    n = (high - low) / q
    if (round(n) - n) > 1e-6:
        raise ValueError("'high - low' should be integer multiple of 'q'")
    return hp.quniform(name, -q / 2.0, high - low + q / 2.0, q) + low


class HyperoptSpace(OptSpace):
    def __init__(self, spec):
        super().__init__(
            spec=spec,
            float_type=hp.uniform,
            continuous_types={hp.uniform},
            discrete_types={mod_quniform},
        )

    def render(self):
        """Get the final `space` dictionary given to hyperopt's `fmin`."""
        out = {}
        for (name, (arg_type, *args)) in self.spec.items():
            out[name] = arg_type(name, *args)
        return out

    def render_discrete(self):
        """Get the final `space` dictionary given to hyperopt's `fmin`.

        Only considers discrete parameters.

        """
        out = {}
        for (name, (arg_type, *args)) in self.spec.items():
            if arg_type in self.discrete_types:
                out[name] = arg_type(name, *args)
        return out

    def shrink(self, trials, factor=0.5):
        """Reduce the range of the values."""
        best_vals = trials.argmin
        new = {}
        for (name, (arg_type, *args)) in self.spec.items():
            best = best_vals[name]

            if arg_type == hp.uniform:
                assert len(args) == 2
                new_range = factor * (args[1] - args[0])
                new[name] = (
                    arg_type,
                    np.clip(best - new_range / 2, args[0], args[1]),
                    np.clip(best + new_range / 2, args[0], args[1]),
                )
            elif arg_type == mod_quniform:
                assert len(args) == 3
                low, high, q = args

                # NOTE: This is a result of the addition `+ low` in the definition of
                # `mod_quniform` - hyperopt does not consider such modifications when
                # reporting the results of trials, only considering the internal call
                # to `hp.quniform` in this case.
                best += low

                if (high - low) / q <= 1:
                    # Nothing left to do here.
                    assert np.isclose(args[0], best)
                    new[name] = (arg_type, best, best + q / 2.0, q)
                else:
                    new_half_range = factor * (high - low) / 2.0
                    if new_half_range < q:
                        # No valid samples in new range.
                        # Only pick the given best value instead.
                        new[name] = (arg_type, best, best + q / 2.0, q)
                    else:
                        # Ignore those values outside of the new range.
                        step_extent = math.floor((new_half_range) / q) * q
                        start = max(low, best - step_extent)
                        end = min(high, best + step_extent)
                        new[name] = (arg_type, start, end, q)
            else:
                raise ValueError(
                    f"Unsupported `arg_type`: {arg_type} for parameter '{name}'."
                )
        return type(self)(new)

    @property
    def n_discrete_product(self):
        product = 1
        for name in self.discrete_param_names:
            (arg_type, *args) = self.spec[name]
            if arg_type == mod_quniform:
                low, high, q = args
                # NOTE: This is highly susceptible to floating point errors, and
                # therefore relies on the checks in `mod_quniform`.
                product *= math.ceil((high - low + q / 2.0) / q)
            else:
                raise ValueError("Unsupported `arg_type`.")
        return product


def fail_func(*args, **kwargs):
    return 10000.0


def success_func(loss, *args, **kwargs):
    return loss


def main_opt(
    expr,
    memo,
    ctrl,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
):
    discrete_params = rec_eval(expr, memo=memo)
    logger.info(f"Discrete parameters:\n{pformat(discrete_params)}")

    start = time()

    # NOTE: These values may differ from `discrete_params` above (but they have the same
    # keys) if modifications like `mod_quniform` are used.
    curr_vals = ctrl.current_trial["misc"]["vals"]

    # Fetch previous trials.
    ctrl.trials.refresh()
    for trial in ctrl.trials.trials:
        if (trial["misc"]["vals"] == curr_vals) and (
            trial["result"]["status"] != "new"
        ):
            logger.info(f"Match found (search time: {time() - start:0.2f}).")
            return trial["result"]

    logger.info(f"No match (search time: {time() - start:0.2f}).")

    to_optimise = gen_to_optimise(
        fail_func=fail_func,
        success_func=success_func,
        # Init (data) params.
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        **discrete_params,
    )

    def to_optimise_with_discrete(x):
        return to_optimise(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x)))
        )

    recorder = Recorder(record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record")

    def basinhopping_callback(x, f, accept):
        # NOTE: Parameters recorded here are authoritative, since hyperopt will not
        # properly report values modified as in e.g. `mod_quniform`.
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

    res = basinhopping(
        to_optimise_with_discrete,
        x0=space.continuous_x0_mid,
        disp=True,
        minimizer_kwargs=dict(
            method="L-BFGS-B",
            jac=None,
            bounds=[(0, 1)] * len(space.continuous_param_names),
            options=dict(maxiter=60, ftol=1e-5, eps=1e-3),
        ),
        seed=0,
        T=0.05,
        niter=100,
        niter_success=15,
        callback=basinhopping_callback,
        take_step=BoundedSteps(stepsize=0.3, rng=np.random.default_rng(0)),
    )

    loss = res.fun

    if loss > 100.0:
        return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}
    else:
        return {"loss": loss, "status": hyperopt.STATUS_OK}
