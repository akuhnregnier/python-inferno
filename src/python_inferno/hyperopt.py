# -*- coding: utf-8 -*-
import math
from pprint import pformat

import hyperopt
import numpy as np
from hyperopt import hp
from hyperopt.pyll import rec_eval
from loguru import logger

from .cache import mark_dependency
from .configuration import default_opt_record_dir
from .space import OptSpace
from .space_opt import space_opt


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


@mark_dependency
def get_space_template(*, dryness_method, fuel_build_up_method, include_temperature):
    space_template = dict(
        overall_scale=(1, [(1e-3, 3e3)], hp.uniform),
        fapar_factor=(3, [(-100, -1)], hp.uniform),
        fapar_centre=(3, [(-1.5, 1.5)], hp.uniform),
        fapar_shape=(3, [(0.1, 30.0)], hp.uniform),
        # NOTE All weights should be in [0, 1], otherwise unintended -ve values
        # may occur!
        fapar_weight=(3, [(0.01, 1.0)], hp.uniform),
        dryness_weight=(3, [(0.01, 1.0)], hp.uniform),
        fuel_weight=(3, [(0.01, 1.0)], hp.uniform),
        # Averaged samples with 42, 114, or 186 4-hour timesteps (7, 19, or 31 days)
        average_samples=(1, [(42, 186, 72)], mod_quniform),
        # `crop_f` suppresses BA in cropland areas.
        crop_f=(1, [(0.0, 0.65)], hp.uniform),
    )
    if dryness_method == 1:
        space_template.update(
            dict(
                dry_day_factor=(3, [(0.0, 0.6)], hp.uniform),
                dry_day_centre=(3, [(100, 200)], hp.uniform),
                dry_day_shape=(3, [(0.1, 30.0)], hp.uniform),
            )
        )
    elif dryness_method == 2:
        space_template.update(
            dict(
                rain_f=(3, [(0.1, 0.6, 0.25)], mod_quniform),
                vpd_f=(3, [(50, 200, 75)], mod_quniform),
                dry_bal_factor=(3, [(-100, -0.1)], hp.uniform),
                dry_bal_centre=(3, [(-3, 3)], hp.uniform),
                dry_bal_shape=(3, [(0.01, 30.0)], hp.uniform),
            )
        )
    else:
        raise ValueError(f"Unknown 'dryness_method' {dryness_method}.")

    if fuel_build_up_method == 1:
        space_template.update(
            dict(
                fuel_build_up_n_samples=(3, [(96, 1302, 402)], mod_quniform),
                fuel_build_up_factor=(3, [(0.5, 40)], hp.uniform),
                fuel_build_up_centre=(3, [(-2, 2)], hp.uniform),
                fuel_build_up_shape=(3, [(0.05, 40.0)], hp.uniform),
            )
        )
    elif fuel_build_up_method == 2:
        space_template.update(
            dict(
                litter_tc=(3, [(1e-10, 1e-9, 4.5e-10)], mod_quniform),
                leaf_f=(3, [(1e-4, 1e-3, 4.5e-4)], mod_quniform),
                litter_pool_factor=(3, [(5e-4, 0.1)], hp.uniform),
                litter_pool_centre=(3, [(1, 1e4)], hp.uniform),
                litter_pool_shape=(3, [(1e-3, 20.0)], hp.uniform),
            )
        )
    else:
        raise ValueError(f"Unknown 'fuel_build_up_method' {fuel_build_up_method}.")

    if include_temperature == 1:
        space_template.update(
            dict(
                temperature_factor=(3, [(0.05, 1.0)], hp.uniform),
                temperature_centre=(3, [(280, 320)], hp.uniform),
                temperature_shape=(3, [(0.05, 60.0)], hp.uniform),
                temperature_weight=(3, [(0.01, 1.0)], hp.uniform),
            )
        )
    elif include_temperature == 0:
        pass
    else:
        raise ValueError(f"Unknown 'include_temperature' {include_temperature}.")

    return space_template


def hyperopt_opt(
    expr,
    memo,
    ctrl,
):
    discrete_params = rec_eval(expr, memo=memo)
    logger.info(f"Discrete parameters:\n{pformat(discrete_params)}")

    # NOTE: These values may differ from `discrete_params` above (but they have the same
    # keys) if modifications like `mod_quniform` are used.
    curr_vals = ctrl.current_trial["misc"]["vals"]

    # Fetch previous trials.
    ctrl.trials.refresh()
    for trial in ctrl.trials.trials:
        if (trial["misc"]["vals"] == curr_vals) and (
            trial["result"]["status"] != "new"
        ):
            logger.info("Prior results found.")
            return trial["result"], discrete_params
    return None, discrete_params


def main_opt(
    expr,
    memo,
    ctrl,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    opt_record_dir=default_opt_record_dir,
):
    hyperopt_res, discrete_params = hyperopt_opt(expr=expr, memo=memo, ctrl=ctrl)
    if hyperopt_res is not None:
        return hyperopt_res

    loss = float(
        space_opt(
            space=space,
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            discrete_params=discrete_params,
            opt_record_dir=opt_record_dir,
        )
    )

    if loss > 100.0:
        return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}
    else:
        return {"loss": loss, "status": hyperopt.STATUS_OK}
