# -*- coding: utf-8 -*-
import math

import numpy as np
from hyperopt import hp

from python_inferno.space import OptSpace


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
