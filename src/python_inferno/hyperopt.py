# -*- coding: utf-8 -*-
import numpy as np
from hyperopt import hp

from python_inferno.space import OptSpace

continuous_types = {hp.uniform}
discrete_types = {hp.quniform}


class HyperoptSpace(OptSpace):
    def __init__(self, spec):
        super().__init__(
            spec=spec,
            float_type=hp.uniform,
            continuous_types={hp.uniform},
            discrete_types={hp.quniform},
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

    def shrink(self, best_vals, factor=0.5):
        """Reduce the range of the values."""
        new = {}
        for (name, (arg_type, *args)) in self.spec.items():
            if arg_type == hp.uniform:
                assert len(args) == 2
                new_range = factor * (args[1] - args[0])
                new[name] = (
                    arg_type,
                    np.clip(best_vals[name] - new_range / 2, args[0], args[1]),
                    np.clip(best_vals[name] + new_range / 2, args[0], args[1]),
                )
            elif arg_type == hp.quniform:
                assert len(args) == 3
                orig_values = np.arange(args[0], args[1], args[2])
                if len(orig_values) == 1:
                    # Nothing left to do here.
                    assert np.isclose(args[0], best_vals[name])
                    new[name] = (arg_type, best_vals[name], best_vals[name] + 1, 1)
                else:
                    new_range = factor * np.ptp(orig_values)

                    # Ignore those values outside of the new range.
                    selection = (orig_values >= best_vals[name] - new_range / 2) & (
                        orig_values <= best_vals[name] + new_range / 2
                    )
                    new_values = orig_values[selection]

                    if not np.any(selection):
                        # No valid samples in new range.
                        # Pick the given best value instead.
                        new[name] = (arg_type, best_vals[name], best_vals[name] + 1, 1)
                    else:
                        start = new_values[0]
                        end = new_values[-1] + 1
                        step = (
                            1 if len(new_values) == 1 else new_values[1] - new_values[0]
                        )

                        new[name] = (arg_type, start, end, step)
            else:
                raise ValueError(
                    f"Unsupported `arg_type`: {arg_type} for parameter '{name}'."
                )

        return type(self)(new)
