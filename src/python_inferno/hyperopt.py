# -*- coding: utf-8 -*-

import numpy as np
from hyperopt import hp


class Space:
    def __init__(self, spec):
        self.spec = spec

    def render(self):
        """Get the final dictionary given the hyperopt's `fmin`."""
        out = {}
        for (name, (arg_type, *args)) in self.spec.items():
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

                    start = new_values[0]
                    end = new_values[-1] + 1
                    step = 1 if len(new_values) == 1 else new_values[1] - new_values[0]

                    new[name] = (arg_type, start, end, step)
            else:
                raise ValueError(f"Unsupported `arg_type`: {arg_type}.")

        return Space(new)

    def __str__(self):
        return str(self.spec)
