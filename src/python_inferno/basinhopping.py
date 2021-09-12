#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import string
import sys
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import basinhopping
from wildfires.dask_cx1.dask_rf import safe_write

from python_inferno.cx1 import run


class BasinHoppingSpace:
    def __init__(self, spec):
        for (arg_type, *args) in spec.values():
            assert arg_type in (float, int)
            if arg_type == float:
                assert len(args) == 2
            elif arg_type == int:
                assert len(args) == 3

        self.spec = spec

    def inv_map_float_to_0_1(self, params):
        """Undo mapping of floats to [0, 1].

        This maps the floats back to their original range.

        """
        remapped = {}
        for name, value in params.items():
            arg_type, *args = self.spec[name]
            if arg_type == float:
                minb, maxb = args
                remapped[name] = (value * (maxb - minb)) + minb
            else:
                remapped[name] = value

        return remapped

    @property
    def float_param_names(self):
        """Return the list of floating point parameters which are to be optimised."""
        return tuple(name for name, value in self.spec.items() if value[0] == float)

    @property
    def int_param_names(self):
        """Return the list of integer parameters."""
        return tuple(name for name, value in self.spec.items() if value[0] == int)

    @property
    def int_param_product(self):
        """Yield all integer parameter combinations."""
        iterables = [range(*self.spec[name][1:]) for name in self.int_param_names]
        for ps in product(*iterables):
            yield dict(zip(self.int_param_names, ps))

    @property
    def float_x0_mid(self):
        """The midpoints of all floating point parameter ranges."""
        return [0.5] * len(self.float_param_names)


class Recorder:
    def __init__(self, record_dir=None):
        """Initialise."""
        self.xs = []
        self.fvals = []
        self.filename = Path(record_dir) / (
            "".join(random.choices(string.ascii_lowercase, k=20)) + ".pkl"
        )
        self.filename.parent.mkdir(exist_ok=True)
        logger.info(f"Minima record filename: {self.filename}")

    def record(self, x, fval):
        """Record parameters and function value."""
        self.xs.append(x)
        self.fvals.append(fval)

    def dump(self):
        """Dump the recorded values to file."""
        safe_write((self.xs, self.fvals), self.filename)


class BoundedSteps:
    def __init__(self, stepsize=0.5, rng=None):
        self.stepsize = stepsize

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def __call__(self, x):
        """Return new coordinates relative to existing coordinates `x`."""
        # New coords cannot be outside of [0, 1].
        logger.info(
            f"Taking a step with stepsize '{self.stepsize:0.5f}' (in [0, 1] space)"
        )
        logger.info(f"Old pos: {x}")

        min_pos = np.clip(x - self.stepsize, 0, 1)
        max_pos = np.clip(x + self.stepsize, 0, 1)

        new = self.rng.uniform(low=min_pos, high=max_pos)

        logger.info(f"New pos: {new}")
        return new


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
