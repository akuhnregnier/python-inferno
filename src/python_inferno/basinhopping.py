#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import string
from enum import Enum
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger
from wildfires.dask_cx1.dask_rf import safe_write

from .space import OptSpace

ArgType = Enum("ArgType", ["FLOAT", "CHOICE"])


class BasinHoppingSpace(OptSpace):
    def __init__(self, spec):
        super().__init__(
            spec=spec,
            float_type=ArgType.FLOAT,
            continuous_types={ArgType.FLOAT},
            discrete_types={ArgType.CHOICE},
        )

    @property
    def discrete_param_product(self):
        """Yield all choice parameter combinations."""
        iterables = [self.spec[name][1:] for name in self.discrete_param_names]
        for ps in product(*iterables):
            yield dict(zip(self.discrete_param_names, ps))


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
    def __init__(self, stepsize=0.5, rng=None, verbose=True):
        self.stepsize = stepsize
        self.verbose = verbose

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def __call__(self, x):
        """Return new coordinates relative to existing coordinates `x`."""
        # New coords cannot be outside of [0, 1].
        if self.verbose:
            logger.info(
                f"Taking a step with stepsize '{self.stepsize:0.5f}' (in [0, 1] space)"
            )
            logger.info(f"Old pos: {x}")

        min_pos = np.clip(x - self.stepsize, 0, 1)
        max_pos = np.clip(x + self.stepsize, 0, 1)

        new = self.rng.uniform(low=min_pos, high=max_pos)

        if self.verbose:
            logger.info(f"New pos: {new}")
        return new
