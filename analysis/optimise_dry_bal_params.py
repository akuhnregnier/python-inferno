#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
from loguru import logger
from numba import njit, prange
from scipy.optimize import basinhopping

from python_inferno.basinhopping import BasinHoppingSpace, BoundedSteps, Recorder
from python_inferno.data import get_climatological_grouped_dry_bal
from python_inferno.space import generate_space

space_template = dict(
    rain_f=(1, [(0.5, 20.0)], float),
    vpd_f=(1, [(1, 5000)], float),
)

space = BasinHoppingSpace(generate_space(space_template))

record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"

if record_dir is not None:
    recorder = Recorder(record_dir=record_dir)
else:
    recorder = None


# Histogram bins for `loss1`.
bins = np.linspace(-1, 1, 20)
# Need equally spaced bins.
bin_diff = np.diff(bins)[0]
assert np.all(np.isclose(np.diff(bins), bin_diff))


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def calc_hist_loss(*, dry_bal, bins, hists):
    for i in prange(dry_bal.shape[1]):
        hists[i] = np.histogram(dry_bal[:, i], bins=bins)[0]
    hists /= dry_bal.shape[0] * (bins[1] - bins[0])
    # Minimise the amount of variation between bins - all values should be represented
    # as equally as possible.
    # Normalise this metric by the number of samples.
    return np.linalg.norm(hists - 0.5) / np.sqrt(hists.size)


def to_optimise(x):
    dry_bal = get_climatological_grouped_dry_bal._orig_func(
        **space.inv_map_float_to_0_1(dict(zip(space.float_param_names, x))),
        verbose=False,
    )
    # Select a single PFT since we are only using single parameters.
    dry_bal = dry_bal[:, 0]
    assert len(dry_bal.shape) == 2

    if not hasattr(to_optimise, "hists"):
        to_optimise.hists = np.empty((dry_bal.shape[1], bins.size - 1))

    loss1 = calc_hist_loss(dry_bal=dry_bal, bins=bins, hists=to_optimise.hists)

    # At the same time, the `dry_bal` variable should fluctuate between high and low
    # values (instead of e.g. monotonically increasing).
    # Add a factor to enhance the weight of this metric.
    loss2 = abs(
        (
            (
                np.sum(np.diff(dry_bal, axis=0) < 0)
                / ((dry_bal.shape[0] - 1) * dry_bal.shape[1])
            )
            - 0.5
        )
    )

    loss3 = np.mean(np.abs(dry_bal[0] - dry_bal[-1]))

    loss4 = np.sum(np.abs(np.diff(dry_bal, axis=0)) < 0.02) / dry_bal.size

    return loss1 + loss2 + 2 * loss3 + 100 * loss4


if __name__ == "__main__":

    def basinhopping_callback(x, f, accept):
        values = space.inv_map_float_to_0_1(dict(zip(space.float_param_names, x)))
        logger.info(f"Minimum found | loss: {f:0.6f}")

        for name, val in values.items():
            logger.info(f" - {name}: {val}")

        if recorder is not None:
            recorder.record(values, f)

            # Update record in file.
            recorder.dump()

    basinhopping(
        to_optimise,
        x0=space.float_x0_mid,
        disp=True,
        minimizer_kwargs=dict(
            method="L-BFGS-B",
            jac=None,
            bounds=[(0, 1)] * len(space.float_param_names),
            options=dict(maxiter=100, ftol=1e-6, eps=1e-4),
        ),
        seed=0,
        niter_success=10,
        callback=basinhopping_callback,
        take_step=BoundedSteps(stepsize=1, rng=np.random.default_rng(1)),
    )
