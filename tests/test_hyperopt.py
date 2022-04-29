# -*- coding: utf-8 -*-
from functools import partial

import hyperopt
import numpy as np
import pytest
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.pyll.stochastic import sample

from python_inferno.hyperopt import HyperoptSpace, mod_quniform


def test_uniform_space():
    trials = Trials()

    part_fmin = partial(
        fmin,
        fn=lambda kwargs: {
            "loss": abs(kwargs["x"] - 0.5),
            "status": hyperopt.STATUS_OK,
        },
        trials=trials,
        algo=tpe.suggest,
        rstate=np.random.default_rng(0),
        verbose=False,
    )

    space = HyperoptSpace({"x": (hp.uniform, -10, 10)})

    out = part_fmin(space=space.render(), max_evals=100)

    shrink_space = space.shrink(trials, factor=0.1)
    shrink_out = part_fmin(space=shrink_space.render(), max_evals=200)

    assert abs(shrink_out["x"] - 0.5) < abs(out["x"] - 0.5)


def test_mod_quniform_space():
    trials = Trials()

    part_fmin = partial(
        fmin,
        fn=lambda kwargs: {
            "loss": abs(kwargs["x"] - 2),
            "status": hyperopt.STATUS_OK,
        },
        trials=trials,
        algo=tpe.suggest,
        rstate=np.random.default_rng(0),
        verbose=False,
    )

    space = HyperoptSpace({"x": (mod_quniform, -10, 10, 2)})

    part_fmin(space=space.render(), max_evals=100)

    assert np.allclose(space.shrink(trials, factor=0.1).spec["x"][1:], (2, 3, 2))
    assert np.allclose(space.shrink(trials, factor=0.5).spec["x"][1:], (-2, 6, 2))
    assert np.allclose(space.shrink(trials, factor=10).spec["x"][1:], (-10, 10, 2))


def test_mod_quniform_check():
    with pytest.raises(ValueError):
        mod_quniform("test", 2, 7, 3)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((2, 8, 3), (2, 5, 8)),
        ((0.2, 0.8, 0.3), (0.2, 0.5, 0.8)),
        ((1e-10, 5e-10, 2e-10), (1e-10, 3e-10, 5e-10)),
        ((2, 2 + 1e-10, 3), (2,)),
        ((-8, -2, 3), (-8, -5, -2)),
        ((-10, 10, 2), (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10)),
    ],
)
def test_mod_quniform_samples(args, expected):
    rng = np.random.default_rng(0)
    samples = [sample(mod_quniform("test", *args), rng=rng) for _ in range(int(1e4))]
    values, counts = np.unique(samples, return_counts=True)
    assert np.std(counts) < (0.04 * np.mean(counts))
    assert np.allclose(values, expected)


@pytest.mark.parametrize(
    "args, n",
    [
        ((2, 8, 3), 3),
        ((0.2, 0.8, 0.3), 3),
        ((1e-10, 5e-10, 2e-10), 3),
        ((2, 2 + 1e-5, 3), 1),
    ],
)
def test_single_n_discrete_product(args, n):
    assert HyperoptSpace({"x": (mod_quniform, *args)}).n_discrete_product == n


def test_multi_n_discrete_product():
    assert (
        HyperoptSpace(
            {
                "x": (mod_quniform, 2, 8, 3),
                "y": (mod_quniform, 1, 4, 1),
            }
        ).n_discrete_product
        == 12
    )

    assert (
        HyperoptSpace(
            {
                "x": (mod_quniform, 2, 8, 3),
                "y": (mod_quniform, 1, 4, 1),
                "z": (mod_quniform, 1e-10, 5.5e-10, 0.5e-10),
            }
        ).n_discrete_product
        == 120
    )
