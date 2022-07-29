# -*- coding: utf-8 -*-
from functools import partial
from operator import eq, ne

import hyperopt
import numpy as np
import pytest
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.pyll.stochastic import sample

from python_inferno.hyperopt import HyperoptSpace, mod_quniform
from python_inferno.space import generate_space_spec


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


@pytest.mark.parametrize(
    "spec_name, comps, var_names",
    [
        (
            "XXY",
            [("paramA", eq, "paramA2"), ("paramA", ne, "paramA3")],
            {"paramA", "paramA3"},
        ),
        (
            "XYX",
            [("paramA", eq, "paramA3"), ("paramA", ne, "paramA2")],
            {"paramA", "paramA2"},
        ),
        (
            "XYY",
            [("paramA2", eq, "paramA3"), ("paramA", ne, "paramA2")],
            {"paramA", "paramA2"},
        ),
    ],
)
@pytest.mark.parametrize("seed", range(10))
def test_param_refs(
    seed,
    spec_name,
    comps,
    var_names,
):
    rng = np.random.default_rng(seed)
    space = HyperoptSpace(
        generate_space_spec(
            space_template=dict(
                paramA=(spec_name, [(-3, 3)], hp.uniform),
                paramB=(1, [(-10, 0)], hp.uniform),
                paramC=(1, [(40, 160, 60)], mod_quniform),
            )
        )
    )

    # NOTE Some 'paramAX' may only be produced when calling `inv_map_float_to_0_1`.
    assert set(space.continuous_param_names) == {"paramB"}.union(var_names)

    mapped = space.inv_map_float_to_0_1(
        {key: rng.random() for key in space.continuous_param_names}
    )

    for comp in comps:
        name1, op, name2 = comp
        assert op(mapped[name1], mapped[name2])

    assert mapped["paramB"]

    for name, val in mapped.items():
        if "paramA" in name:
            assert val >= -3 and val <= 3
        elif "paramB" in name:
            assert val >= -10 and val <= 0


def test_space_mapping():
    space = HyperoptSpace({"x": (hp.uniform, -10, 2)})
    assert np.isclose(space.inv_map_float_to_0_1({"x": 0.0})["x"], -10)
    assert np.isclose(space.inv_map_float_to_0_1({"x": 1.0})["x"], 2)
    assert np.isclose(space.map_float_to_0_1({"x": -10})["x"], 0)
    assert np.isclose(space.map_float_to_0_1({"x": 2})["x"], 1)
