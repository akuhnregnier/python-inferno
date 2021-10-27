# -*- coding: utf-8 -*-
from functools import partial

import hyperopt
import numpy as np
from hyperopt import Trials, fmin, hp, tpe

from python_inferno.hyperopt import HyperoptSpace


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
        rstate=np.random.RandomState(0),
        verbose=False,
    )

    space = HyperoptSpace({"x": (hp.uniform, -10, 10)})

    out = part_fmin(space=space.render(), max_evals=100)

    shrink_space = space.shrink(out, factor=0.1)
    shrink_out = part_fmin(space=shrink_space.render(), max_evals=200)

    assert abs(shrink_out["x"] - 0.5) < abs(out["x"] - 0.5)


def test_quniform_space():
    trials = Trials()

    part_fmin = partial(
        fmin,
        fn=lambda kwargs: {
            "loss": abs(kwargs["x"] - 2),
            "status": hyperopt.STATUS_OK,
        },
        trials=trials,
        algo=tpe.suggest,
        rstate=np.random.RandomState(0),
        verbose=False,
    )

    space = HyperoptSpace({"x": (hp.quniform, -10, 10, 2)})

    out = part_fmin(space=space.render(), max_evals=100)

    assert space.shrink(out, factor=0.1).spec["x"][1:] == (2, 3, 1)
    assert space.shrink(out, factor=0.5).spec["x"][1:] == (-2, 7, 2)
    assert space.shrink(out, factor=10).spec["x"][1:] == (-10, 9, 2)
