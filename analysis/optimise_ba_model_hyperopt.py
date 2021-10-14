#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import hyperopt
import numpy as np
from hyperopt import fmin, hp, tpe
from hyperopt.mongoexp import MongoTrials

from python_inferno.ba_model import gen_to_optimise
from python_inferno.hyperopt import Space
from python_inferno.space import generate_space


def fail_func(*args, **kwargs):
    return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}


def success_func(loss, *args, **kwargs):
    return {
        "loss": loss,
        "status": hyperopt.STATUS_OK,
    }


if __name__ == "__main__":
    to_optimise = gen_to_optimise(
        fail_func=fail_func,
        success_func=success_func,
    )

    space_template = dict(
        fapar_factor=(1, [(-50, -1)], hp.uniform),
        fapar_centre=(1, [(-0.1, 1.1)], hp.uniform),
        fuel_build_up_n_samples=(1, [(100, 1301, 400)], hp.quniform),
        fuel_build_up_factor=(1, [(0.5, 30)], hp.uniform),
        fuel_build_up_centre=(1, [(0.0, 0.5)], hp.uniform),
        temperature_factor=(1, [(0.07, 0.2)], hp.uniform),
        temperature_centre=(1, [(260, 295)], hp.uniform),
        rain_f=(1, [(0.8, 2.0)], hp.uniform),
        vpd_f=(1, [(400, 2200)], hp.uniform),
        dry_bal_factor=(1, [(-100, -1)], hp.uniform),
        dry_bal_centre=(1, [(-3, 3)], hp.uniform),
        # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
        average_samples=(1, [(40, 161, 60)], hp.quniform),
    )

    space = Space(generate_space(space_template))

    trials = MongoTrials(
        "mongo://maritimus.webredirect.org:1234/ba/jobs", exp_key="exp30"
    )

    part_fmin = partial(
        fmin,
        fn=to_optimise,
        algo=tpe.suggest,
        trials=trials,
        rstate=np.random.RandomState(0),
    )

    out1 = part_fmin(space=space.render(), max_evals=5000)

    # shrink_space = space.shrink(out1, factor=0.2)
    # out2 = part_fmin(space=shrink_space.render(), max_evals=4000)
