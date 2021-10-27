#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import hyperopt
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import Trials, fmin, tpe
from tqdm import tqdm

from python_inferno.hyperopt import HyperoptSpace, mod_quniform

if __name__ == "__main__":
    seeds = 3
    iters = 200

    data = np.zeros((seeds, iters))

    losses = []

    for seed in tqdm(range(seeds), desc="Seed repetitions"):
        trials = Trials()
        part_fmin = partial(
            fmin,
            fn=lambda kwargs: {
                "loss": abs(kwargs["x"] - 2.1),
                "status": hyperopt.STATUS_OK,
            },
            algo=tpe.suggest,
            trials=trials,
            rstate=np.random.RandomState(seed),
            verbose=False,
        )

        space = HyperoptSpace({"x": (mod_quniform, -20, 20, 2)})

        out1 = part_fmin(
            space=space.render(),
            max_evals=iters // 2,
        )
        out2 = part_fmin(
            space=space.shrink(trials, factor=0.2).render(), max_evals=iters
        )

        data[seed] = trials.vals["x"]
        losses.append(trials.losses())

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].errorbar(np.arange(iters), np.mean(data, axis=0), yerr=np.std(data, axis=0))
    axes[1].plot(np.array(losses).T)
    axes[1].set_yscale("log")

    plt.show()
