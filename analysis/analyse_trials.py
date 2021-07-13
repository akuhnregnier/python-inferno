#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from hyperopt.mongoexp import MongoTrials

if __name__ == "__main__":
    trials = MongoTrials("mongo://localhost:1234/ba/jobs", exp_key="exp8")
    losses = np.array(trials.losses())
    selection = np.array([l is not None and l < 1 and l > 0.8 for l in trials.losses()])

    for name, vals in trials.vals.items():
        vals = np.array(vals)
        plt.figure()
        plt.title(name)
        plt.plot(
            vals[selection], losses[selection], marker="o", alpha=0.4, linestyle=""
        )
        plt.xlabel(name)
        plt.ylabel("loss")
        plt.show()
