#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

if __name__ == "__main__":
    opt_record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    xs = []
    fvals = []
    for fpath in opt_record_dir.glob("*.pkl"):
        with fpath.open("rb") as f:
            single_xs, single_fvals = pickle.load(f)
        xs.extend(single_xs)
        fvals.extend(single_fvals)

    keys = set()
    for params in xs:
        keys.update(params.keys())

    N = len(keys)
    nrows = math.floor(N ** 0.5)
    ncols = math.ceil(N / nrows)

    fig, axes = plt.subplots(nrows, ncols)

    for thres in [-100, 0.075, 0.086]:
        valid_indices = [i for i, r2 in enumerate(fvals) if r2 > thres]

        valid_xs = [xs[i] for i in valid_indices]
        valid_fvals = [fvals[i] for i in valid_indices]

        param_values = defaultdict(list)
        for key in keys:
            for params in valid_xs:
                param_values[key].append(params[key])

        for ((key, vals), ax) in zip(param_values.items(), axes.ravel()):
            ax.hist(vals, alpha=0.7, label=format(thres, "0.3f"), density=True)
            ax.set_xlabel(key)
            ax.legend()
