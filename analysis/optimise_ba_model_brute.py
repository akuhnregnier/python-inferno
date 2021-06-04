#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_inferno.analysis.optimise_brute import calc_param_r2
from python_inferno.cx1 import run


def calc_param_r2_wrapper(params, N=None, cache_check=False, **kwargs):
    if cache_check:
        return calc_param_r2.check_in_store(params, N=N)
    return calc_param_r2(params, N=N)


if __name__ == "__main__":
    # Define parameters to optimise.
    opt_parameters = OrderedDict(
        # fapar_factor=np.linspace(-50, -30, 3),
        # fapar_centre=np.linspace(0.25, 0.4, 3),
        # fuel_build_up_factor=np.linspace(10, 30, 3),
        # fuel_build_up_centre=np.linspace(0.3, 0.45, 3),
        # temperature_factor=np.linspace(0.08, 0.18, 3),
        # dry_day_factor=np.linspace(0.02, 0.08, 3),
        # dry_day_centre=np.linspace(100, 400, 3),
        rain_f=np.linspace(0.1e-2, 5e-2, 3),
        vpd_f=np.geomspace(0.2e2, 5e2, 3),
        dry_bal_factor=-np.geomspace(0.5e-1, 0.5e1, 4),
        dry_bal_centre=np.linspace(-0.9, 0.9, 4),
    )

    params = [
        {name: val for name, val in zip(opt_parameters, factors)}
        for factors in product(*opt_parameters.values())
    ]

    r2s = run(
        calc_param_r2_wrapper,
        params,
        cx1_kwargs=dict(
            walltime="10:00:00",
            ncpus=1,
            mem="5GB",
        ),
        N=None,
    )

    scores = {}
    for r2, param in zip(r2s, params):
        scores[tuple(param.items())] = r2

    plt.ioff()
    df_data = defaultdict(list)
    for parameter_values, r2 in scores.items():
        df_data["r2"].append(r2)
        for parameter, value in parameter_values:
            df_data[parameter].append(value)

    df = pd.DataFrame(df_data)

    for column in [col for col in df.columns if col != "r2"]:
        df.boxplot(column="r2", by=column)

    plt.show()
