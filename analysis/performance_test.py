#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from time import time

import numpy as np

from python_inferno.configuration import land_pts
from python_inferno.data import get_processed_climatological_data, timestep
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.utils import unpack_wrapped

if __name__ == "__main__":
    data_dict, mon_avg_gfed_ba_1d, jules_time_coord = get_processed_climatological_data(
        n_samples_pft=np.array([20] * 13, dtype=np.int64), average_samples=20
    )

    # Model kwargs.
    kwargs = dict(
        ignition_method=1,
        timestep=timestep,
        flammability_method=2,
        dryness_method=2,
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=0.0,
        dry_day_centre=0.0,
        # TODO - calculation of dry_bal is carried out during data loading/processing
        # now
        # rain_f=0.5,
        # vpd_f=2500,
        dry_bal_factor=1,
        dry_bal_centre=0,
        # These are not used for ignition mode 1, nor do they contain a temporal
        # coordinate.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
    )

    outputs = dict()
    times = defaultdict(list)
    for name, function in [
        ("python", multi_timestep_inferno),
    ]:
        for i in range(2):
            start = time()
            outputs[name] = unpack_wrapped(function)(**{**kwargs, **data_dict})
            times[name].append(time() - start)

    for name, time_vals in times.items():
        print(f"Times taken by '{name}': {time_vals}.")
