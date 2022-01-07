#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from time import time

from python_inferno.ba_model import get_pred_ba_prep
from python_inferno.data import timestep
from python_inferno.utils import unpack_wrapped

if __name__ == "__main__":
    # Model kwargs.
    kwargs = dict(
        # Data args.
        litter_tc=1e-9,
        leaf_f=1e-4,
        fuel_build_up_n_samples=20,
        average_samples=40,
        rain_f=0.5,
        vpd_f=50,
        # Model args.
        ignition_method=1,
        timestep=timestep,
        flammability_method=2,
        dryness_method=2,
        fuel_build_up_method=2,
        include_temperature=1,
        crop_f=0.5,
        dry_bal_centre=0.14,
        dry_bal_factor=-54,
        dry_bal_shape=10,
        fapar_centre=0.8,
        fapar_factor=-22,
        fapar_shape=9.2,
        litter_pool_centre=2000,
        litter_pool_factor=0.05,
        litter_pool_shape=9.2,
        temperature_centre=299,
        temperature_factor=0.23,
        temperature_shape=10,
    )

    outputs = dict()
    times = defaultdict(list)
    for name, function in [
        ("python", get_pred_ba_prep),
    ]:
        for i in range(1):
            start = time()
            outputs[name] = unpack_wrapped(function)(**kwargs)
            times[name].append(time() - start)

    for name, time_vals in times.items():
        print(f"Times taken by '{name}': {time_vals}.")
