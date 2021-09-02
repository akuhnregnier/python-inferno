#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from time import time

import iris
import numpy as np

from python_inferno.configuration import land_pts, npft
from python_inferno.cython.multi_timestep_inferno import cython_multi_timestep_inferno
from python_inferno.data import load_data
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import (
    monthly_average_data,
    temporal_processing,
    unpack_wrapped,
)

timestep = 4 * 60 * 60


if __name__ == "__main__":
    (
        t1p5m_tile,
        q1p5m_tile,
        pstar,
        sthu_soilt,
        frac,
        c_soil_dpm_gb,
        c_soil_rpm_gb,
        canht,
        ls_rain,
        con_rain,
        fuel_build_up,
        fapar_diag_pft,
        jules_lats,
        jules_lons,
        gfed_ba_1d,
        obs_fapar_1d,
        jules_ba_gb,
        jules_time_coord,
        npp_pft,
        npp_gb,
        climatology_output,
    ) = load_data(N=None)

    data_dict = dict(
        t1p5m_tile=t1p5m_tile,
        q1p5m_tile=q1p5m_tile,
        pstar=pstar,
        sthu_soilt=sthu_soilt,
        frac=frac,
        c_soil_dpm_gb=c_soil_dpm_gb,
        c_soil_rpm_gb=c_soil_rpm_gb,
        canht=canht,
        ls_rain=ls_rain,
        con_rain=con_rain,
        # NOTE NPP is used here now, NOT FAPAR!
        fuel_build_up=npp_pft,
        fapar_diag_pft=npp_pft,
        # TODO: How is dry-day calculation affected by climatological input data?
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=1.0, timestep=timestep
        ),
        # NOTE The target BA is only included here to ease processing. It will be
        # removed prior to the modelling function.
        gfed_ba_1d=gfed_ba_1d,
    )

    data_dict, jules_time_coord = temporal_processing(
        data_dict=data_dict,
        antecedent_shifts_dict={"fuel_build_up": [20] * npft},
        average_samples=20,
        aggregator={
            name: {"dry_days": iris.analysis.MAX, "t1p5m_tile": iris.analysis.MAX}.get(
                name, iris.analysis.MEAN
            )
            for name in data_dict
        },
        time_coord=jules_time_coord,
        climatology_input=climatology_output,
    )

    assert jules_time_coord.cell(-1).point.month == 12
    last_year = jules_time_coord.cell(-1).point.year
    for start_i in range(jules_time_coord.shape[0]):
        if jules_time_coord.cell(start_i).point.year == last_year:
            break
    else:
        raise ValueError("Target year not encountered.")

    # Trim the data and temporal coord such that the data spans a single year.
    jules_time_coord = jules_time_coord[start_i:]
    for data_name in data_dict:
        data_dict[data_name] = data_dict[data_name][start_i:]

    # Remove the target BA.
    gfed_ba_1d = data_dict.pop("gfed_ba_1d")

    # NOTE The mask array on `gfed_ba_1d` determines which samples are selected for
    # comparison later on.

    # Calculate monthly averages.
    mon_avg_gfed_ba_1d = monthly_average_data(gfed_ba_1d, time_coord=jules_time_coord)

    # Ensure the data spans a single year.
    assert mon_avg_gfed_ba_1d.shape[0] == 12
    assert (
        jules_time_coord.cell(0).point.year == jules_time_coord.cell(-1).point.year
        and jules_time_coord.cell(0).point.month == 1
        and jules_time_coord.cell(-1).point.month == 12
        and jules_time_coord.shape[0] >= 12
    )

    # XXX Dummy data
    # rng = np.random.default_rng(0)
    # N = 10
    # data_dict = dict(
    #     t1p5m_tile=rng.random((N, 17, 7771), dtype=np.float32),
    #     q1p5m_tile=rng.random((N, 17, 7771), dtype=np.float32),
    #     pstar=rng.random((N, 7771), dtype=np.float32),
    #     sthu_soilt=rng.random((N, 4, 1, 7771), dtype=np.float32),
    #     frac=rng.random((N, 17, 7771), dtype=np.float32),
    #     c_soil_dpm_gb=rng.random((N, 7771), dtype=np.float32),
    #     c_soil_rpm_gb=rng.random((N, 7771), dtype=np.float32),
    #     canht=rng.random((N, 13, 7771), dtype=np.float32),
    #     ls_rain=rng.random((N, 7771), dtype=np.float32),
    #     con_rain=rng.random((N, 7771), dtype=np.float32),
    #     fuel_build_up=rng.random((N, 13, 7771), dtype=np.float32),
    #     fapar_diag_pft=rng.random((N, 13, 7771), dtype=np.float32),
    #     dry_days=rng.random((N, 7771), dtype=np.float32),
    # )
    # XXX Dummy data

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
        rain_f=0.5,
        vpd_f=2500,
        dry_bal_factor=1,
        dry_bal_centre=0,
        # These are not used for ignition mode 1, nor do they contain a temporal
        # coordinate.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
    )

    outputs = defaultdict(list)
    times = defaultdict(list)
    for name, function in [
        ("python", multi_timestep_inferno),
        ("cython", cython_multi_timestep_inferno),
    ]:
        for i in range(1):
            start = time()
            outputs[name].append(unpack_wrapped(function)(**{**kwargs, **data_dict}))
            times[name].append(time() - start)

    for name, time_vals in times.items():
        print(f"Times taken by '{name}': {time_vals}.")
