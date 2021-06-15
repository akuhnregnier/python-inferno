# -*- coding: utf-8 -*-
import os
from pathlib import Path

import iris
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from jules_output_analysis.data import get_1d_to_2d_indices, n96e_lats, n96e_lons
from jules_output_analysis.utils import convert_longitudes
from sklearn.metrics import r2_score
from wildfires.data import ERA5_DryDayPeriod
from wildfires.utils import match_shape

from python_inferno.data import load_data
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import monthly_average_data, tqdm, unpack_wrapped

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


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
        obs_fuel_build_up_1d,
        jules_ba_gb,
        jules_time_coord,
    ) = load_data(N=None)

    # Upon loading of cached data (but not when the original calculation is being
    # done), coordinate points will have masks. Remove these here.
    if np.ma.isMaskedArray(jules_lats.points):
        jules_lats.points = jules_lats.points.data
    if np.ma.isMaskedArray(jules_lons.points):
        jules_lons.points = jules_lons.points.data

    data_time_coord = jules_ba_gb.coord("time")

    indices_1d_to_2d = get_1d_to_2d_indices(
        jules_lats.points[0],
        convert_longitudes(jules_lons.points[0]),
        n96e_lats,
        n96e_lons,
    )

    # Load observed monthly DD.
    era_dd_dataset = ERA5_DryDayPeriod()
    era_dd_dataset.limit_months(
        data_time_coord.cell(0).point, data_time_coord.cell(-1).point
    )
    era_dd_dataset.regrid(
        new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True
    )
    era_dd = era_dd_dataset.cube
    if len(era_dd.shape) == 2:
        # Only a single month was selected, causing the 'time' coordinate to be
        # squeezed.
        _meta = era_dd.metadata
        era_dd = iris.cube.Cube(
            # Restore the squeezed leading time coordinate.
            era_dd.data[np.newaxis],
            dim_coords_and_dims=[
                (era_dd.coord("time"), 0),
                (era_dd.coord("latitude"), 1),
                (era_dd.coord("longitude"), 2),
            ],
        )
        era_dd.metadata = _meta

    mon_era_dd_1d = np.ma.vstack(
        [data[indices_1d_to_2d][np.newaxis] for data in era_dd.data]
    )

    calc_mask = obs_fuel_build_up_1d.mask.any(axis=0)

    mon_era_dd_1d = np.ma.MaskedArray(
        mon_era_dd_1d, mask=match_shape(calc_mask, mon_era_dd_1d.shape)
    )
    results = {}
    for threshold in tqdm(np.linspace(0.5, 2, 30)):
        inferno_dry_days = unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold, timestep=4 * 60 * 60
        )
        inferno_dry_days = np.ma.MaskedArray(
            inferno_dry_days, mask=match_shape(calc_mask, inferno_dry_days.shape)
        )

        mon_avg_inferno_dry_days = monthly_average_data(
            inferno_dry_days, time_coord=jules_time_coord
        )

        if jules_time_coord.cell(-1).point.day == 1 and jules_time_coord.shape[0] > 1:
            # Ignore the last month if there is only a single day in it.
            mon_era_dd_1d = mon_era_dd_1d[: era_dd.shape[0] - 1]
            mon_avg_inferno_dry_days = mon_avg_inferno_dry_days[: era_dd.shape[0] - 1]

        shared_mask = np.ma.getmaskarray(mon_era_dd_1d) | np.ma.getmaskarray(
            mon_avg_inferno_dry_days
        )
        r2 = r2_score(
            y_true=np.ma.getdata(mon_era_dd_1d)[~shared_mask],
            y_pred=np.ma.getdata(mon_avg_inferno_dry_days)[~shared_mask],
        )
        results[threshold] = r2

    for threshold, r2 in results.items():
        print("threshold:", threshold, "R2:", r2)

    plt.figure()
    plt.plot(results.keys(), results.values())
    plt.xlabel("threshold")
    plt.ylabel("R2")
    plt.grid()
