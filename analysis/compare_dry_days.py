# -*- coding: utf-8 -*-
import os
from pathlib import Path

import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from jules_output_analysis.data import (
    cube_1d_to_2d,
    get_1d_data_cube,
    get_1d_to_2d_indices,
    n96e_lats,
    n96e_lons,
)
from jules_output_analysis.utils import convert_longitudes
from wildfires.analysis import cube_plotting
from wildfires.data import ERA5_DryDayPeriod
from wildfires.utils import get_unmasked, match_shape

from python_inferno.data import load_data
from python_inferno.precip_dry_day import calculate_inferno_dry_days

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


def plot_dry_days_comp(inferno_dry_days, mon_era_dd_1d):
    fig, axes = plt.subplots(
        1, 2, subplot_kw=dict(projection=ccrs.Robinson()), squeeze=False
    )

    def _cube_plotting(cube, log=True, boundaries=np.geomspace(1e-6, 1, 12), **kwargs):
        cube_plotting(
            cube / np.max(cube.data), **kwargs, log=log, boundaries=boundaries
        )

    _cube_plotting(
        cube_1d_to_2d(
            get_1d_data_cube(
                np.mean(inferno_dry_days, axis=0), lats=jules_lats, lons=jules_lons
            )
        ),
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("INFERNO dry days")

    _cube_plotting(
        cube_1d_to_2d(
            get_1d_data_cube(
                np.mean(mon_era_dd_1d, axis=0), lats=jules_lats, lons=jules_lons
            )
        ),
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Obs dry days")


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

    # mean_era_dd_1d = np.mean(mon_era_dd_1d, axis=0)

    # y_true = np.ma.getdata(mean_era_dd_1d[~calc_mask])
    # y_true /= np.mean(y_true)

    # scores = {}

    # for threshold in tqdm(np.linspace(1e-6, 1e-3, 10)):
    #     inferno_dry_days = calculate_inferno_dry_days(ls_rain, con_rain, threshold)
    #     inferno_dry_days = np.ma.MaskedArray(
    #         inferno_dry_days, mask=match_shape(calc_mask, inferno_dry_days.shape)
    #     )

    #     # Compute R2 score after normalising each by their mean.
    #     y_pred = np.ma.getdata(np.mean(inferno_dry_days, axis=0)[~calc_mask])
    #     y_pred /= np.mean(y_pred)

    #     r2 = r2_score(y_true=y_true, y_pred=y_pred)

    #     scores[threshold] = r2

    # plt.figure()
    # plt.plot(scores.keys(), scores.values())

    threshold = 4.3e-5

    inferno_dry_days = calculate_inferno_dry_days(ls_rain, con_rain, threshold)
    inferno_dry_days = np.ma.MaskedArray(
        inferno_dry_days, mask=match_shape(calc_mask, inferno_dry_days.shape)
    )

    plt.figure()
    plt.hexbin(
        get_unmasked(np.mean(mon_era_dd_1d, axis=0)),
        get_unmasked(np.mean(inferno_dry_days, axis=0)),
        bins="log",
    )
    plt.xlabel("OBS DD")
    plt.ylabel("INFERNO DD")
    plt.colorbar()

    plt.figure()
    plt.plot(
        get_unmasked(np.mean(mon_era_dd_1d, axis=0)),
        get_unmasked(np.mean(inferno_dry_days, axis=0)),
        linestyle="",
        marker="o",
        alpha=0.1,
    )
    plt.xlabel("OBS DD")
    plt.ylabel("INFERNO DD")

    plot_dry_days_comp(inferno_dry_days, mon_era_dd_1d)
