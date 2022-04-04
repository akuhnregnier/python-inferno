# -*- coding: utf-8 -*-
import os
from pathlib import Path

import iris
import numpy as np
from joblib import Memory
from jules_output_analysis.data import get_1d_to_2d_indices, n96e_lats, n96e_lons
from jules_output_analysis.utils import convert_longitudes
from wildfires.data import ERA5_DryDayPeriod
from wildfires.utils import match_shape

from python_inferno.data import load_data, timestep
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import monthly_average_data, unpack_wrapped

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


def prepare_data():
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
        obs_pftcrop_1d,
        jules_time_coord,
        npp_pft,
        npp_gb,
        is_climatology,
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

    print("Limit dates:", data_time_coord.cell(0).point, data_time_coord.cell(-1).point)

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

    calc_mask = obs_fapar_1d.mask.any(axis=0)

    mon_era_dd_1d = np.ma.MaskedArray(
        mon_era_dd_1d, mask=match_shape(calc_mask, mon_era_dd_1d.shape)
    )

    return (
        ls_rain,
        con_rain,
        calc_mask,
        jules_time_coord,
        mon_era_dd_1d,
        era_dd,
        jules_lats,
        jules_lons,
    )


def dry_day_calc(
    *, ls_rain, con_rain, calc_mask, jules_time_coord, mon_era_dd_1d, threshold
):
    inferno_dry_days = unpack_wrapped(calculate_inferno_dry_days)(
        ls_rain, con_rain, threshold, timestep=timestep
    )
    inferno_dry_days = np.ma.MaskedArray(
        inferno_dry_days, mask=match_shape(calc_mask, inferno_dry_days.shape)
    )

    mon_avg_inferno_dry_days = monthly_average_data(
        inferno_dry_days, time_coord=jules_time_coord, agg_name="MAX"
    )

    if jules_time_coord.cell(-1).point.day == 1 and jules_time_coord.shape[0] > 1:
        # Ignore the last month if there is only a single day in it.
        mon_era_dd_1d = mon_era_dd_1d[:-1]
        mon_avg_inferno_dry_days = mon_avg_inferno_dry_days[:-1]

    return mon_avg_inferno_dry_days, mon_era_dd_1d