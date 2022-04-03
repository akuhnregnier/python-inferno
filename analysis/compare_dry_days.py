#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from wildfires.analysis import cube_plotting
from wildfires.utils import get_unmasked

from python_inferno.analysis.dry_day_analysis import dry_day_calc, prepare_data

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


def plot_dry_days_comp(*, inferno_dry_days, mon_era_dd_1d, jules_lats, jules_lons):
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
        ls_rain,
        con_rain,
        calc_mask,
        jules_time_coord,
        mon_era_dd_1d,
        era_dd,
        jules_lats,
        jules_lons,
    ) = prepare_data()

    threshold = 1.0

    mon_avg_inferno_dry_days, mon_era_dd_1d = dry_day_calc(
        ls_rain=ls_rain,
        con_rain=con_rain,
        calc_mask=calc_mask,
        jules_time_coord=jules_time_coord,
        mon_era_dd_1d=mon_era_dd_1d,
        threshold=threshold,
    )

    plt.figure()
    plt.hexbin(
        get_unmasked(mon_era_dd_1d),
        get_unmasked(mon_avg_inferno_dry_days),
        bins="log",
    )
    plt.xlabel("OBS DD")
    plt.ylabel("INFERNO DD")
    plt.colorbar()

    plt.figure()
    plt.plot(
        get_unmasked(mon_era_dd_1d),
        get_unmasked(mon_avg_inferno_dry_days),
        linestyle="",
        marker="o",
        alpha=0.1,
    )
    plt.xlabel("OBS DD")
    plt.ylabel("INFERNO DD")

    plot_dry_days_comp(
        inferno_dry_days=mon_avg_inferno_dry_days,
        mon_era_dd_1d=mon_era_dd_1d,
        jules_lats=jules_lats,
        jules_lons=jules_lons,
    )
