# -*- coding: utf-8 -*-
from functools import reduce
from operator import add

import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import cube_1d_to_2d
from jules_output_analysis.utils import PFTs, pft_acronyms, pft_names
from wildfires.analysis import cube_plotting

from python_inferno.data import load_data

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

    # Define the ignition method (`ignition_method`).
    ignition_method = 1

    cube = iris.cube.Cube(np.expand_dims(np.mean(frac, axis=0), 1))
    cube.add_aux_coord(jules_lats, data_dims=(1, 2))
    cube.add_aux_coord(jules_lons, data_dims=(1, 2))
    mean_frac_2d = cube_1d_to_2d(cube[:, 0])

    plt.ioff()

    fig, axes = plt.subplots(
        4, 5, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(16, 12)
    )
    for i, ax in enumerate(axes.ravel()[: mean_frac_2d.shape[0]]):
        cube_plotting(
            cube=mean_frac_2d[i],
            ax=ax,
            title="",
            vmin=0,
            vmax=1.0,
            colorbar_kwargs=(
                dict(
                    ax=axes.ravel(),
                    cax=fig.add_axes([0.35, 0.07, 0.3, 0.014]),
                    orientation="horizontal",
                    label="frac",
                )
                if i == 16
                else False
            ),
        )
        ax.set_title(
            f"{pft_names[PFTs.VEG13_ALL][i]} ({pft_acronyms[PFTs.VEG13_ALL][i]})"
        )

    for ax in axes.ravel()[mean_frac_2d.shape[0] :]:
        ax.axis("off")

    fig.subplots_adjust(bottom=0.07, wspace=0.03, hspace=-0.4)

    # plt.show()

    # Separate the PFTs into groups
    pft_groups = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11, 12), (13, 14, 15, 16))

    group_cubes = []
    for pft_group in pft_groups:
        group_cubes.append(reduce(add, (mean_frac_2d[i] for i in pft_group)))

    # Plot the summed groups
    fig, axes = plt.subplots(
        1, len(pft_groups), subplot_kw=dict(projection=ccrs.Robinson()), figsize=(15, 6)
    )
    for ax, group_cube, pft_group in zip(axes.ravel(), group_cubes, pft_groups):
        cube_plotting(
            group_cube,
            ax=ax,
            title="",
            vmin=0,
            vmax=1.0,
            colorbar_kwargs=(
                dict(
                    ax=axes.ravel(),
                    cax=fig.add_axes([0.35, 0.07, 0.3, 0.014]),
                    orientation="horizontal",
                    label="frac",
                )
                if i == len(group_cubes) - 1
                else False
            ),
        )
        ax.set_title("+".join(pft_acronyms[PFTs.VEG13_ALL][i] for i in pft_group))

    total_group_frac = group_cubes[0] + group_cubes[1]
    stacked = np.vstack(
        (group_cubes[0].data[np.newaxis], group_cubes[1].data[np.newaxis])
    )
    dominant = np.ma.MaskedArray(
        np.argmax(stacked, axis=0), mask=np.sum(stacked, axis=0) < 0.1
    )

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Robinson()))
    _, cbar = cube_plotting(
        dominant,
        ax=ax,
        title="Dominant PFT group",
        colorbar_kwargs=dict(label="PFT Group"),
        return_cbar=True,
        boundaries=np.arange(3, dtype=np.float64) - 0.5,
    )
    cbar.set_ticks(np.arange(3, dtype=np.float64))
    cbar.set_ticklabels(
        [
            "+".join(pft_acronyms[PFTs.VEG13_ALL][i] for i in pft_groups[0]),
            "+".join(pft_acronyms[PFTs.VEG13_ALL][i] for i in pft_groups[1]),
        ]
    )

    plt.show()
