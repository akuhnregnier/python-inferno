# -*- coding: utf-8 -*-
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from numba import njit
from wildfires.analysis import cube_plotting

from python_inferno import inferno_io
from python_inferno.configuration import land_pts
from python_inferno.data import load_data

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


def plot_comparison(jules_ba_gb, python_ba_gb, obs_ba, label="BA", title=""):
    # Compare to the values calculated within JULES.
    jules_ba_2d = cube_1d_to_2d(jules_ba_gb)

    # Set up the plots.
    fig, axes = plt.subplots(
        2, 3, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(13.4, 6.9)
    )
    axes[1, 2].axis("off")

    common_kwargs = dict(
        # colorbar_kwargs=dict(label=label),
        colorbar_kwargs=False,
        title=title,
        log=True,
        nbins=6,
    )

    def normalise_cube(cube):
        return cube / np.max(cube.data)

    ax = axes[0, 0]
    cube_plotting(
        normalise_cube(jules_ba_2d),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("JULES BA")

    ax = axes[0, 1]
    cube_plotting(
        normalise_cube(cube_1d_to_2d(python_ba_gb["normal"])),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("Python BA")

    ax = axes[0, 2]
    try:
        cube_plotting(
            normalise_cube(cube_1d_to_2d(python_ba_gb["new"])),
            ax=ax,
            **common_kwargs,
        )
    except:
        print("Could not plot 'new'.")
    ax.set_title("Python BA with new Flamm.")

    ax = axes[1, 0]
    cube_plotting(
        normalise_cube(cube_1d_to_2d(python_ba_gb["new_obs_fapar"])),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("Python BA with new Flamm. & Obs FAPAR")

    ax = axes[1, 1]
    cube_plotting(
        normalise_cube(cube_1d_to_2d(obs_ba)),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("Obs BA")

    fig.subplots_adjust(wspace=0.045, hspace=-0.2)


@njit(parallel=True, nogil=True, cache=True)
def multi_timestep_inferno(
    *,
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
    pop_den,
    flash_rate,
    ignition_method,
    fuel_build_up,
    fapar_diag_pft,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    flammability_method,
):
    # Ensure consistency of the time dimension.
    if not (
        t1p5m_tile.shape[0]
        == q1p5m_tile.shape[0]
        == pstar.shape[0]
        == sthu_soilt.shape[0]
        == frac.shape[0]
        == c_soil_dpm_gb.shape[0]
        == c_soil_rpm_gb.shape[0]
        == canht.shape[0]
        == ls_rain.shape[0]
        == con_rain.shape[0]
        == fuel_build_up.shape[0]
        == fapar_diag_pft.shape[0]
    ):
        raise ValueError("All arrays need to have the same time dimension.")

    # Store the output BA (averaged over PFTs).
    ba = np.zeros_like(pstar)

    land_pts_dummy = np.zeros((land_pts,)) - 1

    for ti in range(fapar_diag_pft.shape[0]):
        # Retrieve the individual time slices.
        ba[ti] = inferno_io(
            t1p5m_tile=t1p5m_tile[ti],
            q1p5m_tile=q1p5m_tile[ti],
            pstar=pstar[ti],
            sthu_soilt=sthu_soilt[ti],
            frac=frac[ti],
            c_soil_dpm_gb=c_soil_dpm_gb[ti],
            c_soil_rpm_gb=c_soil_rpm_gb[ti],
            canht=canht[ti],
            ls_rain=ls_rain[ti],
            con_rain=con_rain[ti],
            # Not used for ignition mode 1.
            pop_den=land_pts_dummy,
            flash_rate=land_pts_dummy,
            ignition_method=ignition_method,
            fuel_build_up=fuel_build_up[ti],
            fapar_diag_pft=fapar_diag_pft[ti],
            fapar_factor=fapar_factor,
            fapar_centre=fapar_centre,
            fuel_build_up_factor=fuel_build_up_factor,
            fuel_build_up_centre=fuel_build_up_centre,
            temperature_factor=temperature_factor,
            temperature_centre=temperature_centre,
            flammability_method=flammability_method,
        )[0]
    return ba


def run_inferno(
    *, jules_lats, jules_lons, obs_fapar_1d, obs_fuel_build_up_1d, **inferno_kwargs
):
    # NOTE this function does not consider masking.
    python_ba_gb = {
        "normal": multi_timestep_inferno(
            **inferno_kwargs,
            # 1 - old, 2 - new flammability calculation
            flammability_method=1,
        ),
        "new": multi_timestep_inferno(
            **inferno_kwargs,
            # 1 - old, 2 - new flammability calculation
            flammability_method=2,
        ),
    }

    inferno_kwargs["fuel_build_up"] = np.repeat(
        np.expand_dims(obs_fuel_build_up_1d, 1), repeats=13, axis=1
    )
    inferno_kwargs["fapar_diag_pft"] = np.repeat(
        np.expand_dims(obs_fapar_1d, 1), repeats=13, axis=1
    )

    python_ba_gb["new_obs_fapar"] = multi_timestep_inferno(
        **inferno_kwargs,
        # 1 - old, 2 - new flammability calculation
        flammability_method=2,
    )

    # Transform the data to cubes.
    python_ba_gb = {
        key: get_1d_data_cube(data, lats=jules_lats, lons=jules_lons)
        for key, data in python_ba_gb.items()
    }

    return python_ba_gb


def main():
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
    ) = load_data(N=50)

    # Define the ignition method (`ignition_method`).
    ignition_method = 1

    python_ba_gb = run_inferno(
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
        # Not used for ignition mode 1.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
        ignition_method=ignition_method,
        fuel_build_up=fuel_build_up,
        fapar_diag_pft=fapar_diag_pft,
        fapar_factor=-20,
        fapar_centre=0.4,
        fuel_build_up_factor=-2,
        fuel_build_up_centre=0.4,
        temperature_factor=0.15,
        temperature_centre=300,
        jules_lats=jules_lats,
        jules_lons=jules_lons,
        obs_fapar_1d=obs_fapar_1d.data,
        obs_fuel_build_up_1d=obs_fuel_build_up_1d.data,
    )

    def average_cube(cube):
        cube.data = np.ma.MaskedArray(cube.data, mask=obs_fuel_build_up_1d.mask)
        return cube[0].copy(data=np.mean(cube.data, axis=0))

    # Average the data.
    avg_python_ba_gb = {key: average_cube(cube) for key, cube in python_ba_gb.items()}

    # Comparison plotting of mean BA.
    plot_comparison(
        jules_ba_gb=jules_ba_gb,
        python_ba_gb=avg_python_ba_gb,
        obs_ba=get_1d_data_cube(
            np.mean(gfed_ba_1d, axis=0), lats=jules_lats, lons=jules_lons
        ),
    )
    plt.show()


if __name__ == "__main__":
    main()
