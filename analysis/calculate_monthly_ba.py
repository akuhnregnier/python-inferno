# -*- coding: utf-8 -*-
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from sklearn.metrics import r2_score
from wildfires.analysis import cube_plotting

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import unpack_wrapped

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


def run_inferno(
    *, jules_lats, jules_lons, obs_fapar_1d, obs_fuel_build_up_1d, **inferno_kwargs
):
    # NOTE this function does not consider masking.
    python_ba_gb = {
        "normal": unpack_wrapped(multi_timestep_inferno)(
            **inferno_kwargs,
            # 1 - old, 2 - new flammability calculation
            flammability_method=1,
        ),
        "new": unpack_wrapped(multi_timestep_inferno)(
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

    python_ba_gb["new_obs_fapar"] = unpack_wrapped(multi_timestep_inferno)(
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
        jules_time_coord,
    ) = load_data(N=None)

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
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=2.83e-5, timestep=3600 * 4
        ),
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=2.0e-2,
        dry_day_centre=1.73e2,
        dryness_method=1,
        rain_f=0,
        vpd_f=0,
        dry_bal_centre=0,
        dry_bal_factor=0,
        jules_lats=jules_lats,
        jules_lons=jules_lons,
        obs_fapar_1d=obs_fapar_1d.data,
        obs_fuel_build_up_1d=obs_fuel_build_up_1d.data,
    )

    combined_mask = gfed_ba_1d.mask | obs_fapar_1d.mask | obs_fuel_build_up_1d.mask

    y_true = np.ma.getdata(gfed_ba_1d)[~combined_mask]
    y_true /= np.mean(y_true)

    def get_r2(data):
        y_pred = np.ma.getdata(data.data)[~combined_mask]
        y_pred /= np.mean(y_pred)
        return r2_score(y_true=y_true, y_pred=y_pred)

    print(f"JULES (python) R2: {get_r2(python_ba_gb['normal']):0.2f}")
    print(f"New flamm. R2: {get_r2(python_ba_gb['new']):0.2f}")
    print(
        f"New flamm. with obs. FAPAR R2: {get_r2(python_ba_gb['new_obs_fapar']):0.2f}"
    )

    def average_cube(cube):
        cube.data = np.ma.MaskedArray(cube.data, mask=combined_mask)
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
