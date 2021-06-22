# -*- coding: utf-8 -*-
import os
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from sklearn.metrics import r2_score
from wildfires.analysis import cube_plotting

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.metrics import mpd, nme, nmse
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import calculate_factor, monthly_average_data, unpack_wrapped

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

    timestep = 4 * 60 * 60

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
            ls_rain, con_rain, threshold=1.0, timestep=timestep
        ),
        dry_day_centre=150.00510288982784,
        dry_day_factor=0.021688204169591885,
        dry_bal_centre=-0.9779010845656665,
        dry_bal_factor=-59.17322076688619,
        fapar_centre=0.737871300105012,
        fapar_factor=-12.408836200357614,
        fuel_build_up_centre=0.3158407150237622,
        fuel_build_up_factor=24.731279424730566,
        rain_f=1.6956186292007316,
        temperature_centre=286.3630645479963,
        temperature_factor=0.15671988683218163,
        vpd_f=823.8710509245902,
        dryness_method=2,
        jules_lats=jules_lats,
        jules_lons=jules_lons,
        obs_fapar_1d=obs_fapar_1d.data,
        obs_fuel_build_up_1d=obs_fuel_build_up_1d.data,
        timestep=timestep,
    )

    combined_mask = gfed_ba_1d.mask | obs_fapar_1d.mask | obs_fuel_build_up_1d.mask

    gfed_ba_1d.mask |= combined_mask

    # Calculate monthly averages.
    mon_avg_gfed_ba_1d = monthly_average_data(gfed_ba_1d, time_coord=jules_time_coord)
    for key, val in python_ba_gb.items():
        assert isinstance(val, iris.cube.Cube)
        avg_data = monthly_average_data(val.data, time_coord=jules_time_coord)
        avg_cube = val[: avg_data.shape[0]].copy(data=avg_data)
        python_ba_gb[key] = avg_cube

    if jules_time_coord.cell(-1).point.day == 1 and jules_time_coord.shape[0] > 1:
        # Ignore the last month if there is only a single day in it.
        mon_avg_gfed_ba_1d = mon_avg_gfed_ba_1d[:-1]
        for key, val in python_ba_gb.items():
            python_ba_gb[key] = val[:-1]
            assert python_ba_gb[key].shape == mon_avg_gfed_ba_1d.shape

    y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

    def get_ypred(cube, name=None, verbose=True):
        y_pred = np.ma.getdata(cube.data)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

        # Estimate the adjustment factor by minimising the NME.
        adj_factor = calculate_factor(y_true=y_true, y_pred=y_pred)

        y_pred *= adj_factor

        assert y_pred.shape == y_true.shape

        if name is not None and verbose:
            max_ba = max(np.max(y_true), np.max(y_pred))
            bins = np.linspace(0, max_ba, 100)

            plt.figure()
            plt.hist(y_true, bins=bins, label="true", alpha=0.4)
            plt.hist(y_pred, bins=bins, label="pred", alpha=0.4)
            plt.title(name)
            plt.yscale("log")
            plt.legend()

        return y_pred, adj_factor

    def print_metrics(name):
        print(name)

        # 1D stats
        y_pred, adj_factor = get_ypred(python_ba_gb[name], name=name)
        print(f"R2: {r2_score(y_true=y_true, y_pred=y_pred):+0.4f}")
        print(f"NME: {nme(obs=y_true, pred=y_pred):+0.4f}")
        print(f"NMSE: {nmse(obs=y_true, pred=y_pred):+0.4f}")

        # Temporal stats.
        pad_func = partial(
            np.pad,
            pad_width=((0, 12 - mon_avg_gfed_ba_1d.shape[0]), (0, 0)),
            constant_values=0.0,
        )
        obs_pad = pad_func(mon_avg_gfed_ba_1d)
        # Apply adjustment factor similarly to y_pred.
        pred_pad = adj_factor * pad_func(python_ba_gb[name].data)
        mpd_val, ignored = mpd(obs=obs_pad, pred=pred_pad, return_ignored=True)
        print(f"MPD: {mpd_val:+0.4f} (skipped: {ignored})")

    for name in ("normal", "new", "new_obs_fapar"):
        print_metrics(name)

    def average_cube(cube):
        cube.data = np.ma.MaskedArray(cube.data, mask=mon_avg_gfed_ba_1d.mask)
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
