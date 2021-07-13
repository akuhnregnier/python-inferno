#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path
from pprint import pprint

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alepython import ale_plot
from alepython.multi_ale_plot_1d import multi_ale_plot_1d
from joblib import Memory
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm
from wildfires.analysis import cube_plotting

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.metrics import loghist, mpd, nme, nmse
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days, filter_rain
from python_inferno.utils import (
    calculate_factor,
    core_unpack_wrapped,
    expand_pft_params,
    exponential_average,
    monthly_average_data,
    temporal_nearest_neighbour_interp,
    unpack_wrapped,
)

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)
mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")

dryness_method = 2

figure_dir = Path(f"~/tmp/python_inferno-dryness-{dryness_method}").expanduser()
figure_dir.mkdir(parents=False, exist_ok=True)


def plot_comparison(jules_ba_gb, python_ba_gb, obs_ba, label="BA", title=""):
    def normalise_cube(cube):
        return cube / (np.mean(cube.data) * 18)

    # Compare to the values calculated within JULES.
    jules_ba_2d = cube_1d_to_2d(jules_ba_gb)

    common_kwargs = dict(
        # colorbar_kwargs=dict(label=label),
        colorbar_kwargs=False,
        title=title,
        log=True,
        nbins=6,
        boundaries=list(np.array([1.0, 2.0, 4.0, 8.0, 13.0, 18.0]) / 18.0),
        extend="both",
        cmap="inferno_r",
    )

    def plot_cube(cube, ax, title, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        try:
            cube_plotting(
                normalise_cube(cube),
                ax=ax,
                **{**common_kwargs, **kwargs},
            )
            ax.set_title(title)
        except:
            print(f"Could not plot '{title}'.")

    # Set up the plots.
    fig, axes = plt.subplots(
        2, 3, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(13.4, 6.9)
    )
    axes[1, 2].axis("off")
    plot_cube(jules_ba_2d, axes[0, 0], "JULES BA")
    plot_cube(cube_1d_to_2d(python_ba_gb["normal"]), axes[0, 1], "Python BA")
    plot_cube(
        cube_1d_to_2d(python_ba_gb["new"]), axes[0, 2], "Python BA with new Flamm."
    )
    plot_cube(
        cube_1d_to_2d(python_ba_gb["new_obs_fapar"]),
        axes[1, 0],
        "Python BA with new Flamm. & Obs FAPAR",
    )
    plot_cube(
        cube_1d_to_2d(obs_ba),
        axes[1, 1],
        "Obs BA",
        {
            **common_kwargs,
            "colorbar_kwargs": dict(
                orientation="horizontal",
                ax=axes,
                cax=fig.add_axes([0.35, 0.15, 0.3, 0.015]),
                format="%0.2f",
                label="Scaled BA",
            ),
        },
    )
    fig.subplots_adjust(wspace=0.045, hspace=-0.2)
    fig.savefig(figure_dir / "mean_ba_all.png")

    # Set up the plots.
    fig, axes = plt.subplots(
        1, 3, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(13.4, 3.5)
    )
    plot_cube(jules_ba_2d, axes[0], "JULES")
    plot_cube(
        cube_1d_to_2d(python_ba_gb["new_obs_fapar"]),
        axes[1],
        "New Parametrisation (Obs FAPAR)",
    )
    plot_cube(
        cube_1d_to_2d(obs_ba),
        axes[2],
        "Observed (GFED4 BA)",
        {
            **common_kwargs,
            "colorbar_kwargs": dict(
                orientation="horizontal",
                ax=axes,
                cax=fig.add_axes([0.35, 0.15, 0.3, 0.015]),
                format="%0.2f",
                label="Scaled BA",
            ),
        },
    )
    fig.subplots_adjust(wspace=0.045, hspace=-0.2)
    fig.savefig(figure_dir / "mean_ba.png")


def run_inferno(*, jules_lats, jules_lons, obs_fapar_1d, jules_fapar, **inferno_kwargs):
    assert "fuel_build_up_alpha" in inferno_kwargs
    alphas = inferno_kwargs.pop("fuel_build_up_alpha")
    assert "fuel_build_up_alpha" not in inferno_kwargs

    inferno_kwargs["fuel_build_up"] = np.ma.stack(
        list(
            exponential_average(
                temporal_nearest_neighbour_interp(jules_fapar, 4),
                alpha,
                repetitions=10,
            )[::4]
            for alpha in alphas
        ),
        axis=1,
    )

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

    inferno_kwargs["fuel_build_up"] = np.ma.stack(
        list(
            exponential_average(
                temporal_nearest_neighbour_interp(core_unpack_wrapped(obs_fapar_1d), 4),
                alpha,
                repetitions=10,
            )[::4]
            for alpha in alphas
        ),
        axis=1,
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
    parser = ArgumentParser()
    parser.add_argument("--rf", action="store_true", help="Run RF analysis")
    parser.add_argument("--ale", action="store_true", help="Run ALE analysis")
    args = parser.parse_args()

    if args.ale and not args.rf:
        raise ValueError("Need '--rf' for '--ale'.")

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

    dry_days = unpack_wrapped(calculate_inferno_dry_days)(
        ls_rain, con_rain, threshold=1.0, timestep=timestep
    )

    def frac_weighted_mean(data):
        assert len(data.shape) == 3, "Need time, PFT, and space coords."
        assert data.shape[1] in (13, 17)
        assert frac.shape[1] == 17

        return np.sum(data * frac[:, : data.shape[1]], axis=1) / np.sum(
            frac[:, : data.shape[1]], axis=1
        )

    dryness_method_params = {
        # dryness-method 1 parameters.
        1: dict(
            dry_day_centre=150.00510288982784,
            dry_day_factor=0.021688204169591885,
            fapar_centre=0.511507816157793,
            fapar_factor=-9.998892695052076,
            fuel_build_up_centre=0.35177320396443457,
            fuel_build_up_factor=27.997267588827928,
            temperature_centre=279.4948294244702,
            temperature_factor=0.17853260102934965,
            rain_f=0.0,
            vpd_f=0.0,
            dry_bal_centre=0.0,
            dry_bal_factor=0.0,
        ),
        # dryness-method 2 parameters.
        2: dict(
            dry_day_centre=0.0,
            dry_day_factor=0.0,
            dry_bal_centre=expand_pft_params(
                [1.3896404841601884, 2.730452647897936, -1.4585709112247074]
            ),
            dry_bal_factor=expand_pft_params(
                [-56.70565653352422, -36.04503042119403, -32.21199794731028]
            ),
            fapar_centre=expand_pft_params(
                [0.7464078027799277, 0.49018626762738304, 0.5770189464352058]
            ),
            fapar_factor=expand_pft_params(
                [-41.69145102416477, -13.195697076993625, -6.796962597913847]
            ),
            fuel_build_up_alpha=expand_pft_params(
                [0.00020685916150977676, 0.0009672063072438808, 0.0003893920236009512]
            ),
            fuel_build_up_centre=expand_pft_params(
                [0.39888656999303357, 0.4300687227645251, 0.3117490070594974]
            ),
            fuel_build_up_factor=expand_pft_params(
                [20.562727110713677, 22.262456060648425, 19.46010114847667]
            ),
            rain_f=expand_pft_params(
                [1.4210025942564055, 1.8303541497619247, 1.7560511543039836]
            ),
            temperature_centre=expand_pft_params(
                [
                    286.92619091266226,
                    284.02937270164404,
                    273.6096059477991,
                ]
            ),
            temperature_factor=expand_pft_params(
                [
                    0.1688349306259583,
                    0.14500470040901745,
                    0.12173578460973336,
                ]
            ),
            vpd_f=expand_pft_params(
                [1978.3640759923983, 1528.5684762665337, 1229.0709845323256]
            ),
        ),
    }

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
        dry_days=dry_days,
        dryness_method=dryness_method,
        jules_lats=jules_lats,
        jules_lons=jules_lons,
        obs_fapar_1d=obs_fapar_1d.data,
        timestep=timestep,
        jules_fapar=frac_weighted_mean(fapar_diag_pft),
        **dryness_method_params[dryness_method],
    )

    combined_mask = gfed_ba_1d.mask | obs_fapar_1d.mask | obs_fuel_build_up_1d.mask

    gfed_ba_1d.mask |= combined_mask

    # Ignore the last month if there is only a single day in it.
    ignore_last_month = (
        jules_time_coord.cell(-1).point.day == 1 and jules_time_coord.shape[0] > 1
    )

    def new_monthly_average_data(*args, **kwargs):
        averaged = monthly_average_data(*args, **kwargs)
        if ignore_last_month:
            return averaged[:-1]
        return averaged

    # Calculate monthly averages.
    mon_avg_gfed_ba_1d = new_monthly_average_data(
        gfed_ba_1d, time_coord=jules_time_coord
    )
    for key, val in python_ba_gb.items():
        assert isinstance(val, iris.cube.Cube)
        avg_data = new_monthly_average_data(val.data, time_coord=jules_time_coord)
        avg_cube = val[: avg_data.shape[0]].copy(data=avg_data)
        python_ba_gb[key] = avg_cube

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

            fig = plt.figure()
            plt.hist(y_true, bins=bins, label="true", alpha=0.4)
            plt.hist(y_pred, bins=bins, label="pred", alpha=0.4)
            plt.title(name)
            plt.yscale("log")
            plt.legend()
            fig.savefig(figure_dir / f"{name}_hist.png")

        return y_pred, adj_factor

    def print_metrics(name):
        print(name)

        # 1D stats
        y_pred, adj_factor = get_ypred(python_ba_gb[name], name=name)
        if np.all(np.isnan(y_pred)):
            print("All NaN!")
            return

        selection = ~np.isnan(y_pred)
        y_pred = y_pred[selection]

        print(f"R2: {r2_score(y_true=y_true[selection], y_pred=y_pred):+0.4f}")
        print(f"NME: {nme(obs=y_true[selection], pred=y_pred):+0.4f}")
        print(f"NMSE: {nmse(obs=y_true[selection], pred=y_pred):+0.4f}")
        loghist_val = loghist(
            obs=y_true[selection], pred=y_pred, edges=np.linspace(0, 0.4, 20)
        )
        print(f"loghist: {loghist_val:+0.4f}")

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

    if args.rf:
        variables = dict(
            temperature=new_monthly_average_data(
                frac_weighted_mean(t1p5m_tile), time_coord=jules_time_coord
            ),
            dpm=new_monthly_average_data(c_soil_dpm_gb, time_coord=jules_time_coord),
            rpm=new_monthly_average_data(c_soil_rpm_gb, time_coord=jules_time_coord),
            rain=new_monthly_average_data(
                unpack_wrapped(filter_rain)(ls_rain, con_rain),
                time_coord=jules_time_coord,
            ),
            dry_days=new_monthly_average_data(dry_days, time_coord=jules_time_coord),
            fuel=new_monthly_average_data(
                frac_weighted_mean(fuel_build_up), time_coord=jules_time_coord
            ),
            fapar=new_monthly_average_data(
                frac_weighted_mean(fapar_diag_pft), time_coord=jules_time_coord
            ),
            obs_fapar=new_monthly_average_data(
                obs_fapar_1d, time_coord=jules_time_coord
            ),
            obs_fuel=new_monthly_average_data(
                obs_fuel_build_up_1d, time_coord=jules_time_coord
            ),
        )

        # Include 1-month antecedent FAPAR and dry-days.

        main_mask = mon_avg_gfed_ba_1d.mask[1:] | mon_avg_gfed_ba_1d.mask[:-1]

        antec_fapar = variables["fapar"][:-1]
        antec_obs_fapar = variables["obs_fapar"][:-1]
        antec_dry_days = variables["dry_days"][:-1]

        new_variables = dict()
        for name, data in variables.items():
            new_variables[name] = data[1:]

        new_variables["antec_fapar"] = antec_fapar
        new_variables["antec_obs_fapar"] = antec_obs_fapar
        new_variables["antec_dry_days"] = antec_dry_days

        variables = new_variables
        del new_variables

        def get_valid_data(data):
            return np.ma.getdata(data)[~main_mask]

        jules_target = get_valid_data(
            new_monthly_average_data(jules_ba_gb, time_coord=jules_time_coord)[1:]
        )
        obs_target = get_valid_data(mon_avg_gfed_ba_1d[1:])
        jules_target *= np.mean(obs_target) / np.mean(jules_target)

        new_target = get_valid_data(python_ba_gb["new_obs_fapar"].data[1:])
        new_target *= np.mean(obs_target) / np.mean(new_target)

        jules_X = pd.DataFrame(
            {
                name: get_valid_data(data)
                for name, data in variables.items()
                if name
                in (
                    "temperature",
                    "dpm",
                    "rpm",
                    "dry_days",
                    "antec_dry_days",
                    "fapar",
                    "antec_fapar",
                )
            }
        )

        obs_X = pd.DataFrame(
            {
                name: get_valid_data(data)
                for name, data in variables.items()
                if name
                in (
                    "temperature",
                    "dpm",
                    "rpm",
                    "dry_days",
                    "antec_dry_days",
                    "obs_fapar",
                    "antec_obs_fapar",
                )
            }
        )

        models = defaultdict(dict)

        for train_name, train_X in zip(("jules-train", "obs-train"), (jules_X, obs_X)):
            for target_name, target_y in zip(
                ("jules-target", "obs-target", "new-target"),
                (jules_target, obs_target, new_target),
            ):
                rf = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=15,
                    random_state=0,
                    oob_score=True,
                    n_jobs=11,
                )
                rf.fit(train_X, target_y)

                models[(train_name, target_name)]["model"] = rf
                models[(train_name, target_name)]["score"] = rf.score(train_X, target_y)
                models[(train_name, target_name)]["oob_score"] = rf.oob_score_

        pprint(models)

    if args.ale:
        # ALE plots.

        for (train_name, train_X) in tqdm(
            (("jules-train", jules_X), ("obs-train", obs_X)), desc="ALE plotting train"
        ):

            for (target_name, target_y) in tqdm(
                (
                    ("jules-target", jules_target),
                    ("obs-target", obs_target),
                    ("new-target", new_target),
                ),
                desc="ALE plotting target",
            ):
                save_dir = figure_dir / f"ale_{train_name}_{target_name}"
                save_dir.mkdir(parents=False, exist_ok=True)

                for feature in train_X.columns:
                    fig, ax = plt.subplots()
                    ale_plot(
                        model=models[(train_name, target_name)]["model"],
                        train_set=train_X,
                        train_response=target_y,
                        features=feature,
                        monte_carlo=False,
                        # monte_carlo_rep=3,
                        verbose=True,
                        # show_full=True,
                        fig=fig,
                        ax=ax,
                    )
                    ax.grid()
                    fig.savefig(save_dir / f"{feature}.png")

        for combination, title in (
            (("obs-train", "new-target"), "New Parametrisation (with obs)"),
            (("obs-train", "obs-target"), "Observations"),
            (("jules-train", "jules-target"), "INFERNO"),
        ):
            train_set = obs_X
            if combination[0] == "obs-train":
                print("Using obs_X")
                train_set = obs_X
            elif combination[0] == "jules-train":
                print("Using jules_X")
                train_set = jules_X
            else:
                raise ValueError("Unknown train name")

            for feature, label in (("fapar", "FAPAR"), ("dry_days", "Dry Days")):
                if combination[0] == "obs-train" and feature == "fapar":
                    feature = f"obs_{feature}"
                fig, ax = plt.subplots(figsize=(4.5, 3.1))
                multi_ale_plot_1d(
                    model=models[combination]["model"],
                    train_set=train_set,
                    features=[f"antec_{feature}", feature],
                    fig=fig,
                    ax=ax,
                    title=title,
                    ylabel="BA",
                    xlabel=label,
                )
                ax.grid()
                fig.savefig(
                    figure_dir / f"{combination[0]}_{combination[1]}_{feature}_ales.png"
                )

    plt.show()
