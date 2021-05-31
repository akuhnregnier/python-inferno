# -*- coding: utf-8 -*-
import os
from collections import OrderedDict, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from numba import njit
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from python_inferno import inferno_io
from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.precip_dry_day import calculate_inferno_dry_days

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


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
    dry_days,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    flammability_method,
    dryness_method,
    dry_day_factor,
    dry_day_centre,
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
            dry_days=dry_days[ti],
            fapar_factor=fapar_factor,
            fapar_centre=fapar_centre,
            fuel_build_up_factor=fuel_build_up_factor,
            fuel_build_up_centre=fuel_build_up_centre,
            temperature_factor=temperature_factor,
            temperature_centre=temperature_centre,
            flammability_method=flammability_method,
            dryness_method=dryness_method,
            dry_day_factor=dry_day_factor,
            dry_day_centre=dry_day_centre,
        )[0]
    return ba


@memory.cache
def optimize_ba(
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
    obs_fuel_build_up_1d_data,
    obs_fapar_1d_data,
    combined_mask,
    gfed_ba_1d_data,
):
    scores = {}

    opt_parameters = OrderedDict(
        fapar_factor=np.linspace(-50, -30, 3),
        # fapar_centre=np.linspace(0.25, 0.4, 3),
        fuel_build_up_factor=np.linspace(10, 30, 3),
        # fuel_build_up_centre=np.linspace(0.3, 0.45, 3),
        # temperature_factor=np.linspace(0.08, 0.18, 3),
        # dry_day_factor=np.linspace(0.02, 0.08, 3),
        # dry_day_centre=np.linspace(100, 400, 3),
    )

    kwargs = dict(
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
        ignition_method=1,
        fuel_build_up=np.repeat(
            np.expand_dims(obs_fuel_build_up_1d_data, 1), repeats=13, axis=1
        ),
        fapar_diag_pft=np.repeat(
            np.expand_dims(obs_fapar_1d_data, 1), repeats=13, axis=1
        ),
        dry_days=calculate_inferno_dry_days(
            ls_rain, con_rain, threshold=4.3e-5, timestep=3600 * 4
        ),
        flammability_method=2,
        dryness_method=1,
        fapar_factor=-30.0,
        fapar_centre=0.3,
        fuel_build_up_factor=25.0,
        fuel_build_up_centre=0.4,
        temperature_factor=0.08,
        temperature_centre=300.0,
        dry_day_factor=0.05,
        dry_day_centre=400.0,
    )

    for factors in tqdm(list(product(*opt_parameters.values()))):
        for name, value in zip(opt_parameters, factors):
            kwargs[name] = value

        model_ba = multi_timestep_inferno(**kwargs)

        if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
            r2 = -1.0
        else:
            # Compute R2 score after normalising each by their mean.
            y_true = gfed_ba_1d_data[~combined_mask]
            y_pred = model_ba[~combined_mask]

            y_true /= np.mean(y_true)
            y_pred /= np.mean(y_pred)

            r2 = r2_score(y_true=y_true, y_pred=y_pred)

        # print(fapar_factor, fuel_build_up_factor, temperature_factor)
        # print(r2)

        scores[tuple(zip(opt_parameters.keys(), factors))] = r2
    return scores


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
    ) = load_data(N=None)

    combined_mask = gfed_ba_1d.mask | obs_fapar_1d.mask | obs_fuel_build_up_1d.mask

    scores = optimize_ba(
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
        obs_fuel_build_up_1d_data=obs_fuel_build_up_1d.data,
        obs_fapar_1d_data=obs_fapar_1d.data,
        combined_mask=combined_mask,
        gfed_ba_1d_data=gfed_ba_1d.data,
    )
    return scores


if __name__ == "__main__":
    scores = main()

    df_data = defaultdict(list)
    for parameter_values, r2 in scores.items():
        df_data["r2"].append(r2)
        for parameter, value in parameter_values:
            df_data[parameter].append(value)

    df = pd.DataFrame(df_data)

    for column in [col for col in df.columns if col != "r2"]:
        df.boxplot(column="r2", by=column)
