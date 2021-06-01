# -*- coding: utf-8 -*-
import os
from collections import OrderedDict, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


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
        # fapar_factor=np.linspace(-50, -30, 3),
        # fapar_centre=np.linspace(0.25, 0.4, 3),
        # fuel_build_up_factor=np.linspace(10, 30, 3),
        # fuel_build_up_centre=np.linspace(0.3, 0.45, 3),
        # temperature_factor=np.linspace(0.08, 0.18, 3),
        # dry_day_factor=np.linspace(0.02, 0.08, 3),
        # dry_day_centre=np.linspace(100, 400, 3),
        rain_f=np.linspace(0.1e-2, 5e-2, 3),
        vpd_f=np.geomspace(0.2e2, 5e2, 3),
        dry_bal_factor=-np.geomspace(0.5e-1, 0.5e1, 4),
        dry_bal_centre=np.linspace(-0.9, 0.9, 4),
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
        dryness_method=2,
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=2.0e-2,
        dry_day_centre=1.73e2,
        rain_f=1,
        vpd_f=5e5,
        dry_bal_factor=1,
        dry_bal_centre=0,
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
    ) = load_data(N=100)

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
