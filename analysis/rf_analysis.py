#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alepython.ale import ale_plot
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from python_inferno.ba_model import get_pred_ba_prep
from python_inferno.cache import cache
from python_inferno.configuration import N_pft_groups, pft_groups
from python_inferno.utils import get_grouped_average


def frac_weighted_mean(*, data, frac):
    assert len(data.shape) == 3, "Need time, PFT, and space coords."
    assert frac.shape[1] == 17

    if data.shape[1] in (13, 17):
        frac = frac[:, : data.shape[1]]
    elif data.shape[1] == N_pft_groups:
        # Grouped averaging.
        grouped_frac_shape = list(frac.shape)
        grouped_frac_shape[1] = N_pft_groups
        grouped_frac = np.zeros(tuple(grouped_frac_shape))
        for group_i in range(N_pft_groups):
            for pft_i in pft_groups[group_i]:
                grouped_frac[:, group_i] += frac[:, pft_i]
        frac = grouped_frac
    else:
        raise ValueError(f"Unsupported shape '{data.shape}'.")
    frac_sum = np.sum(frac, axis=1)
    return np.ma.MaskedArray(
        np.sum(data * frac, axis=1) / frac_sum, mask=np.isclose(frac_sum, 0)
    )


@cache
def fit_rf(df_X, y):
    rf = RandomForestRegressor(random_state=0, n_jobs=-1, oob_score=True, max_depth=10)
    rf.fit(df_X, y)
    return rf


if __name__ == "__main__":
    params = {
        "average_samples": 100.0,
        "crop_f": 0.4038945574017179,
        "dry_bal_centre": 0.30106082460835726,
        "dry_bal_centre2": -3.0,
        "dry_bal_centre3": 2.5098663316951004,
        "dry_bal_factor": -52.85520422295671,
        "dry_bal_factor2": -3.040809599885577,
        "dry_bal_factor3": -31.162196180381017,
        "dry_bal_shape": 10.749350826306456,
        "dry_bal_shape2": 10.144661753704803,
        "dry_bal_shape3": 3.0443562978997987,
        "dryness_method": 2,
        "fapar_centre": 0.9142234198681213,
        "fapar_centre2": 0.816403524795134,
        "fapar_centre3": 1.0213279256026366,
        "fapar_factor": -37.21337521309709,
        "fapar_factor2": -5.7354882396827165,
        "fapar_factor3": -49.27145943892189,
        "fapar_shape": 3.773236885339327,
        "fapar_shape2": 16.983958534020456,
        "fapar_shape3": 5.431870434265425,
        "fuel_build_up_method": 2,
        "include_temperature": 1,
        "leaf_f": 0.0001,
        "leaf_f2": 0.00055,
        "leaf_f3": 0.00055,
        "litter_pool_centre": 1759.0279343903185,
        "litter_pool_centre2": 4781.938547415133,
        "litter_pool_centre3": 2749.253956720885,
        "litter_pool_factor": 0.07341569206379095,
        "litter_pool_factor2": 0.07785426046627182,
        "litter_pool_factor3": 0.08524565837916545,
        "litter_pool_shape": 16.51963919378482,
        "litter_pool_shape2": 13.681358419141965,
        "litter_pool_shape3": 16.42018115081201,
        "litter_tc": 5.5e-10,
        "litter_tc2": 1e-10,
        "litter_tc3": 5.5e-10,
        "rain_f": 0.35,
        "rain_f2": 0.35,
        "rain_f3": 0.35,
        "temperature_centre": 312.7548225034614,
        "temperature_centre2": 307.16451042866356,
        "temperature_centre3": 294.66243804325495,
        "temperature_factor": 0.2547997933238464,
        "temperature_factor2": 0.2856463102881636,
        "temperature_factor3": 0.2580687913455504,
        "temperature_shape": 17.759922283366016,
        "temperature_shape2": 4.905154864425698,
        "temperature_shape3": 17.558341492025246,
        "vpd_f": 50.0,
        "vpd_f2": 200.0,
        "vpd_f3": 125.0,
    }

    (
        model_ba,
        data_params,
        obs_pftcrop_1d,
        jules_time_coord,
        mon_avg_gfed_ba_1d,
        data_dict,
    ) = get_pred_ba_prep(**params)

    for key, data in data_dict.items():
        print(key, data.shape)

    proc_data_dict = {}
    frac = data_dict["frac"]
    for key, data in data_dict.items():
        if key in ("t1p5m_tile", "q1p5m_tile"):
            proc_data_dict[key] = frac_weighted_mean(data=data, frac=frac)
        elif key == "sthu_soilt":
            proc_data_dict[key] = data[:, 0, 0]
        elif key in ("frac", "fapar_diag_pft", "litter_pool"):
            proc_data_dict[key] = get_grouped_average(data[:, :13])
        elif key in ("grouped_dry_bal", "con_rain"):
            # Leave as-is.
            proc_data_dict[key] = data
        elif key in (
            "canht",
            "ls_rain",
            "pstar",
            "c_soil_dpm_gb",
            "c_soil_rpm_gb",
            "fuel_build_up",
            "dry_days",
        ):
            # TODO Compute 'inferno rain' from ls_rain and con_rain.
            # Ignore.
            pass
        else:
            raise ValueError(key)

    df_X_data = {}
    for name, data in proc_data_dict.items():
        if data.ndim == 3:
            # One variable for each PFT group.
            for pft_i in range(data.shape[1]):
                df_X_data[f"{name}_{pft_i}"] = data[:, pft_i].ravel()
        else:
            df_X_data[name] = data.ravel()

    for key, data in df_X_data.items():
        print(key, data.shape)

    df_X = pd.DataFrame(df_X_data)
    y = model_ba.ravel()

    logger.info("Fitting rf.")
    rf = fit_rf(df_X, y)
    logger.info("Done fitting rf.")

    ale_dir = Path("~/tmp/ba-model-rf-ale-plots").expanduser()
    ale_dir.mkdir(exist_ok=True, parents=False)

    for column in tqdm(df_X.columns, desc="ALE plotting"):
        fig = plt.figure()
        ale_plot(rf, df_X, column, bins=15, fig=fig)
        fig.savefig(ale_dir / f"{column}.png")
        plt.close(fig)
