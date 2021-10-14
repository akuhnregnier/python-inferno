#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import sys
from functools import partial
from itertools import product
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from sklearn.metrics import r2_score
from wildfires.analysis import cube_plotting

from python_inferno.data import get_processed_climatological_data, load_jules_lats_lons
from python_inferno.metrics import loghist, mpd, nme, nmse
from python_inferno.optimisation import process_params, run_model
from python_inferno.utils import calculate_factor, monthly_average_data


def check_params(params, key):
    if key in params[0]:
        assert all(key in p for p in params)
        return True
    return False


def get_pred_ba(
    *,
    defaults=dict(
        rain_f=0.3,
        vpd_f=400,
        crop_f=0.5,
        fuel_build_up_n_samples=0,
        litter_tc=1e-9,
        leaf_f=1e-3,
    ),
    dryness_method=2,
    fuel_build_up_method=1,
    **opt_kwargs,
):
    (
        data_params,
        single_opt_kwargs,
        expanded_opt_kwargs,
    ) = process_params(opt_kwargs=opt_kwargs, defaults=defaults)

    (
        data_dict,
        mon_avg_gfed_ba_1d,
        jules_time_coord,
    ) = get_processed_climatological_data(
        litter_tc=data_params["litter_tc"],
        leaf_f=data_params["leaf_f"],
        n_samples_pft=data_params["n_samples_pft"],
        average_samples=data_params["average_samples"],
        rain_f=data_params["rain_f"],
        vpd_f=data_params["vpd_f"],
    )

    # Shallow copy to allow popping of the dictionary without affecting the
    # memoized copy.
    data_dict = data_dict.copy()
    # Extract variables not used further below.
    obs_pftcrop_1d = data_dict.pop("obs_pftcrop_1d")

    model_ba = run_model(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        single_opt_kwargs=single_opt_kwargs,
        expanded_opt_kwargs=expanded_opt_kwargs,
        data_dict=data_dict,
    )

    # Modify the predicted BA using the crop fraction (i.e. assume a certain
    # proportion of cropland never burns, even though this may be the case in
    # given the weather conditions).
    model_ba *= 1 - data_params["crop_f"] * obs_pftcrop_1d

    def fail_func():
        raise RuntimeError()

    if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
        return fail_func()

    # Calculate monthly averages.
    avg_ba = monthly_average_data(model_ba, time_coord=jules_time_coord)
    assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

    # Get ypred.
    y_pred = np.ma.getdata(avg_ba)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

    y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

    # Estimate the adjustment factor by minimising the NME.
    adj_factor = calculate_factor(y_true=y_true, y_pred=y_pred)

    y_pred *= adj_factor

    assert y_pred.shape == y_true.shape

    pad_func = partial(
        np.pad,
        pad_width=((0, 12 - mon_avg_gfed_ba_1d.shape[0]), (0, 0)),
        constant_values=0.0,
    )
    obs_pad = pad_func(mon_avg_gfed_ba_1d)
    # Apply adjustment factor similarly to y_pred.
    pred_pad = adj_factor * pad_func(avg_ba)
    mpd_val, ignored = mpd(obs=obs_pad, pred=pred_pad, return_ignored=True)

    if ignored > 5600:
        # Ensure that not too many samples are ignored.
        return fail_func()

    scores = dict(
        # 1D stats
        r2=r2_score(y_true=y_true, y_pred=y_pred),
        nme=nme(obs=y_true, pred=y_pred),
        nmse=nmse(obs=y_true, pred=y_pred),
        loghist=loghist(obs=y_true, pred=y_pred, edges=np.linspace(0, 0.4, 20)),
        # Temporal stats.
        mpd=mpd_val,
    )

    if any(np.ma.is_masked(val) for val in scores.values()):
        return fail_func()

    return avg_ba, scores, mon_avg_gfed_ba_1d


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record_bak"
    assert record_dir.is_dir()

    global_params = []
    global_losses = []

    for fname in record_dir.glob("*"):
        with fname.open("rb") as f:
            params, losses = pickle.load(f)
        if check_params(params, "dry_day_factor"):
            dryness_method = 1
        elif check_params(params, "dry_bal_factor"):
            dryness_method = 2
        else:
            raise ValueError()

        if check_params(params, "fuel_build_up_factor"):
            fuel_build_up_method = 1
        elif check_params(params, "litter_pool_factor"):
            fuel_build_up_method = 2
        else:
            raise ValueError()

        for ps, loss in zip(params, losses):
            if loss > 0.95:
                # Skip poor samples.
                continue

            global_params.append(
                {
                    **ps,
                    **dict(
                        dryness_method=dryness_method,
                        fuel_build_up_method=fuel_build_up_method,
                    ),
                }
            )
            global_losses.append(loss)

    df = pd.DataFrame(global_params)
    df["loss"] = global_losses
    print(df.head())

    for col in [col for col in df.columns if col != "loss"]:
        plt.figure()
        plt.plot(df[col], df["loss"], linestyle="", marker="o", alpha=0.6)
        plt.xlabel(col)
        plt.ylabel("loss")

    df["dryness_method"] = df["dryness_method"].astype("int")
    df["fuel_build_up_method"] = df["fuel_build_up_method"].astype("int")

    for dryness_method, fuel_build_up_method in product([1, 2], [1, 2]):
        sel = (df["dryness_method"] == dryness_method) & (
            df["fuel_build_up_method"] == fuel_build_up_method
        )
        df_sel = df[sel]
        min_index = df_sel["loss"].argmin()
        min_loss = df_sel.iloc[min_index]["loss"]

        dryness_descr = {1: "Dry Day", 2: "VPD & Precip"}
        fuel_descr = {1: "Antec NPP", 2: "Leaf Litter Pool"}

        exp_name = f"Dry:{dryness_descr[dryness_method]}, Fuel:{fuel_descr[fuel_build_up_method]}"
        print(exp_name)

        params = {
            key: val
            for key, val in df_sel.iloc[min_index].to_dict().items()
            if not pd.isna(val) and key not in ("loss",)
        }
        pprint(params)

        model_ba, scores, mon_avg_gfed_ba_1d = get_pred_ba(**params)
        model_ba_1d = get_1d_data_cube(model_ba, lats=jules_lats, lons=jules_lons)
        model_ba_2d = cube_1d_to_2d(model_ba_1d)
        cube_plotting(model_ba_2d.data, title=exp_name, fig=plt.figure(figsize=(12, 5)))

    cube_plotting(
        cube_1d_to_2d(
            get_1d_data_cube(mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
        ).data,
        title="GFED4",
        fig=plt.figure(figsize=(12, 5)),
    )
    plt.show()
