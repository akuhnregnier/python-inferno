#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import pickle
import sys
from enum import Enum
from itertools import product
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from wildfires.analysis import cube_plotting

from python_inferno.ba_model import get_pred_ba
from python_inferno.data import load_jules_lats_lons
from python_inferno.utils import memoize

NoVal = Enum("NoVal", ["NoVal"])


def check_params(params, key, value=NoVal.NoVal):
    if all(key in p for p in params):
        if value is not NoVal.NoVal:
            if all(p[key] == value for p in params):
                return True
        else:
            return True
    return False


def lin_cube_plotting(*, data, exp_name):
    cube_plotting(
        data,
        title=exp_name,
        nbins=9,
        vmin_vmax_percentiles=(5, 95),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def log_cube_plotting(*, data, exp_name, raw_data):
    cube_plotting(
        data,
        title=exp_name,
        boundaries=np.geomspace(*np.quantile(raw_data[raw_data > 0], [0.05, 0.95]), 8),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    save_dir = Path("~/tmp/ba-model-analysis/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record_bak"
    assert record_dir.is_dir()

    # To prevent memory accumulation during repeated calculations below.
    memoize.active = False

    global_params = []
    global_losses = []

    for fname in record_dir.glob("*"):
        with fname.open("rb") as f:
            params, losses = pickle.load(f)

        if check_params(params, "dryness_method", 1):
            assert check_params(params, "dry_day_factor")
        elif check_params(params, "dryness_method", 2):
            assert check_params(params, "dry_bal_factor")
        else:
            raise ValueError("dryness_method")

        if check_params(params, "fuel_build_up_method", 1):
            assert check_params(params, "fuel_build_up_factor")
        elif check_params(params, "fuel_build_up_method", 2):
            assert check_params(params, "litter_pool_factor")
        else:
            raise ValueError("fuel_build_up_method")

        if check_params(params, "include_temperature", 1):
            assert check_params(params, "temperature_factor")
        elif check_params(params, "include_temperature", 0):
            assert not check_params(params, "temperature_factor")
        else:
            raise ValueError("include_temperature")

        for ps, loss in zip(params, losses):
            if loss > 0.95:
                # Skip poor samples.
                continue

            global_params.append(ps)
            global_losses.append(loss)

    df = pd.DataFrame(global_params)
    df["loss"] = global_losses

    cat_names = ["dryness_method", "fuel_build_up_method", "include_temperature"]

    for name in cat_names:
        df[name] = df[name].astype("int")

    print(df.head())
    print(df.groupby(cat_names).size())

    hist_bins = 50

    for dryness_method, fuel_build_up_method in product([1, 2], [1, 2]):
        sel = (df["dryness_method"] == dryness_method) & (
            df["fuel_build_up_method"] == fuel_build_up_method
        )
        if not np.any(sel):
            continue

        dryness_descr = {1: "Dry Day", 2: "VPD & Precip"}
        fuel_descr = {1: "Antec NPP", 2: "Leaf Litter Pool"}

        exp_name = f"Dry:{dryness_descr[dryness_method]}, Fuel:{fuel_descr[fuel_build_up_method]}"
        print(exp_name)

        dryness_keys = {1: "Dry_Day", 2: "VPD_Precip"}
        fuel_keys = {1: "Antec_NPP", 2: "Leaf_Litter_Pool"}

        exp_key = f"dry_{dryness_keys[dryness_method]}__fuel_{fuel_keys[fuel_build_up_method]}"
        print(exp_key)

        hist_save_dir = save_dir / exp_key
        hist_save_dir.mkdir(exist_ok=True, parents=False)

        df_sel = df[sel]
        min_index = df_sel["loss"].argmin()
        min_loss = df_sel.iloc[min_index]["loss"]

        for col in [col for col in df_sel.columns if col != "loss"]:
            if df_sel[col].isna().all():
                continue

            plt.figure()
            plt.plot(df_sel[col], df_sel["loss"], linestyle="", marker="o", alpha=0.6)
            plt.xlabel(col)
            plt.ylabel("loss")
            plt.title(exp_name)
            if col in ("rain_f", "vpd_f", "litter_tc", "leaf_f"):
                plt.xscale("log")
            plt.savefig(hist_save_dir / f"{col}.png")
            plt.close()

        params = {
            key: val
            for key, val in df_sel.iloc[min_index].to_dict().items()
            if not pd.isna(val) and key not in ("loss",)
        }
        pprint(params)

        logger.info("Predicting BA")
        model_ba, scores, mon_avg_gfed_ba_1d, calc_factors = get_pred_ba(**params)
        model_ba *= calc_factors["adj_factor"]

        gc.collect()

        model_ba_1d = get_1d_data_cube(model_ba, lats=jules_lats, lons=jules_lons)
        logger.info("Getting 2D cube")
        model_ba_2d = cube_1d_to_2d(model_ba_1d)

        gc.collect()

        raw_data = np.ma.getdata(model_ba)[~np.ma.getmaskarray(model_ba)]

        logger.info("Plotting hist")
        plt.figure()
        plt.hist(raw_data, bins=hist_bins)
        plt.yscale("log")
        plt.title(exp_name)
        plt.savefig(save_dir / f"hist_{exp_key}.png")
        plt.close()

        log_cube_plotting(data=model_ba_2d.data, exp_name=exp_name, raw_data=raw_data)
        plt.savefig(save_dir / f"BA_map_{exp_key}.png")
        plt.close()

        lin_cube_plotting(data=model_ba_2d.data, exp_name=exp_name)
        plt.savefig(save_dir / f"BA_map_lin_{exp_key}.png")
        plt.close()

        lin_cube_plotting(
            data=calc_factors["arcsinh_adj_factor"]
            * np.arcsinh(calc_factors["arcsinh_factor"] * model_ba_2d.data),
            exp_name=exp_name,
        )
        plt.savefig(save_dir / f"BA_map_arcsinh_{exp_key}.png")
        plt.close()

        gc.collect()

    raw_data = np.ma.getdata(mon_avg_gfed_ba_1d)[
        ~np.ma.getmaskarray(mon_avg_gfed_ba_1d)
    ]

    plt.figure()
    plt.hist(
        raw_data,
        bins=hist_bins,
    )
    plt.yscale("log")
    plt.title("GFED4")
    plt.savefig(save_dir / "hist_GFED4.png")

    GFED_2d = cube_1d_to_2d(
        get_1d_data_cube(mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
    ).data

    log_cube_plotting(
        data=GFED_2d,
        exp_name="GFED4",
        raw_data=raw_data,
    )
    plt.savefig(save_dir / "BA_map_GFED4.png")
    plt.close()

    lin_cube_plotting(
        data=GFED_2d,
        exp_name="GFED4",
    )
    plt.savefig(save_dir / "BA_map_lin_GFED4.png")
    plt.close()

    lin_cube_plotting(
        # NOTE: Assuming arcsinh_factor is constant across all experiments, which
        # should be true.
        data=np.arcsinh(calc_factors["arcsinh_factor"] * GFED_2d),
        exp_name="GFED4",
    )
    plt.savefig(save_dir / "BA_map_arcsinh_GFED4.png")
    plt.close()
