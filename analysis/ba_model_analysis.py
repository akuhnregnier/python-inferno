#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import pickle
import sys
from enum import Enum
from functools import partial
from itertools import product
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from tqdm import tqdm

from python_inferno.ba_model import Status, calculate_scores, get_pred_ba
from python_inferno.cache import cache
from python_inferno.data import load_data, load_jules_lats_lons
from python_inferno.metrics import null_model_analysis
from python_inferno.plotting import plotting
from python_inferno.utils import PartialDateTime, memoize, temporal_processing

NoVal = Enum("NoVal", ["NoVal"])


def frac_weighted_mean(*, data, frac):
    assert len(data.shape) == 3, "Need time, PFT, and space coords."
    assert data.shape[1] in (13, 17)
    assert frac.shape[1] == 17

    return np.sum(data * frac[:, : data.shape[1]], axis=1) / np.sum(
        frac[:, : data.shape[1]], axis=1
    )


@cache(dependencies=[load_data, temporal_processing])
def get_processed_climatological_jules_ba():
    logger.debug("start data")
    (
        _,
        _,
        _,
        _,
        frac,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        jules_ba_gb,
        _,
        jules_time_coord,
        _,
        _,
        climatology_output,
    ) = load_data(
        filenames=(
            tuple(
                [
                    str(Path(s).expanduser())
                    for s in [
                        "~/tmp/climatology6.nc",
                    ]
                ]
            )
        ),
        N=None,
        output_timesteps=4,
        climatology_dates=(PartialDateTime(2000, 1), PartialDateTime(2016, 12)),
    )
    logger.debug("Got data")

    data_dict = dict(
        frac=frac,
        jules_ba_gb=jules_ba_gb.data,
    )

    logger.debug("Populated data_dict")

    assert jules_time_coord.cell(-1).point.month == 12
    last_year = jules_time_coord.cell(-1).point.year
    for start_i in range(jules_time_coord.shape[0]):
        if jules_time_coord.cell(start_i).point.year == last_year:
            break
    else:
        raise ValueError("Target year not encountered.")

    # Trim the data and temporal coord such that the data spans a single year.
    jules_time_coord = jules_time_coord[start_i:]
    for data_name in data_dict:
        data_dict[data_name] = data_dict[data_name][start_i:]

    assert (
        jules_time_coord.cell(0).point.year == jules_time_coord.cell(-1).point.year
        and jules_time_coord.cell(0).point.month == 1
        and jules_time_coord.cell(-1).point.month == 12
        and jules_time_coord.shape[0] >= 12
    )
    return data_dict, jules_time_coord


def check_params(params, key, value=NoVal.NoVal):
    if all(key in p for p in params):
        if value is not NoVal.NoVal:
            if all(p[key] == value for p in params):
                return True
        else:
            return True
    return False


if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    save_dir = Path("~/tmp/ba-model-analysis/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)
    plotting = partial(plotting, save_dir=save_dir)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record_bak"
    assert record_dir.is_dir()

    # To prevent memory accumulation during repeated calculations below.
    memoize.active = False

    global_params = []
    global_losses = []

    for fname in tqdm(list(record_dir.glob("*")), desc="Reading opt_record files"):
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
    print("\nNumber of trials:\n")
    print(df.groupby(cat_names).size())

    print("\nMinimum loss by parametrisation approach:\n")
    print(df.groupby(cat_names)["loss"].min())

    hist_bins = 50

    plot_data = dict()
    plot_prog = tqdm(desc="Generating plot data", total=6)

    for dryness_method, fuel_build_up_method in product([1, 2], [1, 2]):
        sel = (df["dryness_method"] == dryness_method) & (
            df["fuel_build_up_method"] == fuel_build_up_method
        )
        if not np.any(sel):
            continue

        dryness_descr = {1: "Dry Day", 2: "VPD & Precip"}
        fuel_descr = {1: "Antec NPP", 2: "Leaf Litter Pool"}

        exp_name = f"Dry:{dryness_descr[dryness_method]}, Fuel:{fuel_descr[fuel_build_up_method]}"
        logger.info(exp_name)

        dryness_keys = {1: "Dry_Day", 2: "VPD_Precip"}
        fuel_keys = {1: "Antec_NPP", 2: "Leaf_Litter_Pool"}

        exp_key = f"dry_{dryness_keys[dryness_method]}__fuel_{fuel_keys[fuel_build_up_method]}"
        logger.info(exp_key)

        hist_save_dir = save_dir / "parameter_histograms" / exp_key
        hist_save_dir.mkdir(exist_ok=True, parents=True)

        df_sel = df[sel]
        min_index = df_sel["loss"].argmin()
        min_loss = df_sel.iloc[min_index]["loss"]

        logger.info("Plotting histograms.")

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

        plot_data[exp_name] = dict(
            exp_key=exp_key,
            raw_data=np.ma.getdata(model_ba)[~np.ma.getmaskarray(model_ba)],
            model_ba_2d_data=model_ba_2d.data,
            hist_bins=hist_bins,
            arcsinh_adj_factor=calc_factors["arcsinh_adj_factor"],
            arcsinh_factor=calc_factors["arcsinh_factor"],
            scores=scores,
        )
        gc.collect()
        plot_prog.update()

    # GFED4
    reference_obs = cube_1d_to_2d(
        get_1d_data_cube(mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
    ).data
    plot_data["GFED4"] = dict(
        raw_data=np.ma.getdata(mon_avg_gfed_ba_1d)[
            ~np.ma.getmaskarray(mon_avg_gfed_ba_1d)
        ],
        model_ba_2d_data=reference_obs,
        hist_bins=hist_bins,
        # NOTE: Assuming arcsinh_factor is constant across all experiments, which
        # should be true.
        arcsinh_adj_factor=1.0,
        arcsinh_factor=calc_factors["arcsinh_factor"],
    )
    plot_prog.update()

    # Old INFERNO BA.
    data_dict, jules_time_coord = get_processed_climatological_jules_ba()
    jules_ba_gb = data_dict.pop("jules_ba_gb")
    scores, status, avg_jules_ba, calc_factors = calculate_scores(
        model_ba=jules_ba_gb,
        jules_time_coord=jules_time_coord,
        mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
    )
    assert status is Status.SUCCESS, "Score calculation failed!"

    avg_jules_ba *= calc_factors["adj_factor"]

    plot_data["Old INFERNO BA"] = dict(
        raw_data=np.ma.getdata(avg_jules_ba)[~np.ma.getmaskarray(avg_jules_ba)],
        model_ba_2d_data=cube_1d_to_2d(
            get_1d_data_cube(avg_jules_ba, lats=jules_lats, lons=jules_lons)
        ).data,
        hist_bins=hist_bins,
        arcsinh_adj_factor=calc_factors["arcsinh_adj_factor"],
        arcsinh_factor=calc_factors["arcsinh_factor"],
        scores=scores,
    )
    plot_prog.update()
    plot_prog.close()

    for exp_name, data in tqdm(list(plot_data.items()), desc="Plotting"):
        plotting(
            exp_name=exp_name,
            ref_2d_data=(reference_obs if exp_name != "GFED4" else None),
            **data,
        )

    null_model_analysis(
        reference_data=reference_obs,
        comp_data={
            key: vals["model_ba_2d_data"]
            for key, vals in plot_data.items()
            if key != "GFED4"
        },
        rng=np.random.default_rng(0),
        save_dir=save_dir,
    )
