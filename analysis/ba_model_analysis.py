#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import sys
from itertools import product
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from wildfires.analysis import cube_plotting

from python_inferno.ba_model import get_pred_ba
from python_inferno.data import load_jules_lats_lons


def check_params(params, key):
    if key in params[0]:
        assert all(key in p for p in params)
        return True
    return False


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
