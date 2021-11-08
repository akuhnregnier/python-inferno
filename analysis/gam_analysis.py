#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import math
import os
import pickle
import sys
from enum import Enum
from functools import partial, reduce
from itertools import product
from operator import add
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from pygam import GAM, LogisticGAM, s
from tqdm import tqdm

from python_inferno.ba_model import Status, calculate_scores, process_params
from python_inferno.cache import cache
from python_inferno.configuration import N_pft_groups, pft_group_names, pft_groups
from python_inferno.data import get_processed_climatological_data, load_jules_lats_lons
from python_inferno.inferno import sigmoid
from python_inferno.plotting import plotting
from python_inferno.utils import memoize

NoVal = Enum("NoVal", ["NoVal"])


@cache
def fit_gam(
    *, X, y, lam, gam_type="logistic", distribution=None, link=None, N_features
):
    terms = reduce(add, (s(i) for i in range(N_features)))
    if gam_type == "logistic":
        gam_func = LogisticGAM
    elif gam_type is None:
        gam_func = partial(GAM, distribution=distribution, link=link)

    gam = gam_func(
        terms,
        verbose=True,
        # TODO Hyperopt parameter optimisation of `lam` parameter.
        lam=[213] * N_features,
    ).fit(X, y)
    return gam


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
    save_dir = Path("~/tmp/gam-model-analysis/").expanduser()
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

    all_name_map = {
        "fapar_diag_pft": "fapar",
        "dry_days": "dry_day",
        "t1p5m_tile": "temperature",
        "fuel_build_up": "fuel_build_up",
        "grouped_dry_bal": "dry_bal",
        "litter_pool": "litter_pool",
    }

    for dryness_method, fuel_build_up_method in product([1, 2], [1, 2]):
        name_map = all_name_map.copy()
        if dryness_method == 1:
            del name_map["grouped_dry_bal"]
        elif dryness_method == 2:
            del name_map["dry_days"]

        if fuel_build_up_method == 1:
            del name_map["litter_pool"]
        elif fuel_build_up_method == 2:
            del name_map["fuel_build_up"]

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

        df_sel = df[sel]
        min_index = df_sel["loss"].argmin()
        min_loss = df_sel.iloc[min_index]["loss"]

        params = {
            key: val
            for key, val in df_sel.iloc[min_index].to_dict().items()
            if not pd.isna(val) and key not in ("loss",)
        }
        pprint(params)

        del params["dryness_method"]
        del params["fuel_build_up_method"]
        del params["include_temperature"]

        (data_params, single_opt_kwargs, expanded_opt_kwargs) = process_params(
            opt_kwargs=params,
            defaults=dict(
                rain_f=0.3,
                vpd_f=400,
                crop_f=0.5,
                fuel_build_up_n_samples=0,
                litter_tc=1e-9,
                leaf_f=1e-3,
            ),
        )

        # NOTE - This will make parameters deviate from the originally-found minimum
        # (with the new INFERNO model), but is required to get 12 output months for
        # comparison with GFED4 data!
        data_params["average_samples"] = 183

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

        gc.collect()

        for key, val in data_dict.items():
            if val.shape[0] != 12:
                raise ValueError(
                    f"'{key}' data did not have 12 months, but instead: {val.shape}."
                )

        # Shallow copy to allow popping of the dictionary without affecting the
        # memoized copy.
        data_dict = data_dict.copy()
        # Extract variables not used further below.
        obs_pftcrop_1d = data_dict.pop("obs_pftcrop_1d")

        # Select single soil layer of soil moisture.
        data_dict["inferno_sm"] = data_dict.pop("sthu_soilt")[:, 0, 0]

        frac = data_dict.pop("frac")

        # Take frac-average.
        for key in data_dict:
            if len(data_dict[key].shape) == 3:
                data_dict[key] = frac_weighted_mean(data=data_dict[key], frac=frac)

        combined_mask = np.logical_or(
            np.ma.getmaskarray(mon_avg_gfed_ba_1d),
            reduce(
                np.logical_or, (np.ma.getmaskarray(data) for data in data_dict.values())
            ),
        )

        valid_data_dict = {
            key: np.ma.getdata(data)[~combined_mask]
            for key, data in data_dict.items()
            if key in name_map
        }
        for valid_data in valid_data_dict.values():
            assert not np.any(np.isnan(valid_data))

        valid_mon_avg_gfed_ba_1d = np.ma.getdata(mon_avg_gfed_ba_1d)[~combined_mask]

        X = np.zeros(((~combined_mask).sum(), len(valid_data_dict)))
        for i, valid_data in enumerate(valid_data_dict.values()):
            X[:, i] = valid_data
        y = valid_mon_avg_gfed_ba_1d

        N_features = X.shape[1]
        feature_names = list(valid_data_dict.keys())

        gam = fit_gam(
            X=X,
            y=y,
            lam=[213] * len(name_map),
            gam_type=None,
            distribution="normal",
            link="log",
            N_features=N_features,
        )

        # gam = LogisticGAM(
        #     reduce(add, (s(i) for i in range(N_features))),
        #     verbose=True,
        #     # TODO Hyperopt parameter optimisation of `lam` parameter.
        #     # lam=[213] * len(name_map),
        # # ).fit(X, y)
        # ).gridsearch(X, y, lam=[np.logspace(1.5, 3, 3)] * N_features)

        # ~30 parameters per minute
        # X parameters for 120 minutes? -> 120*20/3 -> 800 parameters

        gam.summary()

        # NOTE Unsupported (for now).
        assert not single_opt_kwargs

        name_roots = {"_".join(name.split("_")[:-1]) for name in expanded_opt_kwargs}

        fig_width = 12.8
        fig_height = 6.9

        # nrows * ncols >= N_features
        # -> nrows >= N_features / ncols
        # nrows / ncols >= fig_height / fig_width
        # nrows / ncols = x * fig_height / fig_width
        # nrows = x * ncols * fig_height / fig_width
        # -> x * ncols * fig_height / fig_width >= N_features / ncols
        # -> x * fig_height / fig_width * ncols**2 >= N_features
        nrows = math.floor((1.5 * N_features * fig_height / fig_width) ** 0.5)
        ncols = math.ceil(N_features / nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        for (ax, (i, term)) in zip(
            axes.ravel(), enumerate(term for term in gam.terms if not term.isintercept)
        ):
            sigmoid_params = {}
            for key, suffix in [
                ("factors", "_factor"),
                ("centres", "_centre"),
                ("shapes", "_shape"),
            ]:
                sigmoid_params[key] = expanded_opt_kwargs[
                    name_map[feature_names[i]] + suffix
                ]

            assert all(
                len(values) == N_pft_groups for values in sigmoid_params.values()
            )

            ax.set_title(feature_names[i])

            XX = gam.generate_X_grid(term=i)
            xs = XX[:, i]
            ax.plot(xs, gam.partial_dependence(term=i, X=XX), label="GAM")

            for (i, (factor, centre, shape)) in enumerate(
                zip(*sigmoid_params.values())
            ):
                ax.plot(
                    xs, sigmoid(xs, factor, centre, shape), label=pft_group_names[i]
                )
            ax.legend()

        for ax in axes.ravel()[N_features:]:
            ax.set_axis_off()

        fig.suptitle(exp_name)
        plt.tight_layout()

        plt.savefig(save_dir / f"{exp_key}_gam_curves.png")
        plt.close()

        pred_y_1d = np.ma.MaskedArray(
            np.zeros_like(combined_mask, dtype=np.float64), mask=True
        )
        pred_y_1d[~combined_mask] = gam.predict(X)
        mon_avg_gfed_ba_1d.mask |= combined_mask

        # TODO NOTE How to tie this in GAM analysis??
        # # Modify the predicted BA using the crop fraction (i.e. assume a certain
        # # proportion of cropland never burns, even though this may be the case in
        # # given the weather conditions).
        # model_ba *= 1 - data_params["crop_f"] * obs_pftcrop_1d

        scores, status, avg_ba, calc_factors = calculate_scores(
            model_ba=pred_y_1d,
            jules_time_coord=jules_time_coord,
            mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
        )
        assert status is Status.SUCCESS
        avg_ba *= calc_factors["adj_factor"]

        gc.collect()

        gam_ba_1d = get_1d_data_cube(avg_ba, lats=jules_lats, lons=jules_lons)
        logger.info("Getting 2D cube")
        model_ba_2d = cube_1d_to_2d(gam_ba_1d)

        gc.collect()

        plot_data[exp_name] = dict(
            exp_key=exp_key,
            raw_data=np.ma.getdata(avg_ba)[~np.ma.getmaskarray(avg_ba)],
            model_ba_2d_data=model_ba_2d.data,
            hist_bins=hist_bins,
            arcsinh_adj_factor=calc_factors["arcsinh_adj_factor"],
            arcsinh_factor=calc_factors["arcsinh_factor"],
            scores=scores,
        )
        gc.collect()

    for exp_name, data in tqdm(list(plot_data.items()), desc="Plotting"):
        plotting(exp_name=exp_name, **data)
