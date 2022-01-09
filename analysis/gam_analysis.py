#!/mnt/small-ssd/miniconda3/envs/python-inferno/bin/python
# -*- coding: utf-8 -*-

# Use the correct R installation.

# isort: off
import os

os.environ["R_HOME"] = "/mnt/small-ssd/miniconda3/envs/python-inferno/lib/R"
# isort: on

import gc
import hashlib
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from functools import partial, reduce
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scipy.stats
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from tqdm import tqdm

from python_inferno.ba_model import Status, calculate_scores, process_params
from python_inferno.cache import cache
from python_inferno.configuration import N_pft_groups, pft_group_names, pft_groups
from python_inferno.data import get_processed_climatological_data, load_jules_lats_lons
from python_inferno.inferno import sigmoid
from python_inferno.metrics import null_model_analysis
from python_inferno.plotting import plotting
from python_inferno.utils import get_exp_key, get_exp_name, memoize

NoVal = Enum("NoVal", ["NoVal"])


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


def read_global_param_data():
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
    return global_params, global_losses


def file_hash_calc(path):
    r_gam_hash = hashlib.md5()
    r_gam_hash.update(
        open("/home/alexander/Documents/PhD/python-inferno/R/gam.R").read().encode()
    )
    return {"path": path, "value": r_gam_hash.hexdigest()}


@cache
def gam_analysis(
    gam_data_df,
    # Ensure that the function is recalculated when the R source file is changed.
    r_gam_file_data=file_hash_calc(
        Path(__file__).resolve().parent.parent / "R" / "gam.R"
    ),
):
    """GAM fitting and prediction using R.

    TODO:
        Currently there is no clear encapsulation of the R (rpy2) 'globalenv' used to
        store objects here. This could lead to issues if rpy2 is used elsewhere in the
        Python process prior to this function being called, as it may result in
        unintended side effects or differences that are not picked up by the @cache
        decorator.

    """
    ro.r["source"](str(r_gam_file_data["path"]))

    # Store partial predictions (all but one variable held fixed).
    partial_plot_vals = {}
    partial_preds = {}
    partial_pred_errs = {}

    with localconverter(ro.default_converter + pandas2ri.converter):
        # Call the sourced R GAM fitting function while automatically converting the
        # pandas df to R.
        ro.globalenv["fitted_gam"] = ro.globalenv["fit_gam"](
            exp_key,
            gam_data_df,
            str(stdout_file),
            str(save_dir),
        )
        gam_pred = ro.r('predict(fitted_gam, type="response")')
        print("Gam pred:")
        print(scipy.stats.describe(gam_pred))

        N = 100

        # NOTE Could be used to get from link to response scale.
        # inv_link = ro.r("family(fitted_gam)$linkinv")

        for feature_name in feature_names:
            feature_vals = np.linspace(
                gam_data_df[feature_name].min(), gam_data_df[feature_name].max(), N
            )
            ro.globalenv["grid_data_df"] = pd.DataFrame(
                {
                    key: feature_vals if key == feature_name else ([col.median()] * N)
                    for key, col in gam_data_df.iteritems()
                }
            )
            ro.globalenv["grid_pred"] = ro.r(
                'predict(fitted_gam, grid_data_df, type="response", se.fit=TRUE)'
            )
            partial_plot_vals[feature_name] = feature_vals
            partial_preds[feature_name] = ro.r("grid_pred$fit")
            partial_pred_errs[feature_name] = ro.r("grid_pred$se.fit")

    return gam_pred, partial_plot_vals, partial_preds, partial_pred_errs


def partial_dependence_plots(
    *,
    fig_width=12.8,
    fig_height=6.9,
    se_factor=0.1,
    feature_names,
    expanded_opt_kwargs,
    name_map,
    partial_plot_vals,
    partial_preds,
    partial_pred_errs,
    save_dir,
    exp_key,
):
    # nrows * ncols >= N_features
    # -> nrows >= N_features / ncols
    # nrows / ncols >= fig_height / fig_width
    # nrows / ncols = x * fig_height / fig_width
    # nrows = x * ncols * fig_height / fig_width
    # -> x * ncols * fig_height / fig_width >= N_features / ncols
    # -> x * fig_height / fig_width * ncols**2 >= N_features
    N_features = len(feature_names)
    nrows = math.floor((1.5 * N_features * fig_height / fig_width) ** 0.5)
    ncols = math.ceil(N_features / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    for (ax, feature_name) in zip(axes.ravel(), feature_names):
        sigmoid_params = {}
        for key, suffix in [
            ("factors", "_factor"),
            ("centres", "_centre"),
            ("shapes", "_shape"),
        ]:
            sigmoid_params[key] = expanded_opt_kwargs[name_map[feature_name] + suffix]

        assert all(len(values) == N_pft_groups for values in sigmoid_params.values())

        handles = []

        ax.set_title(name_map[feature_name])

        gam_ax = ax.twinx()

        ax.tick_params(axis="y", labelcolor="k")
        gam_ax.tick_params(axis="y", labelcolor="tab:blue")

        handles.append(
            gam_ax.plot(
                partial_plot_vals[feature_name],
                partial_preds[feature_name],
                label="GAM",
                c="tab:blue",
                zorder=3,
            )[0]
        )
        # Standard error factor `se_factor`. Should be 2, but perhaps a lower factor
        # is better for plotting.
        handles.append(
            gam_ax.fill_between(
                partial_plot_vals[feature_name],
                partial_preds[feature_name]
                - se_factor * partial_pred_errs[feature_name],
                partial_preds[feature_name]
                + se_factor * partial_pred_errs[feature_name],
                alpha=0.2,
                fc="tab:blue",
                zorder=2,
                label=fr"$\pm{se_factor:0.1f}\ \mathrm{{SE}}$",
            )
        )

        for (i, (factor, centre, shape)) in enumerate(zip(*sigmoid_params.values())):
            handles.append(
                ax.plot(
                    partial_plot_vals[feature_name],
                    sigmoid(partial_plot_vals[feature_name], factor, centre, shape),
                    label=pft_group_names[i],
                    c=f"C{i+1}",
                )[0]
            )

        ax.legend(handles=handles)

    for ax in axes.ravel()[N_features:]:
        ax.set_axis_off()

    fig.suptitle(exp_name)
    plt.tight_layout()

    plt.savefig(save_dir / f"{exp_key}_gam_curves.png")
    plt.close()


if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    save_dir = Path("~/tmp/gam-model-analysis/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)

    stdout_dir = save_dir / "stdout"
    stdout_dir.mkdir(exist_ok=True, parents=False)

    stdout_file = (
        stdout_dir / f"stdout_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    )

    plotting = partial(plotting, save_dir=save_dir)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record_bak"
    assert record_dir.is_dir()

    # To prevent memory accumulation during repeated calculations below.
    memoize.active = False

    global_params, global_losses = read_global_param_data()

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

    executor = ProcessPoolExecutor(max_workers=10)
    futures = []

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

        exp_name = get_exp_name(
            dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
        )
        logger.info(exp_name)

        exp_key = get_exp_key(
            dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
        )
        logger.info(exp_key)

        df_sel = df[sel]
        min_index = df_sel["loss"].argmin()
        min_loss = df_sel.iloc[min_index]["loss"]

        params = {
            key: val
            for key, val in df_sel.iloc[min_index].to_dict().items()
            if not pd.isna(val) and key not in ("loss",)
        }
        # pprint(params)

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

        # Create df for GAM fitting.
        X = np.zeros(((~combined_mask).sum(), len(valid_data_dict)))
        for i, valid_data in enumerate(valid_data_dict.values()):
            X[:, i] = valid_data

        feature_names = list(valid_data_dict.keys())

        gam_data_df = pd.DataFrame(
            np.hstack((X, valid_mon_avg_gfed_ba_1d.reshape(-1, 1))),
            columns=feature_names + ["response"],
        )

        gam_pred, partial_plot_vals, partial_preds, partial_pred_errs = gam_analysis(
            gam_data_df
        )

        # NOTE Unsupported (for now).
        assert not single_opt_kwargs

        # Partial dependence plots.
        futures.append(
            executor.submit(
                partial_dependence_plots,
                feature_names=feature_names,
                expanded_opt_kwargs=expanded_opt_kwargs,
                name_map=name_map,
                partial_plot_vals=partial_plot_vals,
                partial_preds=partial_preds,
                partial_pred_errs=partial_pred_errs,
                save_dir=save_dir,
                exp_key=exp_key,
            )
        )

        # Further data processing for map plots etc...

        pred_y_1d = np.ma.MaskedArray(
            np.zeros_like(combined_mask, dtype=np.float64), mask=True
        )
        pred_y_1d[~combined_mask] = gam_pred
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

    reference_obs = cube_1d_to_2d(
        get_1d_data_cube(mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
    ).data
    for exp_name, data in plot_data.items():
        futures.append(
            executor.submit(
                plotting,
                exp_name=exp_name,
                ref_2d_data=(reference_obs if exp_name != "GFED4" else None),
                **data,
            )
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

    for _ in tqdm(
        as_completed(futures), total=len(futures), desc="Waiting for executor"
    ):
        pass

    executor.shutdown()
