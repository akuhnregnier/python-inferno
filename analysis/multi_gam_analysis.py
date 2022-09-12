#!/Users/alexander/separate_miniconda/envs/python-inferno/bin/python
# -*- coding: utf-8 -*-

# Use the correct R installation.

# isort: off
import os

os.environ["R_HOME"] = "/Users/alexander/separate_miniconda/envs/python-inferno/lib/R"
# isort: on

import gc
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import reduce
from itertools import product
from operator import add
from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scipy.stats
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from numpy.testing import assert_allclose
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from tqdm import tqdm

from python_inferno.ba_model import ModelParams, calculate_scores
from python_inferno.cache import cache, mark_dependency
from python_inferno.configuration import (
    N_pft_groups,
    default_opt_record_dir,
    get_exp_key,
    get_exp_name,
    get_name_key_map,
    get_weight_key_map,
    land_pts,
    npft,
    pft_group_names,
    pft_groups,
)
from python_inferno.data import get_processed_climatological_data, load_jules_lats_lons
from python_inferno.inferno import sigmoid
from python_inferno.metrics_plotting import null_model_analysis
from python_inferno.plotting import (
    get_plot_name_map,
    get_plot_units_map,
    plot_label_case,
    plotting,
)
from python_inferno.utils import ConsMonthlyAvg, get_distinct_params, memoize

NoVal = Enum("NoVal", ["NoVal"])


def check_params(params, key, value=NoVal.NoVal):
    if all(key in p for p in params):
        if value is not NoVal.NoVal:
            if all(p[key] == value for p in params):
                return True
        else:
            return True
    return False


def read_global_param_data(*, record_dir=default_opt_record_dir):
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


def get_agg_data_mask(data):
    assert len(data.shape) in (2, 3)
    mask = np.ma.getmaskarray(data) | np.isnan(np.ma.getdata(data))
    if len(mask.shape) == 3:
        return np.any(mask, axis=1)
    return mask


def get_pft_group_valid_arrays(data, mask):
    if len(data.shape) != 3 or data.shape[1] < npft:
        raise ValueError(f"Unexpected shape: {data.shape}.")

    sel = ~mask
    n_valid = np.sum(sel)

    for pft_group in pft_groups:
        X = np.zeros((n_valid, len(pft_group)), dtype=data.dtype)
        for i, pft_i in enumerate(pft_group):
            X[:, i] = data[:, pft_i][sel]
        yield X


@contextmanager
@mark_dependency
def numpy2ri_context():
    try:
        numpy2ri.activate()
        yield
    finally:
        # Deactivate to prep for repeated call of `activate()` elsewhere.
        numpy2ri.deactivate()


@cache(dependencies=[numpy2ri_context])
def gam_analysis(
    *,
    valid_data_dict,
    valid_frac,
    valid_combined_frac,
    feature_names,
    response,
    save_dir,
    pft_group_names=pft_group_names,
    N_pft_groups=N_pft_groups,
    exp_key,
    exp_name,
):
    """GAM fitting and prediction using R.

    TODO:
        Currently there is no clear encapsulation of the R (rpy2) 'globalenv' used to
        store objects here. This could lead to issues if rpy2 is used elsewhere in the
        Python process prior to this function being called, as it may result in
        unintended side effects or differences that are not picked up by the @cache
        decorator.

    """
    stdout_dir = save_dir / "stdout"
    stdout_dir.mkdir(exist_ok=True, parents=False)

    stdout_file = (
        stdout_dir / f"stdout_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    )

    ro.r("library(mgcv)")
    ro.r(
        """
        file_append = function(s, stdout_file) {
            cat(s, file=stdout_file, sep='\n', append=TRUE)
        }
    """
    )

    # Store partial predictions (all but one variable held fixed).
    partial_plot_vals = {}
    partial_preds = {}
    partial_pred_errs = {}

    # Prepare formula string.
    gam_smooth_terms = []
    for feature_name in feature_names:
        if (
            len(list(filter(lambda s: feature_name in s, valid_data_dict.keys())))
            == N_pft_groups
        ):
            for pft_group_name in pft_group_names:
                comb_name = f"{feature_name}_{pft_group_name}"
                if len(valid_data_dict[comb_name].shape) == 2:
                    frac_name = f"frac_{pft_group_name}"
                    frac_data = valid_frac[frac_name]
                elif len(valid_data_dict[comb_name].shape) == 1:
                    frac_name = f"combined_frac_{pft_group_name}"
                    frac_data = valid_combined_frac[frac_name]
                else:
                    raise ValueError(
                        f"Unexpected shape: {valid_data_dict[comb_name].shape}"
                    )

                assert valid_data_dict[comb_name].shape == frac_data.shape
                gam_smooth_terms.append(f"s({comb_name}, by={frac_name})")
        else:
            assert len(valid_data_dict[feature_name].shape) == 1, str(
                (feature_name, valid_data_dict[feature_name].shape)
            )
            gam_smooth_terms.append(f"s({feature_name})")

    assert len(gam_smooth_terms) == len(valid_data_dict)

    formula_str = "y~" + "+".join(gam_smooth_terms)

    with localconverter(ro.default_converter), numpy2ri_context():
        # Transfer data to R.
        list_name_map = {"y": "response"}
        ro.globalenv["response"] = response

        for data_map in (
            valid_data_dict,
            valid_frac,
            valid_combined_frac,
        ):
            for key, data in data_map.items():
                ro.globalenv[key] = data
                list_name_map[key] = key

        ro.globalenv["stdout_file"] = str(stdout_file)
        ro.globalenv["exp_name"] = exp_name
        ro.globalenv["exp_key"] = exp_key
        ro.globalenv["image_dir"] = str(save_dir) + "/"

        # Evaluate the formula string.
        ro.r(f"gam_formula = as.formula({formula_str})")

        # Collate data given to the GAM (although using `globalenv` would also be an
        # option, this seems clearer).

        data_list_str = "".join(
            (
                "list(",
                ", ".join(f"{key}={val}" for key, val in list_name_map.items()),
                ")",
            )
        )

        ro.r(
            f"""
            fitted_gam = gam(
                gam_formula,
                method="REML",
                family=quasibinomial(link="logit"),
                data={data_list_str},
            )

            file_append(paste("Index:", exp_name), stdout_file)
            file_append(capture.output(print(summary(fitted_gam))), stdout_file)

            png(
              paste0(image_dir, paste0('gam_curves_', exp_key, '.png')),
              width=1500, height=1000, pointsize=15
            )
            plot(
                fitted_gam,
                residuals=FALSE,
                rug=TRUE,
                se=FALSE,
                shade=FALSE,
                pages=1,
                all.terms=FALSE
            )
            dev.off()
        """
        )

        gam_pred = ro.r('predict(fitted_gam, type="response")')
        print("Gam pred:")
        print(scipy.stats.describe(gam_pred))

        # Partial dependence evaluation - keep all but one variable fixed.
        partial_dep_N = 1000

        # Used to get from link to response scale.
        inv_link = ro.r("family(fitted_gam)$linkinv")

        for feature_name in valid_data_dict:
            partial_dep_name_map = {}

            feature_vals = np.linspace(
                np.min(valid_data_dict[feature_name]),
                np.max(valid_data_dict[feature_name]),
                partial_dep_N,
            )

            for name, data in valid_data_dict.items():
                partial_dep_name_map[name] = f"pd_{name}"
                ro.globalenv[partial_dep_name_map[name]] = (
                    feature_vals
                    if (name == feature_name)
                    else (np.ones(partial_dep_N) * np.median(data))
                )

            for frac_map in (
                valid_frac,
                valid_combined_frac,
            ):
                for key in frac_map:
                    partial_dep_name_map[key] = f"pd_{key}"
                    ro.globalenv[partial_dep_name_map[key]] = np.ones(partial_dep_N)

            partial_dep_data_list_str = "".join(
                (
                    "list(",
                    ", ".join(
                        f"{key}={val}" for key, val in partial_dep_name_map.items()
                    ),
                    ")",
                )
            )

            ro.globalenv["grid_pred"] = ro.r(
                f"""predict(
                    fitted_gam,
                    newdata={partial_dep_data_list_str},
                    type="link",
                    se.fit=TRUE
                )"""
            )

            partial_plot_vals[feature_name] = feature_vals

            partial_preds[feature_name] = ro.r("grid_pred$fit")
            partial_pred_errs[feature_name] = ro.r("grid_pred$se.fit")

    return gam_pred, inv_link, partial_plot_vals, partial_preds, partial_pred_errs


def partial_dependence_plots(
    *,
    fig_width=12.8,
    fig_height=6.9,
    se_factor=1e-4,
    feature_names,
    proc_params,
    name_map,
    weight_map,
    plot_name_map,
    plot_units_map,
    partial_plot_vals,
    inv_link,
    partial_preds,
    partial_pred_errs,
    save_dir,
    exp_key,
    exp_name,
    valid_data_dict,
):
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")

    N_features = len(feature_names)
    nrows = 4
    ncols = N_features

    assert N_features == len(plot_name_map)
    # Sort features using the order in `plot_name_map`.
    assert set(plot_name_map) == set(feature_names) == set(plot_units_map)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    for (col_axes, feature_name) in zip(axes.T, plot_name_map):
        sigmoid_params = {}
        for key, suffix in [
            ("factors", "_factor"),
            ("centres", "_centre"),
            ("shapes", "_shape"),
        ]:
            sigmoid_params[key] = proc_params[name_map[feature_name] + suffix]

        weights = proc_params[weight_map[feature_name]]

        assert all(len(values) == N_pft_groups for values in sigmoid_params.values())
        assert len(weights) == N_pft_groups

        col_axes[0].set_title(plot_name_map[feature_name])
        col_axes[-1].set_xlabel(
            plot_label_case(plot_name_map[feature_name])
            + f" ({plot_units_map[feature_name]})"
        )

        shift_mag = 120
        N_shift = 7

        for i, pft_group_name in enumerate(pft_group_names):
            col_ax = col_axes[i + 1]

            comb_name = f"{feature_name}_{pft_group_name}"
            ppred = partial_preds[comb_name]
            ppred -= np.mean(ppred)

            # Data for rug plots to visualise data distribution.
            rug_data = valid_data_dict[comb_name].ravel().copy()
            np.random.default_rng(0).shuffle(rug_data)
            rug_data = rug_data[::100]
            col_ax.plot(
                rug_data,
                [0.02] * rug_data.size,
                marker="|",
                alpha=0.13,
                c="k",
                zorder=5,
            )

            def fn(shift):
                with numpy2ri_context():
                    return inv_link(ppred + shift)

            for shift_i, shift in enumerate(
                get_distinct_params(fn, -shift_mag, shift_mag, N_shift, seed_N=2000)
            ):
                s_ppred = ppred + shift

                with numpy2ri_context():
                    col_ax.plot(
                        partial_plot_vals[comb_name],
                        inv_link(s_ppred),
                        label=(f"GAM {pft_group_name}" if shift_i == 0 else None),
                        c=f"C{i}",
                        zorder=3,
                    )

                # Standard error factor `se_factor`. Should be 2, but perhaps a lower factor
                # is better for plotting.

                expon = math.floor(np.log10(se_factor))
                significand = se_factor / 10**expon
                with numpy2ri_context():
                    col_ax.fill_between(
                        partial_plot_vals[comb_name],
                        inv_link(s_ppred - se_factor * partial_pred_errs[comb_name]),
                        inv_link(s_ppred + se_factor * partial_pred_errs[comb_name]),
                        alpha=0.2,
                        fc=f"C{i}",
                        zorder=2,
                        label=(
                            rf"$\pm{significand:0.1f} \times 10^{{{expon}}}\ \mathrm{{SE}}$"
                            if shift_i == 0
                            else None
                        ),
                    )

        for (i, (pft_group_name, (factor, centre, shape), weight, ls)) in enumerate(
            zip(
                pft_group_names,
                zip(*sigmoid_params.values()),
                weights,
                ["--", ":", "-."],
            )
        ):
            comb_name = f"{feature_name}_{pft_group_name}"
            col_axes[0].plot(
                partial_plot_vals[comb_name],
                sigmoid(partial_plot_vals[comb_name], factor, centre, shape),
                label=f"{pft_group_name} ({weight:0.2f})",
                c=f"C{i}",
                ls=ls,
            )

        for ax in col_axes:
            ax.legend()

    for letter, ax in zip(ascii_lowercase, axes.ravel()):
        ax.text(-0.01, 1.05, f"({letter})", transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_dir / f"{exp_key}_gam_curves.png")
    plt.close()


def main():
    save_dir = Path("~/tmp/multi-gam-model-analysis/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

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

    executor = ProcessPoolExecutor(max_workers=10)
    futures = []

    for dryness_method, fuel_build_up_method in product([1, 2], [1, 2]):
        name_map = get_name_key_map(
            dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
        )

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

        params = {
            key: val
            for key, val in df_sel.iloc[min_index].to_dict().items()
            if not pd.isna(val) and key not in ("loss",)
        }
        orig_params = params.copy()

        model_params = ModelParams(
            dryness_method=params.pop("dryness_method"),
            fuel_build_up_method=params.pop("fuel_build_up_method"),
            include_temperature=params.pop("include_temperature"),
            disc_params=params,
        )

        # NOTE - This will make parameters deviate from the originally-found minimum
        # (with the new INFERNO model), but is required to get 12 output months for
        # comparison with GFED4 data!
        model_params.average_samples = 183

        (
            data_dict,
            mon_avg_gfed_ba_1d,
            jules_time_coord,
        ) = get_processed_climatological_data(**model_params.disc_params)

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
        data_dict.pop("obs_pftcrop_1d")

        # Select single soil layer of soil moisture.
        data_dict["inferno_sm"] = data_dict.pop("sthu_soilt_single")

        combined_mask = np.logical_or(
            np.ma.getmaskarray(mon_avg_gfed_ba_1d),
            reduce(
                np.logical_or, (get_agg_data_mask(data) for data in data_dict.values())
            ),
        )

        frac = data_dict.pop("frac")
        combined_mask |= np.sum(frac[:, :npft], axis=1) < 1e-10

        feature_names = []
        valid_data_dict = {}
        for key, data in data_dict.items():
            if key not in name_map:
                continue

            feature_names.append(key)

            if len(data.shape) == 2:
                # e.g. for dry days, repeat data for each PFT group to allow fitting a
                # separate curve for each.
                sel_data = data[~combined_mask]
                for group_name in pft_group_names:
                    valid_data_dict[f"{key}_{group_name}"] = sel_data
                continue

            assert len(data.shape) == 3

            if data.shape[1] == N_pft_groups:
                for i, group_name in enumerate(pft_group_names):
                    valid_data_dict[f"{key}_{group_name}"] = data[:, i][~combined_mask]
                continue

            for group_name, X in zip(
                pft_group_names, get_pft_group_valid_arrays(data, combined_mask)
            ):
                valid_data_dict[f"{key}_{group_name}"] = X

        for valid_data in valid_data_dict.values():
            assert not np.any(np.isnan(valid_data))

        valid_mon_avg_gfed_ba_1d = np.ma.getdata(mon_avg_gfed_ba_1d)[~combined_mask]

        valid_frac = {}
        for group_name, frac_X in zip(
            pft_group_names,
            get_pft_group_valid_arrays(
                (frac[:, :npft] / np.sum(frac[:, :npft], axis=1)[:, np.newaxis, :]),
                combined_mask,
            ),
        ):
            valid_frac[f"frac_{group_name}"] = frac_X

        valid_combined_frac = {}
        for key, frac_X in valid_frac.items():
            valid_combined_frac[f"combined_{key}"] = np.sum(frac_X, axis=1)

        comb_valid_frac = reduce(add, (valid_combined_frac.values()))
        assert not np.any(np.isnan(comb_valid_frac))
        assert len(comb_valid_frac.shape) == 1
        assert_allclose(comb_valid_frac, 1.0, atol=1e-6, rtol=0)

        (
            gam_pred,
            inv_link,
            partial_plot_vals,
            partial_preds,
            partial_pred_errs,
        ) = gam_analysis(
            valid_data_dict=valid_data_dict,
            valid_frac=valid_frac,
            valid_combined_frac=valid_combined_frac,
            feature_names=feature_names,
            response=valid_mon_avg_gfed_ba_1d,
            save_dir=save_dir,
            exp_name=exp_name,
            exp_key=exp_key,
        )

        # Partial dependence plots.
        futures.append(
            executor.submit(
                partial_dependence_plots,
                feature_names=feature_names,
                proc_params=model_params.process_kwargs(**params),
                name_map=name_map,
                weight_map=get_weight_key_map(
                    dryness_method=dryness_method,
                    fuel_build_up_method=fuel_build_up_method,
                ),
                plot_name_map=get_plot_name_map(
                    dryness_method=dryness_method,
                    fuel_build_up_method=fuel_build_up_method,
                ),
                plot_units_map=get_plot_units_map(
                    dryness_method=dryness_method,
                    fuel_build_up_method=fuel_build_up_method,
                ),
                partial_plot_vals=partial_plot_vals,
                inv_link=inv_link,
                partial_preds=partial_preds,
                partial_pred_errs=partial_pred_errs,
                save_dir=save_dir,
                exp_key=exp_key,
                exp_name=exp_name,
                valid_data_dict=valid_data_dict,
            )
        )

        # Further data processing for map plots etc...

        pred_y_1d = np.ma.MaskedArray(
            np.zeros_like(combined_mask, dtype=np.float64), mask=True
        )
        pred_y_1d[~combined_mask] = gam_pred
        mon_avg_gfed_ba_1d.mask |= combined_mask

        # NOTE GAM analysis does not take crop cover into account?

        scores, avg_ba = calculate_scores(
            model_ba=pred_y_1d,
            cons_monthly_avg=ConsMonthlyAvg(jules_time_coord, L=land_pts),
            mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
        )

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
            scores=scores,
            data_params=orig_params,
        )
        gc.collect()

    reference_obs = cube_1d_to_2d(
        get_1d_data_cube(mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
    ).data

    for exp_name, data in plot_data.items():
        futures.append(
            executor.submit(
                plotting,
                save_dir=save_dir,
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

    for f in tqdm(
        as_completed(futures), total=len(futures), desc="Waiting for executor"
    ):
        # Query the result to notice exceptions (if any).
        f.result()

    executor.shutdown()


if __name__ == "__main__":
    main()
