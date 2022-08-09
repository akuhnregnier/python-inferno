# -*- coding: utf-8 -*-
import pickle
from contextlib import contextmanager
from enum import Enum
from itertools import product
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from tqdm import tqdm

from .cache import mark_dependency
from .configuration import get_exp_key, get_exp_name

NoVal = Enum("NoVal", ["NoVal"])


def check_params(params, key, value=NoVal.NoVal):
    if all(key in p for p in params):
        if value is not NoVal.NoVal:
            if all(p[key] == value for p in params):
                return True
        else:
            return True
    return False


def get_model_params(*, record_dir, verbose=False, progress=True):
    assert record_dir.is_dir()

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

    if verbose:
        print(df.head())
        print("\nNumber of trials:\n")
        print(df.groupby(cat_names).size())

        print("\nMinimum loss by parametrisation approach:\n")
        print(df.groupby(cat_names)["loss"].min())

    def method_iter():
        for dryness_method, fuel_build_up_method in tqdm(
            list(product([1, 2], [1, 2])), disable=not progress, desc="Methods"
        ):
            sel = (df["dryness_method"] == dryness_method) & (
                df["fuel_build_up_method"] == fuel_build_up_method
            )
            if not np.any(sel):
                continue

            df_sel = df[sel]
            min_index = df_sel["loss"].argmin()
            min_loss = df_sel.iloc[min_index]["loss"]

            params = {
                key: val
                for key, val in df_sel.iloc[min_index].to_dict().items()
                if not pd.isna(val) and key not in ("loss",)
            }
            if verbose:
                pprint(params)

            exp_name = get_exp_name(
                dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
            )

            exp_key = get_exp_key(
                dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
            )

            yield (
                dryness_method,
                fuel_build_up_method,
                df_sel,
                min_index,
                min_loss,
                params,
                exp_name,
                exp_key,
            )

    return df, method_iter


@contextmanager
def param_fig(*, param_vals, losses, title, col, save_dir):
    try:
        fig, ax = plt.subplots()
        ax.plot(param_vals, losses, linestyle="", marker="o", alpha=0.6)
        ax.set_xlabel(col)
        ax.set_ylabel("loss")
        ax.set_title(title)
        if col in ("rain_f", "vpd_f", "litter_tc", "leaf_f"):
            plt.xscale("log")

        yield fig, ax
    finally:
        plt.savefig(save_dir / f"{col}.png")
        plt.close()


@mark_dependency
def get_param_uncertainties(*, df_sel, plot=False, exp_name=None, save_dir=None):
    param_ranges = {}

    for col in [col for col in df_sel.columns if col != "loss"]:
        if df_sel[col].isna().all():
            continue

        param_vals = df_sel[col]
        losses = df_sel["loss"]

        min_loss = df_sel["loss"].min()
        df_sel["loss"].max()

        quantiles = np.percentile(df_sel["loss"], [25, 75])
        iqr = quantiles[1] - quantiles[0]

        comp_loss = min_loss + 0.3 * iqr

        points = np.hstack(
            (
                np.asarray(param_vals).reshape(-1, 1),
                np.asarray(losses).reshape(-1, 1),
            )
        )
        hull = ConvexHull(points)

        param_range = []

        for a, b in zip(hull.vertices, np.roll(hull.vertices, -1)):
            point_a = points[a]
            point_b = points[b]

            point_losses = np.array([point_a[1], point_b[1]])

            if np.all(point_losses > comp_loss):
                continue
            elif (
                np.sum(point_losses >= comp_loss) == 1
                and np.sum(point_losses <= comp_loss) == 1
            ):
                # Compute intersection between the line joining the two points and the
                # horizontal line at `comp_loss`.

                # y = ax + b
                # b = y - ax

                dx = point_b[0] - point_a[0]

                if abs(dx) < 1e-15:
                    param_range.append(point_a[0])
                else:
                    dy = point_b[1] - point_a[1]
                    grad = dy / dx
                    intersect = point_a[1] - grad * point_a[0]

                    # comp_loss = a*x + b
                    # x = (comp_loss - b) /a

                    param_range.append((comp_loss - intersect) / grad)

                assert (
                    (point_a[0] <= param_range[-1]) and (point_b[0] >= param_range[-1])
                ) or (
                    (point_a[0] >= param_range[-1]) and (point_b[0] <= param_range[-1])
                )
            else:
                continue

        assert len(param_range) == 2

        param_ranges[col] = sorted(param_range)

        if plot:
            assert save_dir is not None

            with param_fig(
                param_vals=param_vals,
                losses=losses,
                title=exp_name,
                col=col,
                save_dir=save_dir,
            ) as (fig, ax):
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                convex_hull_plot_2d(hull, ax=ax)

                ax.hlines(min_loss, *xlim, colors="g")
                ax.hlines(comp_loss, *xlim, colors="r")

                ax.vlines(param_range, *ylim, colors="b")

    return param_ranges


def plot_param_histograms(df_sel, exp_name, hist_save_dir):
    # NOTE Not actually histograms!
    for col in [col for col in df_sel.columns if col != "loss"]:
        if df_sel[col].isna().all():
            continue

        with param_fig(
            param_vals=df_sel[col],
            losses=df_sel["loss"],
            title=exp_name,
            col=col,
            save_dir=hist_save_dir,
        ):
            pass
