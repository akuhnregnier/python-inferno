# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numba import njit
from scipy.optimize import minimize
from tqdm import tqdm

from .cache import cache, mark_dependency
from .data import get_gfed_regions, get_pnv_mega_plot_data
from .utils import linspace_no_endpoint


@mark_dependency
def nme(*, obs, pred, return_std=False):
    """Normalised mean error.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.

    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    denom = np.mean(np.abs(obs - np.mean(obs)))

    abs_diff = np.abs(pred - obs)
    err = np.mean(abs_diff) / denom

    if return_std:
        N = np.ma.getdata(abs_diff)[~np.ma.getmaskarray(abs_diff)].size
        # Return the standard deviation of the mean.
        return err, (np.mean(np.abs(pred - obs)) / denom) / (N ** 0.5)
    return err


def nmse(*, obs, pred):
    """Normalised mean squared error.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.

    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    return np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


@njit(nogil=True, cache=True)
def calculate_phase(x):
    if len(x.shape) != 2 or x.shape[0] != 12:
        raise ValueError("Unexpected shape encountered (should be (12, N)).")
    theta = (2 * np.pi * linspace_no_endpoint(0, 1, 12)).reshape(12, 1)
    lx = np.sum(x * np.cos(theta), axis=0)
    ly = np.sum(x * np.sin(theta), axis=0)
    return np.arctan2(lx, ly)


@njit(nogil=True, cache=True)
def calculate_phase_2d(data):
    """Calculate phase of data with shape (12, M, N)."""
    if len(data.shape) != 3 or data.shape[0] != 12:
        raise ValueError("Unexpected shape encountered (should be (12, M, N)).")

    phase = np.zeros(data.shape[1:])
    for i in range(data.shape[1]):
        phase[i] = calculate_phase(data[:, i])
    return phase


@mark_dependency
def mpd(*, obs, pred, return_ignored=False, return_std=False):
    """Mean phase difference.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.
        return_ignored (bool): If True, return the number of ignored samples.

    """
    if len(obs.shape) == 2 and len(pred.shape) == 2:
        phase_func = calculate_phase
    elif len(obs.shape) == 3 and len(pred.shape) == 3:
        phase_func = calculate_phase_2d
    else:
        raise ValueError(f"Shape should be (12, N), got {obs.shape} and {pred.shape}.")
    if obs.shape[0] != 12 or pred.shape[0] != 12:
        raise ValueError(f"Shape should be (12, N), got {obs.shape} and {pred.shape}.")

    # Ignore those locations with all 0s in either `obs` or `pred`.
    def close_func(a, b):
        return np.all(np.isclose(a, b, rtol=0, atol=1e-15), axis=0)

    ignore_mask = (
        close_func(np.ma.getdata(obs), 0) | close_func(np.ma.getdata(pred), 0)
    ).reshape(1, *(obs.shape[1:]))

    combined_mask = (
        np.any(np.ma.getmaskarray(pred), axis=0)
        | np.any(np.ma.getmaskarray(obs), axis=0)
        | ignore_mask
    )

    def add_mask(arr):
        return np.ma.MaskedArray(np.ma.getdata(arr), mask=combined_mask)

    vals = (1 / np.pi) * add_mask(np.arccos(np.cos(phase_func(pred) - phase_func(obs))))
    mpd_val = np.ma.mean(vals)

    to_return = [mpd_val]
    if return_ignored:
        to_return.append(np.sum(np.all(ignore_mask, axis=0)))
    if return_std:
        N = np.ma.getdata(vals)[~np.ma.getmaskarray(vals)].size
        # Return the standard deviation of the mean.
        to_return.append(np.ma.std(vals) / (N ** 0.5))
    if return_ignored or return_std:
        return tuple(to_return)
    return to_return[0]


def loghist(*, obs, pred, edges):
    """Logarithmic histogram metric.

    TODO: Implement weighting, e.g. by bin widths.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.
        edges (array-like): Bin edges used for binning.

    """

    def bin_func(x):
        binned = np.histogram(x, bins=edges)[0]
        sel = binned > 0
        binned[sel] = np.log10(binned[sel])
        # Replace 0s with -1.
        binned[~sel] = -1
        return binned

    binned_obs = bin_func(obs)
    binned_pred = bin_func(pred)

    return np.linalg.norm(binned_obs - binned_pred) / np.linalg.norm(binned_obs)


def calculate_factor(*, y_true, y_pred):
    """Calculate adjustment factor to convert `y_pred` to `y_true`.

    This is done by minimising the NME.

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def f(factor):
        return nme(obs=y_true, pred=factor * y_pred)

    # Minimize `f`, with the initial guess being the ratio of the means.
    guess = np.mean(y_true) / np.mean(y_pred)
    factor = minimize(f, guess).x[0]
    logger.debug(f"Initial guess: {guess:0.1e}, final factor: {factor:0.1e}.")
    return factor


@cache(dependencies=[nme, mpd])
def calculate_resampled_errors(*, reference_data, valid_reference_data, total_sel, N):
    # Error for repeated subsampling (with replacement) of the reference data
    # (Observations).
    nme_errors = np.zeros(N)
    mpd_errors = np.zeros(N)
    resampled_map = np.ma.MaskedArray(np.zeros_like(reference_data), mask=True)
    for i in tqdm(range(N), desc="Calculating resampling errors"):
        resampled = np.random.choice(
            valid_reference_data, size=valid_reference_data.size
        )
        nme_errors[i] = nme(obs=valid_reference_data, pred=resampled)
        resampled_map[total_sel] = resampled
        mpd_errors[i] = mpd(obs=reference_data, pred=resampled_map)
    return nme_errors, mpd_errors


def error_hist(*, errors, title, error_dict, save_path, fig=None, ax=None):
    if fig is None and ax is None:
        fig = plt.figure()
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

    ax.set_title(title)

    if errors is not None:
        ax.hist(errors, bins="auto", density=True)

    xmins = [np.min(errors)] if errors is not None else []
    xmaxs = [np.max(errors)] if errors is not None else []

    ax2 = ax.twinx()

    # Indicate other errors.
    prev_ylim = ax.get_ylim()
    for (i, (key, (err, std))) in enumerate(error_dict.items()):
        ax.vlines(err, *prev_ylim, color=f"C{i+1}", label=key)

        xs = np.linspace(err - std, err + std, 100)
        ax2.plot(
            xs,
            (1 / np.sqrt(2 * np.pi * std ** 2))
            * np.exp(-((xs - err) ** 2) / (2 * std ** 2)),
            c=f"C{i+1}",
            linestyle="--",
            alpha=0.8,
        )

        xmins.append(err - std)
        xmaxs.append(err + std)

    ax.set_ylim(*prev_ylim)
    ax.set_xlim(np.min(xmins), np.max(xmaxs))

    ax.legend(loc="best")
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)


def null_model_analysis(
    reference_data,
    comp_data=None,
    rng=None,
    save_dir=None,
    N=10000,
    regions="PNV",
):
    """Data should have the 3D shape (12, M, N), i.e. map data over 12 months.

    For N96e, M=144, N=192

    Args:
        reference_data (12, lat, lon) numpy array: Reference data.
        comp_data (dict of (12, lat, lon) numpy array): Data to compare to
            `reference_data`.
        rng (numpy.random.default_rng):
        save_dir (save directory):
        N (int):  Number of resampling operations.
        regions ({None, "GFED", "PNV"}): If not None, perform a regional error
            breakdown.

    """
    total_mask = np.zeros_like(reference_data, dtype=np.bool_)

    if comp_data is None:
        comp_data = {}
    else:
        for data in comp_data.values():
            assert len(data.shape) == 3
            total_mask |= np.ma.getmaskarray(data)

    assert reference_data.ndim == 3
    total_mask |= np.ma.getmaskarray(reference_data)
    total_sel = ~total_mask

    if rng is None:
        rng = np.random.default_rng()

    valid_reference_data = np.ma.getdata(reference_data)[total_sel]
    valid_comp_data = {
        key: np.ma.getdata(data)[total_sel] for key, data in comp_data.items()
    }

    nme_errors, mpd_errors = calculate_resampled_errors(
        reference_data=reference_data,
        valid_reference_data=valid_reference_data,
        total_sel=total_sel,
        N=N,
    )

    nme_error_dict = {}
    mpd_error_dict = {}

    # Error given just the mean state.
    nme_error_dict["mean_state"] = nme(
        obs=valid_reference_data,
        pred=np.zeros_like(valid_reference_data) + np.mean(valid_reference_data),
        return_std=True,
    )
    mpd_error_dict["mean_state"] = mpd(
        obs=reference_data,
        pred=np.zeros_like(reference_data) + np.mean(valid_reference_data),
        return_std=True,
    )

    # Errors for the other data.
    for key, data in valid_comp_data.items():
        nme_error_dict[key] = nme(obs=valid_reference_data, pred=data, return_std=True)
    for key, data in comp_data.items():
        mpd_error_dict[key] = mpd(obs=reference_data, pred=data, return_std=True)

    # NME Errors.
    error_hist(
        errors=nme_errors,
        title="NME errors (with std of mean)",
        error_dict=nme_error_dict,
        save_path=save_dir / "nme_errors.png",
    )

    # MPD Errors.
    error_hist(
        errors=mpd_errors,
        title="MPD errors (with std of mean)",
        error_dict=mpd_error_dict,
        save_path=save_dir / "mpd_errors.png",
    )

    if regions is not None:
        # Regional plotting.
        if regions == "GFED":
            regions_cube, N_plots = get_gfed_regions()
        elif regions == "PNV":
            regions_cube, N_plots = get_pnv_mega_plot_data()
        else:
            raise ValueError(f"Unknown regions: '{regions}'.")

        region_nrows_ncols = dict(nrows=math.ceil(N_plots / 2), ncols=2)
        regions_cube.data.mask |= np.any(total_mask, axis=0)

        nme_fig, nme_axes = plt.subplots(
            sharex=True, sharey=True, figsize=(9, 9), **region_nrows_ncols
        )
        mpd_fig, mpd_axes = plt.subplots(
            sharex=True, sharey=True, figsize=(9, 9), **region_nrows_ncols
        )

        for (plot_i, (region_code, region_name)) in enumerate(
            {
                code: name
                for code, name in regions_cube.attributes["short_regions"].items()
                # Ignore the Ocean region.
                if code != 0
            }.items()
        ):
            region_sel = (
                np.ones((12, 1, 1), dtype=np.bool_)
                & (np.ma.getdata(regions_cube.data) == region_code)[np.newaxis]
            )
            # NOTE hist() does not seem to handle masked arrays, so ensure that only valid
            # entries are passed to it.

            # Regional errors.
            reg_nme_error_dict = {}
            reg_mpd_error_dict = {}

            reg_reference_data = np.ma.MaskedArray(
                np.ma.getdata(reference_data),
                mask=~(~np.ma.getmaskarray(reference_data) & region_sel),
            )
            reg_valid_reference_data = np.ma.getdata(reference_data)[region_sel]

            reg_valid_comp_data = {
                key: np.ma.getdata(data)[region_sel] for key, data in comp_data.items()
            }
            reg_comp_data = {
                key: np.ma.MaskedArray(
                    np.ma.getdata(data), mask=~(~np.ma.getmaskarray(data) & region_sel)
                )
                for key, data in comp_data.items()
            }

            for key, data in reg_valid_comp_data.items():
                reg_nme_error_dict[key] = nme(
                    obs=reg_valid_reference_data, pred=data, return_std=True
                )
            for key, data in reg_comp_data.items():
                reg_mpd_error_dict[key] = mpd(
                    obs=reg_reference_data, pred=data, return_std=True
                )

            error_hist(
                errors=None,
                # Show the region name and number of selected locations.
                title=f"{region_name} (n={np.sum(region_sel) / 12})",
                error_dict=reg_nme_error_dict,
                save_path=None,
                fig=nme_fig,
                ax=nme_axes.ravel()[plot_i],
            )

            error_hist(
                errors=None,
                # Show the region name and number of selected locations.
                title=f"{region_name} (n={np.sum(region_sel) / 12})",
                error_dict=reg_mpd_error_dict,
                save_path=None,
                fig=mpd_fig,
                ax=mpd_axes.ravel()[plot_i],
            )

        nme_fig.suptitle("Regional NME Errors")
        mpd_fig.suptitle("Regional MPD Errors")

        for fig in [nme_fig, mpd_fig]:
            fig.tight_layout(rect=[0, 0.0, 1, 0.98])

        nme_fig.savefig(save_dir / "regional_nme_errors.png")
        mpd_fig.savefig(save_dir / "regional_mpd_errors.png")

        for fig in [nme_fig, mpd_fig]:
            plt.close(fig)
