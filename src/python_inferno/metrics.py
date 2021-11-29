# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numba import njit
from scipy.optimize import minimize
from tqdm import tqdm

from .cache import cache, mark_dependency
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
    err = np.mean(np.abs(pred - obs)) / denom
    if return_std:
        return err, np.mean(np.abs(pred - obs)) / denom
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
        to_return.append(np.ma.std(vals))
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


def null_model_analysis(
    reference_data, comp_data=None, rng=None, save_dir=None, N=10000
):
    """Data should have the 3D shape (12, M, N), i.e. map data over 12 months.

    Args:
        N (int):  Number of resampling operations.

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

    def error_hist(*, errors, title, error_dict, filename):
        plt.figure()
        plt.hist(errors, bins="auto", density=True)
        plt.title(title)

        ax2 = plt.gca().twinx()

        # Indicate other errors.
        prev_ylim = plt.ylim()
        for (i, (key, (err, std))) in enumerate(error_dict.items()):
            plt.vlines(err, *prev_ylim, color=f"C{i+1}", label=key)
            plt.vlines(err - std, *prev_ylim, color=f"C{i+1}", alpha=0.2)
            plt.vlines(err + std, *prev_ylim, color=f"C{i+1}", alpha=0.2)

            xs = np.linspace(0, 2, 100)
            ax2.plot(
                xs,
                (1 / np.sqrt(2 * np.pi * std ** 2))
                * np.exp(-((xs - err) ** 2) / (2 * std ** 2)),
                c=f"C{i+1}",
            )
        plt.ylim(*prev_ylim)
        plt.legend(loc="best")
        if save_dir is not None:
            plt.savefig(save_dir / filename)
            plt.close()

    # NME Errors.
    error_hist(
        errors=nme_errors,
        title="NME errors",
        error_dict=nme_error_dict,
        filename="nme_errors.png",
    )

    # MPD Errors.
    error_hist(
        errors=mpd_errors,
        title="MPD errors",
        error_dict=mpd_error_dict,
        filename="mpd_errors.png",
    )
