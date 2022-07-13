# -*- coding: utf-8 -*-
from enum import Enum

import numpy as np
from loguru import logger
from numba import njit
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from tqdm import tqdm

from .cache import cache, mark_dependency
from .utils import linspace_no_endpoint

Metrics = Enum(
    "Metrics",
    ["NME", "NMSE", "MPD", "R2", "LOGHIST", "ARCSINH_NME", "SSE", "ARCSINH_SSE"],
)


@njit(nogil=True, cache=True, fastmath=True)
def nme_simple(*, obs, pred):
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

    return err


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
        return err, (np.mean(np.abs(pred - obs)) / denom) / (N**0.5)
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


def sse(*, obs, pred):
    """Sum of squared errors.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.

    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    return np.sum((pred - obs) ** 2)


@njit(nogil=True, cache=True)
def calculate_phase(x):
    if len(x.shape) != 2 or x.shape[0] != 12:
        raise ValueError("Unexpected shape encountered (should be (12, N)).")
    theta = (2 * np.pi * linspace_no_endpoint(0, 1, 12)).reshape(12, 1)
    lx = np.sum(x * np.cos(theta), axis=0)
    ly = np.sum(x * np.sin(theta), axis=0)
    return np.arctan2(lx, ly)


@njit(nogil=True, cache=True)
def calculate_phase_2d(x):
    """Calculate phase of data with shape (12, M, N)."""
    if len(x.shape) != 3 or x.shape[0] != 12:
        raise ValueError("Unexpected shape encountered (should be (12, M, N)).")

    phase = np.zeros(x.shape[1:])
    for i in range(x.shape[1]):
        phase[i] = calculate_phase(x[:, i])
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

    phase_diff = phase_func(x=pred) - phase_func(x=obs)
    # assert obs.shape == pred.shape
    # all_phases = phase_func(x=np.concatenate((pred, obs), axis=1))
    # phase_diff = all_phases[: obs.shape[1]] - all_phases[obs.shape[1] :]

    vals = add_mask(np.arccos(np.cos(phase_diff)))
    mpd_val = (1 / np.pi) * np.ma.mean(vals)

    to_return = [mpd_val]
    if return_ignored:
        to_return.append(np.sum(np.all(ignore_mask, axis=0)))
    if return_std:
        N = np.ma.getdata(vals)[~np.ma.getmaskarray(vals)].size
        # Return the standard deviation of the mean.
        to_return.append(np.ma.std(vals) / (N**0.5))
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


def calculate_factor_r2(*, y_true, y_pred):
    """Calculate adjustment factor to convert `y_pred` to `y_true`.

    This is done by maximising the R2 (i.e. minimising -R2).

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def f(factor):
        return -r2_score(y_true=y_true, y_pred=factor * y_pred)

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
        resampled = np.random.default_rng(0).choice(
            valid_reference_data, size=valid_reference_data.size
        )
        nme_errors[i] = nme(obs=valid_reference_data, pred=resampled)
        resampled_map[total_sel] = resampled
        mpd_errors[i] = mpd(obs=reference_data, pred=resampled_map)
    return nme_errors, mpd_errors
