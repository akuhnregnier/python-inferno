# -*- coding: utf-8 -*-
import numpy as np


def nme(*, obs, pred):
    """Normalised mean error.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.

    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    return np.sum(np.abs(pred - obs)) / np.sum(np.abs(obs - np.mean(obs)))


def nmse(*, obs, pred):
    """Normalised mean squared error.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.

    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    return np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


def calculate_phase(x):
    if len(x.shape) != 2 or x.shape[0] != 12:
        raise ValueError(f"Shape should be (12, N), got {x.shape}.")
    theta = (2 * np.pi * np.linspace(0, 1, 12, endpoint=False)).reshape(12, 1)
    lx = np.sum(x * np.cos(theta), axis=0)
    ly = np.sum(x * np.sin(theta), axis=0)
    return np.arctan2(lx, ly)


def mpd(*, obs, pred, return_ignored=False):
    """Mean phase difference.

    Args:
        obs (array-like): Observations.
        pred (array-like): Predictions.
        return_ignored (bool): If True, return the number of ignored samples.

    """
    if (
        len(obs.shape) != 2
        or obs.shape[0] != 12
        or len(pred.shape) != 2
        or pred.shape[0] != 12
    ):
        raise ValueError(f"Shape should be (12, N), got {x.shape}.")

    # Ignore those locations with all 0s in either `obs` or `pred`.
    def close_func(a, b):
        return np.all(np.isclose(a, b, rtol=0, atol=1e-15), axis=0)

    ignore_mask = np.ones((12, 1), dtype=np.bool_) & (
        (close_func(obs, 0) | close_func(pred, 0)).reshape(1, -1)
    )

    def add_mask(arr):
        if np.ma.isMaskedArray(arr):
            carr = arr.copy()
            carr.mask |= ignore_mask
            return carr
        return np.ma.MaskedArray(arr, mask=ignore_mask)

    mpd_val = np.mean(
        (1 / np.pi)
        * np.arccos(
            np.cos(calculate_phase(add_mask(pred)) - calculate_phase(add_mask(obs)))
        )
    )
    if return_ignored:
        return mpd_val, np.sum(np.all(ignore_mask, axis=0))
    return mpd_val
