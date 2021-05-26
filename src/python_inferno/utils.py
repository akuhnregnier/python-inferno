# -*- coding: utf-8 -*-
import numpy as np
from numba import njit


def temporal_nearest_neighbour_interp(data, factor):
    """Temporal nearest-neighbour interpolation along axis 0.

    Interpolation starts at the first sample.

    Args:
        data (array-like): Data to interpolate.
        factor (int): Multiple of the number of input samples (along axis=0) which
            determines the number of output samples.

    """
    data = np.ma.asarray(data)
    output_samples = data.shape[0] * factor
    closest_indices = np.clip(
        np.rint(
            np.linspace(0, data.shape[0], output_samples, endpoint=False),
        ).astype(np.int64),
        0,
        data.shape[0] - 1,
    )
    return data[closest_indices]


@njit(nogil=True, cache=True)
def _exponential_average(data, alpha, weighted):
    N = data.shape[0]
    for i, sample in enumerate(data):
        temp = weighted[(i - 1) % N]
        temp *= 1 - alpha
        temp += alpha * sample
        weighted[i] = temp
    return weighted


def exponential_average(data, alpha, repetitions=1):
    """Exponential averaging (temporal shifting).

    Args:
        data (array-like): Data to be averaged.
        alpha (float): Alpha parameter.
        repetitions (int): The number of repetitions. If `repetitions > 1`, this
            allows for repeated exponential averaging to approach convergence.

    Returns:
        array-like: The averaged data.

    """
    if not isinstance(data, np.ma.core.MaskedArray):
        data = np.ma.asarray(data)

    weighted = np.ma.MaskedArray(
        np.zeros_like(data, dtype=np.float64),
        mask=np.repeat(np.any(data.mask, axis=0)[None], data.shape[0], axis=0),
    )
    if np.all(weighted.mask):
        # No need to carry out averaging.
        return np.ma.MaskedArray(data, mask=True)
    # Only process those elements which are unmasked.

    output = np.ma.masked_all(data.shape, dtype=np.float64)
    sel = ~np.ma.getmaskarray(weighted)
    selected = data.data[sel]
    weighted = np.zeros_like(selected, dtype=np.float64)
    for i in range(repetitions):
        weighted = _exponential_average(selected, alpha, weighted=weighted)
    output[sel] = weighted
    return output
