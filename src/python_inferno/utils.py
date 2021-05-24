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
    data = np.asarray(data)
    output_samples = data.shape[0] * factor
    closest_indices = np.clip(
        np.rint(
            np.linspace(0, data.shape[0], output_samples, endpoint=False),
        ).astype(np.int64),
        0,
        data.shape[0] - 1,
    )
    return data[closest_indices]


@njit(cache=True)
def exponential_average(data, alpha):
    """Exponential averaging (temporal shifting).

    Args:
        data (array-like): Data to be averaged.
        alpha (float): Alpha parameter.

    Returns:
        array-like: The averaged data.

    """
    weighted = np.zeros_like(data, dtype=np.float64)
    N = data.shape[0]
    for i, sample in enumerate(data):
        temp = weighted[(i - 1) % N]
        temp *= 1 - alpha
        temp += alpha * sample
        weighted[i] = temp
    return weighted


@njit(cache=True)
def pre_seed_exponential_average(data, alpha, weighted):
    """Exponential averaging (temporal shifting).

    Args:
        data (array-like): Data to be averaged.
        alpha (float): Alpha parameter.
        weighted (array-like): Array with the same shape as `data`. This can be used
            to seed the averaging.

    Returns:
        array-like: The averaged data.

    """
    N = data.shape[0]
    for i, sample in enumerate(data):
        temp = weighted[(i - 1) % N]
        temp *= 1 - alpha
        temp += alpha * sample
        weighted[i] = temp
    return weighted


def repeated_exponential_average(data, alpha, repetitions=10):
    """Repeated exponential averaging.

    Args:
        data (array-like): Data to be averaged.
        alpha (float): Alpha parameter.
        repetitions (int): The number of repetitions.

    Returns:
        array-like: The averaged data after `repetitions` rounds of averaging.

    """
    weighted = np.zeros_like(data, dtype=np.float64)
    for i in range(repetitions):
        weighted = pre_seed_exponential_average(data, alpha, weighted=weighted)
    return weighted
