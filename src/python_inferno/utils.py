# -*- coding: utf-8 -*-
import os
from functools import reduce, wraps

import iris
import numpy as np
from numba import njit

from .cache import mark_dependency

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm  # noqa
else:
    from tqdm import tqdm  # noqa


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


def combine_masks(*masks):
    """Combine boolean arrays using `np.logical_or`."""
    return reduce(np.logical_or, masks)


def combine_ma_masks(*masked_arrays):
    """Combine masks of MaskedArray."""
    return combine_masks(*(np.ma.getmaskarray(arr) for arr in masked_arrays))


@mark_dependency
def make_contiguous(*arrays):
    """Return C-contiguous arrays."""
    out = []
    for arr in arrays:
        if np.ma.isMaskedArray(arr):
            out.append(
                np.ma.MaskedArray(
                    np.ascontiguousarray(arr.data),
                    mask=(
                        np.ascontiguousarray(arr.mask)
                        if isinstance(arr.mask, np.ndarray)
                        else arr.mask
                    ),
                )
            )
        elif isinstance(arr, iris.cube.Cube):
            arr.data = make_contiguous(arr.data)
            out.append(arr)
        else:
            out.append(np.ascontiguousarray(arr))
    return tuple(out) if len(out) != 1 else out[0]


def core_unpack_wrapped(*objs):
    """Extract the __wrapped__ attribute if it exists."""
    out = []
    for obj in objs:
        if hasattr(obj, "__wrapped__"):
            out.append(obj.__wrapped__)
        else:
            out.append(obj)
    return tuple(out) if len(out) != 1 else out[0]


def unpack_wrapped(func):
    # NOTE: This decorator does not support nested proxy objects.
    @wraps(func)
    def inner(*args, **kwargs):
        # Call the wrapped function, unpacking any wrapped parameters in the process
        # (no nesting).
        return func(
            *core_unpack_wrapped(*args),
            **{key: core_unpack_wrapped(val) for key, val in kwargs.items()}
        )

    return inner
