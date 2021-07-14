# -*- coding: utf-8 -*-
import os
from functools import reduce, wraps

import iris
import numpy as np
from iris.coord_categorisation import add_month_number, add_year
from numba import njit
from scipy.optimize import minimize

from .cache import mark_dependency
from .metrics import nme

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm  # noqa
else:
    from tqdm import tqdm  # noqa


@mark_dependency
def temporal_nearest_neighbour_interp(data, factor, mode):
    """Temporal nearest-neighbour interpolation along axis 0.

    Interpolation starts at the first sample.

    Args:
        data (array-like): Data to interpolate.
        factor (int): Multiple of the number of input samples (along axis=0) which
            determines the number of output samples.
        mode ({'start', 'end', 'centre'}): Position of the source sample relative to
            the interpolated samples. For 'start', samples will be taken like [0, 0,
            0, 1, 1, 1, 2, ...]. For 'end', samples will be taken like [0, 1, 1, 1, 2,
            2, ...]. For 'centre', samples will be taken like [0, 0, 1, 1, 1, 2, ...].

    """
    data = np.ma.asarray(data)
    closest_indices = np.clip(
        np.repeat(np.arange(data.shape[0] + 1), factor)[
            {
                "start": slice(None, -factor),
                "end": slice(factor - 1, None),
                "centre": slice(factor // 2, None),
            }[mode]
        ],
        0,
        data.shape[0] - 1,
    )[: data.shape[0] * factor]
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

    data.mask = np.ma.getmaskarray(data)

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
            **{key: core_unpack_wrapped(val) for key, val in kwargs.items()},
        )

    return inner


def monthly_average_data(data, time_coord=None, trim_single=True):
    """Calculate monthly average of data.

    If `trim_single` is True, a single day e.g. (01/01/2000) at the end of the input
    data will be discarded to avoid generating an 'average' month from a single sample
    only.

    """
    if isinstance(data, iris.cube.Cube):
        return monthly_average_data(data.data, time_coord=data.coord("time"))

    assert time_coord is not None, "time_coord required for non-cubes."
    dummy_cube = iris.cube.Cube(data, dim_coords_and_dims=[(time_coord, 0)])

    if (
        time_coord.cell(-1).point.month == 1
        and time_coord.cell(-1).point.day == 1
        and time_coord.cell(-2).point.month == 12
    ):
        # Trim the last point.
        dummy_cube = dummy_cube[:-1]

    add_year(dummy_cube, "time")
    add_month_number(dummy_cube, "time")

    avg_cube = dummy_cube.aggregated_by(("year", "month_number"), iris.analysis.MEAN)
    return avg_cube.data


@njit(nogil=True, cache=True)
def moving_sum(data, samples):
    """Calculates a moving sum over the first axis.

    Including the current sample (at a given index), this takes into account `samples`
    previous samples.

    """
    out = np.zeros_like(data)

    for i in range(out.shape[0]):
        out[i] = np.sum(data[max(i - samples + 1, 0) : i + 1], axis=0)

    return out


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
    print(f"Initial guess: {guess:0.1e}, final factor: {factor:0.1e}.")
    return factor


def expand_pft_params(
    params, pft_groups=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10), (11, 12))
):
    """Given N values in `params`, repeat these according to `pft_groups`."""
    if len(params) != len(pft_groups):
        raise ValueError("There should be as many 'params' as 'pft_groups'.")

    if len(set(e for elements in pft_groups for e in elements)) != 13:
        raise ValueError("All 13 PFTs should have a unique entry in 'pft_groups'.")

    out = np.zeros(13, dtype=np.float64)
    for param, pft_group in zip(params, pft_groups):
        for e in pft_group:
            out[e] = param

    return out
