# -*- coding: utf-8 -*-
import os
from functools import reduce, wraps

import iris
import numpy as np
from iris.coord_categorisation import add_month_number, add_year
from iris.time import PartialDateTime as IrisPartialDateTime
from numba import njit
from scipy.optimize import minimize
from wildfires.cache.hashing import PartialDateTimeHasher

from .cache import mark_dependency
from .metrics import nme

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm  # noqa
else:
    from tqdm import tqdm  # noqa


class PartialDateTime(IrisPartialDateTime):
    def __hash__(self):
        return int(PartialDateTimeHasher.calculate_hash(self), 16)


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


@mark_dependency
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


def shift_climatology_data(*, antecedent_shifts_dict, data_dict, time_coord):
    """Carry out antecedent shifting of climatological data."""
    # Shift the required variables.
    for variable, shifts in antecedent_shifts_dict.items():
        if len(shifts) != 13:
            raise ValueError(
                f"'{len(shifts)}' shifts specified for variable '{variable}'. "
                "Expected 13."
            )
        if variable not in data_dict:
            raise ValueError(f"Variable '{variable}' was not found in the data.")

        # Shift the variable.

        # Shape is (time, [pft,] space).
        if len(data_dict[variable].shape) == 2:
            # Add a PFT dimensions by replicating the data.
            data_dict[variable] = np.repeat(
                np.expand_dims(data_dict[variable], 1), repeats=len(shifts), axis=1
            )

        if len(data_dict[variable].shape) != 3:
            raise ValueError(
                f"Variable '{variable}' had unexpected shape "
                f"'{data_dict[variable].shape}'."
            )
        # Since we are dealing with climatological data, simply roll the data to shift
        # the samples forward in time.
        data_dict[variable] = np.stack(
            [
                np.roll(data_dict[variable][:, pft_i], shift, axis=0)
                for pft_i, shift in enumerate(shifts)
            ],
            axis=1,
        )

    # Ensure all time coordinates are the same.
    assert len(set(data.shape[0] for data in data_dict.values())) == 1

    return data_dict, time_coord


def shift_data(*, antecedent_shifts_dict, data_dict, time_coord):
    """Carry out antecedent shifting of data."""
    # Determine the maximum shift.
    max_shift = 0
    for shifts in antecedent_shifts_dict.values():
        if np.max(shifts) > max_shift:
            max_shift = np.max(shifts)

    # Shift the required variables.
    for variable, shifts in antecedent_shifts_dict.items():
        if len(shifts) != 13:
            raise ValueError(
                f"'{len(shifts)}' shifts specified for variable '{variable}'. "
                "Expected 13."
            )
        if variable not in data_dict:
            raise ValueError(f"Variable '{variable}' was not found in the data.")

        # Shift the variable.

        # Shape is (time, [pft,] space).
        if len(data_dict[variable].shape) == 2:
            # Add a PFT dimensions by replicating the data.
            data_dict[variable] = np.repeat(
                np.expand_dims(data_dict[variable], 1), repeats=len(shifts), axis=1
            )

        if len(data_dict[variable].shape) != 3:
            raise ValueError(
                f"Variable '{variable}' had unexpected shape "
                f"'{data_dict[variable].shape}'."
            )
        # Remove 'shift' from the end to shift the samples forward in time.
        # Also remove samples from the beginning to ensure all arrays are the same
        # length.
        data_dict[variable] = np.stack(
            [
                data_dict[variable][max_shift - shift : -shift, pft_i]
                for pft_i, shift in enumerate(shifts)
            ],
            axis=1,
        )

    # For all other variables, simply trim off `max_shift` number of elements from the
    # front.
    for variable in set(data_dict) - set(antecedent_shifts_dict):
        data_dict[variable] = data_dict[variable][max_shift:]

    # Ditto for the temporal coord.
    assert time_coord.shape[0] >= max_shift
    time_coord = time_coord[max_shift:]

    # Ensure all time coordinates are the same.
    assert len(set(data.shape[0] for data in data_dict.values())) == 1

    return data_dict, time_coord


@mark_dependency
def temporal_processing(
    *,
    data_dict,
    antecedent_shifts_dict,
    average_samples,
    aggregator="MEAN",
    time_coord,
    climatology_input=False,
):
    # First carry out the antecedent shifting.

    if not climatology_input:
        data_dict, time_coord = shift_data(
            antecedent_shifts_dict=antecedent_shifts_dict,
            data_dict=data_dict,
            time_coord=time_coord,
        )
    else:
        data_dict, time_coord = shift_climatology_data(
            antecedent_shifts_dict=antecedent_shifts_dict,
            data_dict=data_dict,
            time_coord=time_coord,
        )

    if not average_samples or average_samples == 1:
        # If no averaging has been requested, simply return at this point.
        return data_dict, time_coord

    if average_samples < 0:
        raise ValueError(
            f"Expected a positive number of `average_samples`. Got '{average_samples}'."
        )

    def get_average_i():
        return iris.coords.AuxCoord(
            np.floor(
                np.linspace(
                    0,
                    data.shape[0] / average_samples,
                    data.shape[0],
                    endpoint=False,
                )
            ),
            long_name="average_i",
        )

    # Carry out temporal averaging.

    if not isinstance(aggregator, dict):
        # Use the same aggregator for each variable.
        aggregator = {name: aggregator for name in data_dict}

    # Use strings as an intermediary here to enable caching of function input values
    # (iris aggregators are not easily hashable).
    aggregators_map = {
        "MAX": iris.analysis.MAX,
        "MEAN": iris.analysis.MEAN,
        "MEDIAN": iris.analysis.MEDIAN,
        "MIN": iris.analysis.MIN,
    }

    # Choose the corresponding aggregation functions.
    aggregator = {
        name: aggregators_map[agg_name] for name, agg_name in aggregator.items()
    }

    for variable in data_dict:
        data = data_dict[variable]
        cube = iris.cube.Cube(
            data,
            aux_coords_and_dims=[
                (
                    get_average_i(),
                    0,
                )
            ],
        )
        agg_cube = cube.aggregated_by("average_i", aggregator[variable])
        assert np.all(
            agg_cube.coord("average_i").points
            == np.sort(agg_cube.coord("average_i").points)
        )
        data_dict[variable] = agg_cube.data

    def time_points_agg(aggregator):
        return (
            iris.cube.Cube(
                time_coord.points, aux_coords_and_dims=[(get_average_i(), 0)]
            )
            .aggregated_by("average_i", aggregator)
            .data
        )

    time_coord = time_coord.copy(
        points=time_points_agg(iris.analysis.MEAN),
        bounds=np.stack(
            (time_points_agg(iris.analysis.MIN), time_points_agg(iris.analysis.MAX)),
            axis=1,
        ),
    )

    return data_dict, time_coord
