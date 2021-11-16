# -*- coding: utf-8 -*-
import os
from datetime import datetime
from functools import reduce, wraps

import iris
import joblib
import numpy as np
from dateutil.relativedelta import relativedelta
from iris.coord_categorisation import add_month_number, add_year
from iris.time import PartialDateTime as IrisPartialDateTime
from loguru import logger
from numba import njit, prange
from wildfires.cache.hashing import PartialDateTimeHasher
from wildfires.utils import ensure_datetime

from .cache import mark_dependency
from .configuration import (
    N_pft_groups,
    npft,
    pft_groups,
    pft_groups_array,
    pft_groups_lengths,
)

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


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _cons_avg(Nt, Nout, weights, in_data, in_mask, out_data, out_mask, cum_weights):
    assert (
        len(in_data.shape)
        == len(in_mask.shape)
        == len(cum_weights.shape)
        == len(out_data.shape)
        == len(out_mask.shape)
        == 2
    )

    # Take weighted mean considering masked data.
    for i in range(Nt):
        for j in range(Nout):
            weight = weights[i, j]
            if weight < 1e-9:
                # Skip negligible (or 0) weights.
                continue

            selection = ~in_mask[i]
            for k in range(selection.shape[0]):
                if selection[k]:
                    out_data[j, k] += in_data[i, k] * weight
                    out_mask[j, k] = False
                    cum_weights[j, k] += weight

    cum_weights_close = cum_weights < 1e-9
    assert np.sum(cum_weights_close) == np.sum(cum_weights_close & out_mask)

    for i in prange(out_data.shape[0]):
        for j in range(out_data.shape[1]):
            if not out_mask[i, j]:
                out_data[i, j] /= cum_weights[i, j]

    return out_data, out_mask


@mark_dependency
def monthly_average_data(data, time_coord=None, trim_single=True, conservative=False):
    """Calculate monthly average of data.

    If `trim_single` is True, a single day e.g. (01/01/2000) at the end of the input
    data will be discarded to avoid generating an 'average' month from a single sample
    only.

    `conservative` averaging will take into account the bounds of `time_coord`.

    """
    if isinstance(data, iris.cube.Cube):
        data = data.data
        time_coord = data.coord("time")

    assert time_coord is not None, "time_coord is required"
    assert time_coord.shape[0] == data.shape[0]

    dummy_cube = iris.cube.Cube(
        np.ma.MaskedArray(np.ma.getdata(data), mask=np.ma.getmaskarray(data)),
        dim_coords_and_dims=[(time_coord, 0)],
    )

    last_datetimes = time_coord.units.num2date(time_coord.points[-2:])

    if (
        last_datetimes[-1].month == 1
        and last_datetimes[-1].day == 1
        and last_datetimes[-2].month == 12
    ):
        # Trim the last point.
        dummy_cube = dummy_cube[:-1]
        # Need to consider this for conservative averaging.
        time_coord = time_coord[:-1]

    if not conservative:
        add_year(dummy_cube, "time")
        add_month_number(dummy_cube, "time")

        avg_cube = dummy_cube.aggregated_by(
            ("year", "month_number"), iris.analysis.MEAN
        )
        return avg_cube.data

    logger.debug("Starting conservative temporal averaging.")

    data = dummy_cube.data
    Nt = data.shape[0]

    # For conservative averaging, take into account the bounds on the temporal coord.
    assert time_coord.bounds is not None, "bounds are required"
    assert time_coord.bounds.shape == (Nt, 2)

    # Pre-calculate bound datetimes to save time.
    bound_dts = time_coord.units.num2date(time_coord.bounds)

    first_date = bound_dts[0][0]
    last_date = bound_dts[-1][1]

    lower_dates = [datetime(first_date.year, first_date.month, 1)]
    while not (
        (lower_dates[-1].year == last_date.year)
        and (lower_dates[-1].month == last_date.month)
    ):
        lower_dates.append(lower_dates[-1] + relativedelta(months=1))

    upper_dates = lower_dates.copy()
    upper_dates.append(upper_dates[-1] + relativedelta(months=1))
    upper_dates.pop(0)
    assert len(upper_dates) == len(lower_dates)
    # NOTE: Temporal bins are contiguous, i.e. the end of one bin is the beginning of
    # the next, at year, month, 1, 0, ...

    Nout = len(lower_dates)
    logger.debug(f"Input shape: {dummy_cube.shape}, nr. of output months: {Nout}.")

    # Calculate overlaps between bounds and the bins given by
    # lower_dates[i], upper_dates[i]
    weights = np.zeros((Nt, Nout))

    cell_bounds = []
    for i in range(Nt):
        bounds = bound_dts[i]
        cell_bounds.append(
            # NOTE: Will this ignore the specifics of different calendars?
            # (comparing datetime with real_datetime, etc...).
            tuple(map(ensure_datetime, bounds))
        )
    for i in range(Nt):
        for (j, (lower_bin, upper_bin)) in enumerate(zip(lower_dates, upper_dates)):
            lower_bound, upper_bound = cell_bounds[i]

            if (lower_bin >= upper_bound) or (upper_bin <= lower_bound):
                weights[i, j] = 0.0
            else:
                weights[i, j] = (
                    (upper_bin - lower_bin).total_seconds()
                    - max((lower_bound - lower_bin).total_seconds(), 0)
                    - max((upper_bin - upper_bound).total_seconds(), 0)
                )

    logger.debug("Finished calculating weights.")

    out_data = np.zeros((Nout, *data.shape[1:]))
    out_mask = np.ones(out_data.shape, dtype=np.bool_)
    cum_weights = np.zeros(out_data.shape)

    out_data, out_mask = _cons_avg(
        Nt=Nt,
        Nout=Nout,
        weights=weights,
        in_data=dummy_cube.data.data,
        in_mask=dummy_cube.data.mask,
        out_data=out_data,
        out_mask=out_mask,
        cum_weights=cum_weights,
    )

    if np.all(out_mask[-1]):
        # Ignore the last month, since this is likely an artefact of the final bound
        # being the beginning of the next month (e.g. a bound of (2000, x, 1, 0, 0)
        # for the month x - 1).
        out_data = out_data[:-1]
        out_mask = out_mask[:-1]

    logger.debug("Finished conservative temporal averaging.")

    return np.ma.MaskedArray(out_data, mask=out_mask)


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


def expand_pft_params(params, pft_groups=pft_groups, dtype=np.float64):
    """Given N values in `params`, repeat these according to `pft_groups`."""
    if len(params) != len(pft_groups):
        raise ValueError("There should be as many 'params' as 'pft_groups'.")

    # Verify the integrity of `pft_groups`.
    if len(set(e for elements in pft_groups for e in elements)) != npft:
        raise ValueError(f"All {npft} PFTs should have a unique entry in 'pft_groups'.")

    out = np.zeros(npft, dtype=dtype)
    for param, pft_group in zip(params, pft_groups):
        for e in pft_group:
            out[e] = param

    return out


def var_shifts_check(*, shifts, variable, data_dict):
    if len(shifts) != N_pft_groups:
        raise ValueError(
            f"'{len(shifts)}' shifts specified for variable '{variable}'. "
            f"Expected '{N_pft_groups}'."
        )
    if variable not in data_dict:
        raise ValueError(f"Variable '{variable}' was not found in the data.")

    if len(data_dict[variable].shape) == 3:
        assert data_dict[variable].shape[1] in (npft, N_pft_groups)


def _pft_dim_processing(*, shifts, variable, data_dict):
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

    # Expand `shifts` if needed.
    if data_dict[variable].shape[1] == npft:
        shifts = expand_pft_params(shifts, dtype=np.int64)

    return data_dict, shifts


def shift_climatology_data(*, antecedent_shifts_dict, data_dict, time_coord):
    """Carry out antecedent shifting of climatological data."""
    # Shift the required variables.
    for variable, shifts in antecedent_shifts_dict.items():
        var_shifts_check(shifts=shifts, variable=variable, data_dict=data_dict)

        # Shift the variable.
        data_dict, shifts = _pft_dim_processing(
            shifts=shifts, variable=variable, data_dict=data_dict
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
        var_shifts_check(shifts=shifts, variable=variable, data_dict=data_dict)

        # Shift the variable.
        data_dict, shifts = _pft_dim_processing(
            shifts=shifts, variable=variable, data_dict=data_dict
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


def memoize(func):
    cache = dict()

    if not hasattr(memoize, "active"):
        memoize.active = True

    @wraps(func)
    def wrapped(*args, **kwargs):
        if not memoize.active:
            # If memoization is disabled, simply call the function.
            return func(*args, **kwargs)

        hashed = joblib.hashing.hash(
            (
                joblib.hashing.hash(args),
                joblib.hashing.hash(kwargs),
            )
        )
        if hashed in cache:
            return cache[hashed]
        output = func(*args, **kwargs)
        cache[hashed] = output
        return output

    return wrapped


@njit(nogil=True, cache=True, fastmath=True)
def get_pft_group_index(i):
    """Return the PFT-group index of the input PFT `i`."""
    group_i = 0
    for index in range(N_pft_groups):
        if i <= pft_groups_array[index][pft_groups_lengths[index] - 1]:
            return group_i
        group_i += 1
    return -1


def key_cache(f):
    """Memoization decorator which uses a special keyword argument `cache_key`."""
    cache = dict()

    @wraps(f)
    def cached(*args, cache_key, **kwargs):
        if cache_key in cache:
            return cache[cache_key]
        out = f(*args, **kwargs)
        cache[cache_key] = out
        return out

    return cached


def dict_match(a, b, rtol=1e-5, atol=1e-8):
    """Determine if two dictionaries are identical, up to floating point precision."""
    if a.keys() != b.keys():
        return False

    for key in a:
        if not np.isclose(a[key], b[key], rtol=rtol, atol=atol):
            return False

    return True


@njit(nogil=True, cache=True)
def linspace_no_endpoint(start, stop, n):
    """Equivalent to `np.linspace(start, stop, n, endpoint=False)`."""
    step = (stop - start) / n
    return start + np.arange(n) * step


def wrap_phase_diffs(x):
    """Wrap phase differences.

    Assumes phase differences (in months) for climatological are cyclical (mod 12).

    """
    x = np.asarray(x)
    return ((x + 6) % 12) - 6
