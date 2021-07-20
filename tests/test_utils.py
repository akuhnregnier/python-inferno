# -*- coding: utf-8 -*-
from datetime import datetime

import cf_units
import iris
import numpy as np
import pytest
from dateutil.relativedelta import relativedelta
from numpy.testing import assert_allclose

from python_inferno.utils import (
    expand_pft_params,
    exponential_average,
    moving_sum,
    temporal_nearest_neighbour_interp,
    temporal_processing,
)


def test_temporal_nearest_neighbour_interp_0d_centre():
    initial = np.array([10, 11, 12])
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "centre")
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]),
        interpolated.data,
    )
    assert interpolated.mask == False


def test_temporal_nearest_neighbour_interp_0d_start():
    initial = np.array([10, 11, 12])
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "start")
    assert_allclose(
        np.array([10, 10, 10, 11, 11, 11, 12, 12, 12]),
        interpolated.data,
    )
    assert interpolated.mask == False


def test_temporal_nearest_neighbour_interp_0d_end():
    initial = np.array([10, 11, 12])
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "end")
    assert_allclose(
        np.array([10, 11, 11, 11, 12, 12, 12, 12, 12]),
        interpolated.data,
    )
    assert interpolated.mask == False


def test_temporal_nearest_neighbour_interp_Nd():
    initial = np.array([10, 11, 12]).reshape(3, 1)
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]).reshape(9, 1),
        temporal_nearest_neighbour_interp(initial, 3, "centre").data,
    )


def test_temporal_nearest_neighbour_interp_Nd_mask():
    initial = np.ma.MaskedArray(
        np.array([10, 11, 12]).reshape(3, 1), mask=np.array([0, 1, 0], dtype=np.bool_)
    )
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "centre")
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]).reshape(9, 1),
        interpolated.data,
    )
    assert np.all(
        interpolated.mask
        == np.array([0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.bool_).reshape(9, 1)
    )


def test_exponential_average_1d():
    data = np.arange(1, 4)
    assert exponential_average(data, 0.1).shape == data.shape
    assert_allclose(exponential_average(data, 1), data)
    assert_allclose(exponential_average(data, 0), np.zeros_like(data))


def test_exponential_average_1d_mask():
    data = np.ma.MaskedArray(np.arange(1, 4), mask=True)
    assert exponential_average(data, 0.1).shape == data.shape
    assert np.all(exponential_average(data, 0.1).mask)

    data = np.ma.MaskedArray(np.arange(1, 4), mask=np.array([0, 1, 0], dtype=np.bool_))
    assert exponential_average(data, 0.1).shape == data.shape
    assert np.all(exponential_average(data, 0.1).mask)


def test_exponential_average_Nd_mask():
    data = np.ma.MaskedArray(
        np.arange(0, 6).reshape(2, 3),
        mask=np.array([[0, 1, 0], [0, 0, 0]], dtype=np.bool_),
    )
    assert exponential_average(data, 0.1).shape == data.shape
    assert np.all(
        exponential_average(data, 0.1).mask
        == np.array([[0, 1, 0], [0, 1, 0]], dtype=np.bool_)
    )

    assert_allclose(exponential_average(data[:, [0, 2]], 1), data[:, [0, 2]])


def test_moving_sum():
    data = np.arange(3)
    assert_allclose(moving_sum(data, 1), data)
    assert_allclose(moving_sum(data, 2), [0, 1, 3])
    assert_allclose(moving_sum(data, 3), [0, 1, 3])

    data = np.arange(5)
    assert_allclose(moving_sum(data, 1), data)
    assert_allclose(moving_sum(data, 2), [0, 1, 3, 5, 7])
    assert_allclose(moving_sum(data, 3), [0, 1, 3, 6, 9])

    assert_allclose(moving_sum(data[:, None], 3), np.array([0, 1, 3, 6, 9])[:, None])


def test_expand_pft_params():
    assert tuple(expand_pft_params([1, 2, 3])) == (
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
    )


def get_daily_time_coord(n):
    units = cf_units.Unit("days since 1970-01-01", calendar="gregorian")
    num_dates = units.date2num(
        [datetime(1970, 1, 1) + relativedelta(days=days) for days in range(n)]
    )
    return iris.coords.DimCoord(
        num_dates, standard_name="time", var_name="time", units=units
    )


def test_temporal_processing_no_agg():
    data_dict = {
        "a": np.arange(10 * 13 * 10).reshape(10, 13, 10),
        "b": np.arange(10 * 10).reshape(10, 10),
    }
    shifts = expand_pft_params([1, 2, 3]).astype("int64")
    antecedent_shifts_dict = {"a": shifts}
    out, time_coord = temporal_processing(
        data_dict=data_dict.copy(),
        antecedent_shifts_dict=antecedent_shifts_dict,
        average_samples=0,
        time_coord=get_daily_time_coord(10),
    )
    assert np.allclose(out["a"][:, :5], data_dict["a"][2:-1, :5])
    assert np.allclose(out["a"][:, 5:11], data_dict["a"][1:-2, 5:11])
    assert np.allclose(out["a"][:, 11:13], data_dict["a"][:-3, 11:13])
    assert np.allclose(out["b"], data_dict["b"][3:])

    assert time_coord.cell(0).point == datetime(1970, 1, 4)
    assert time_coord.cell(-1).point == datetime(1970, 1, 10)


@pytest.mark.parametrize(
    "aggregator, agg_method",
    [
        (iris.analysis.MEAN, np.mean),
        (iris.analysis.MIN, np.min),
        (iris.analysis.MAX, np.max),
    ],
)
def test_temporal_processing_agg(aggregator, agg_method):
    data_dict = {
        "a": np.arange(10 * 13 * 10).reshape(10, 13, 10),
        "b": np.arange(10 * 10).reshape(10, 10),
    }
    shifts = expand_pft_params([1, 2, 3]).astype("int64")
    antecedent_shifts_dict = {"a": shifts}
    out, time_coord = temporal_processing(
        data_dict=data_dict.copy(),
        antecedent_shifts_dict=antecedent_shifts_dict,
        average_samples=2,
        aggregator=aggregator,
        time_coord=get_daily_time_coord(10),
    )
    assert np.allclose(out["a"][0, :5], agg_method(data_dict["a"][2:4, :5], axis=0))
    assert np.allclose(out["a"][0, 5:11], agg_method(data_dict["a"][1:3, 5:11], axis=0))
    assert np.allclose(
        out["a"][0, 11:13], agg_method(data_dict["a"][:2, 11:13], axis=0)
    )
    assert np.allclose(out["b"][0], agg_method(data_dict["b"][3:5], axis=0))

    assert_allclose(time_coord.points[0], 3.5)
    assert_allclose(time_coord.points[-1], 9)
    assert time_coord.cell(0).bound == (datetime(1970, 1, 4), datetime(1970, 1, 5))
    assert time_coord.cell(-1).bound == (datetime(1970, 1, 10), datetime(1970, 1, 10))


def test_temporal_processing_multi_agg():
    data_dict = {
        "a": np.arange(10 * 13 * 10).reshape(10, 13, 10),
        "b": np.arange(10 * 10).reshape(10, 10),
    }
    shifts = expand_pft_params([1, 2, 3]).astype("int64")
    antecedent_shifts_dict = {"a": shifts}
    out, time_coord = temporal_processing(
        data_dict=data_dict.copy(),
        antecedent_shifts_dict=antecedent_shifts_dict,
        average_samples=2,
        aggregator={"a": iris.analysis.MIN, "b": iris.analysis.MAX},
        time_coord=get_daily_time_coord(10),
    )
    assert np.allclose(out["a"][0, :5], np.min(data_dict["a"][2:4, :5], axis=0))
    assert np.allclose(out["a"][0, 5:11], np.min(data_dict["a"][1:3, 5:11], axis=0))
    assert np.allclose(out["a"][0, 11:13], np.min(data_dict["a"][:2, 11:13], axis=0))
    assert np.allclose(out["b"][0], np.max(data_dict["b"][3:5], axis=0))

    assert_allclose(time_coord.points[0], 3.5)
    assert_allclose(time_coord.points[-1], 9)
    assert time_coord.cell(0).bound == (datetime(1970, 1, 4), datetime(1970, 1, 5))
    assert time_coord.cell(-1).bound == (datetime(1970, 1, 10), datetime(1970, 1, 10))
