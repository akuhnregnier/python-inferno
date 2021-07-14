# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose

from python_inferno.utils import (
    expand_pft_params,
    exponential_average,
    moving_sum,
    temporal_nearest_neighbour_interp,
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
