# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose

from python_inferno.utils import exponential_average, temporal_nearest_neighbour_interp


def test_temporal_nearest_neighbour_interp_0d():
    initial = np.array([0, 1, 2])
    interpolated = temporal_nearest_neighbour_interp(initial, 3)
    assert_allclose(
        np.array([0, 0, 1, 1, 1, 2, 2, 2, 2]),
        interpolated.data,
    )
    assert interpolated.mask == False

    initial = np.array([10, 11, 12])
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]),
        temporal_nearest_neighbour_interp(initial, 3).data,
    )


def test_temporal_nearest_neighbour_interp_Nd():
    initial = np.array([10, 11, 12]).reshape(3, 1)
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]).reshape(9, 1),
        temporal_nearest_neighbour_interp(initial, 3).data,
    )


def test_temporal_nearest_neighbour_interp_Nd_mask():
    initial = np.ma.MaskedArray(
        np.array([10, 11, 12]).reshape(3, 1), mask=np.array([0, 1, 0], dtype=np.bool_)
    )
    interpolated = temporal_nearest_neighbour_interp(initial, 3)
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
