# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose

from python_inferno.utils import temporal_nearest_neighbour_interp


def test_temporal_nearest_neighbour_interp_0d():
    initial = np.array([0, 1, 2])
    assert_allclose(
        np.array([0, 0, 1, 1, 1, 2, 2, 2, 2]),
        temporal_nearest_neighbour_interp(initial, 3),
    )

    initial = np.array([10, 11, 12])
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]),
        temporal_nearest_neighbour_interp(initial, 3),
    )


def test_temporal_nearest_neighbour_interp_Nd():
    initial = np.array([10, 11, 12]).reshape(3, 1)
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]).reshape(9, 1),
        temporal_nearest_neighbour_interp(initial, 3),
    )
