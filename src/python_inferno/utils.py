# -*- coding: utf-8 -*-
import numpy as np


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
