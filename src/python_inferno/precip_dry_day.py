# -*- coding: utf-8 -*-
import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def calculate_inferno_dry_days(ls_rain, con_rain, threshold, timestep=1):
    """INFERNO dry-day period calculation across timesteps for comparison.

    Args:
        ls_rain (array with shape (timesteps, land_pts)):
        con_rain (array with shape (timesteps, land_pts)):
        threshold (float): Precipitation threshold above which a precipitation event
            disrupts a dry-day period.
        timestep (int): Timestep between samples in seconds.

    Returns:
        dry_day_period (array with shape (timesteps, land_pts)): Dry-day period in
            units of days.

    """
    assert len(ls_rain.shape) == 2, "Expect shape (timesteps, land_pts)."
    assert ls_rain.shape == con_rain.shape, "Expect equal shapes."

    # Tolerance number to filter non-physical rain values
    rain_tolerance = 1.0e-18  # kg/m2/s

    ls_rain_filtered = ls_rain.copy()
    con_rain_filtered = con_rain.copy()

    for i in range(ls_rain_filtered.size):
        if ls_rain_filtered.ravel()[i] < rain_tolerance:
            ls_rain_filtered.ravel()[i] = 0.0

    for i in range(con_rain_filtered.size):
        if con_rain_filtered.ravel()[i] < rain_tolerance:
            con_rain_filtered.ravel()[i] = 0.0

    inferno_rain = ls_rain_filtered + con_rain_filtered

    dry_days = np.zeros(ls_rain.shape, dtype=np.float64)

    # Iterate over timesteps.
    for i in range(ls_rain.shape[0]):
        for l in range(ls_rain.shape[1]):
            if inferno_rain[i, l] > threshold:
                # Reset dry days above threshold.
                dry_days[i, l] = 0
            else:
                # Accumulate dry days below threshold.
                dry_days[i, l] = dry_days[i - 1, l] + 1

    return dry_days * timestep / (24 * 60 * 60)
