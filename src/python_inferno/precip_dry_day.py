# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
from wildfires.utils import parallel_njit

from .cache import mark_dependency
from .utils import moving_sum

day_s = 24 * 60 * 60


@njit(nogil=True, cache=True)
def filter_rain(ls_rain, con_rain):
    """Combine `ls_rain` and `con_rain` and filter unphysical values."""
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

    return ls_rain_filtered + con_rain_filtered


@njit(nogil=True, cache=True)
@mark_dependency
def precip_moving_sum(ls_rain, con_rain, timestep):
    """Calculate a moving sum of precipitation.

    Note that the precipitation rate is converted to total precipitation by
    multiplying by the timestep.

    Args:
        ls_rain (array with shape (timesteps, land_pts)):
        con_rain (array with shape (timesteps, land_pts)):
        timestep (int): Timestep between samples in seconds.

    Returns:
        rain_mov_sum (array with shape (timesteps, land_pts)): Moving sum of
            precipitation.

    Raises:
        ValueError: If a day is not divisible by the timestep.

    """
    assert len(ls_rain.shape) == 2, "Expect shape (timesteps, land_pts)."
    assert ls_rain.shape == con_rain.shape, "Expect equal shapes."

    if (day_s % timestep) != 0:
        raise ValueError("A day is not divisible by the timestep.")

    # Number of timesteps per day.
    day_timesteps = round(day_s / timestep)

    inferno_rain = filter_rain(ls_rain=ls_rain, con_rain=con_rain)

    return moving_sum(inferno_rain, day_timesteps) * timestep


@parallel_njit(cache=True)
@mark_dependency
def calculate_inferno_dry_days(ls_rain, con_rain, threshold, timestep):
    """INFERNO dry-day period calculation across timesteps for comparison.

    Calculation uses the daily cumulative moving sum of precipitation.

    This calculation assumes that the first timestep is the first timestep of a day.

    TODO - Respect local timezones?

    Args:
        ls_rain (array with shape (timesteps, land_pts)):
        con_rain (array with shape (timesteps, land_pts)):
        threshold (float): Precipitation threshold above which cumulative daily
            precipitation disrupts a dry-day period.
        timestep (int): Timestep between samples in seconds.

    Returns:
        dry_day_period (array with shape (timesteps, land_pts)): Dry-day period in
            units of days.

    Raises:
        ValueError: If a day is not divisible by the timestep.

    """
    precip_sum = precip_moving_sum(ls_rain, con_rain, timestep)
    dry_days = np.zeros(ls_rain.shape, dtype=np.float64)

    # Iterate over locations.
    for l in prange(ls_rain.shape[1]):
        # Iterate over timesteps.
        for i in range(ls_rain.shape[0]):
            if precip_sum[i, l] > threshold:
                # If the daily precipitation has exceeded the threshold, reset the dry
                # days.
                dry_days[i, l] = 0
            else:
                # Accumulate dry days otherwise.
                dry_days[i, l] = dry_days[i - 1, l] + timestep

    # Return dry days in units of days.
    return dry_days / day_s
