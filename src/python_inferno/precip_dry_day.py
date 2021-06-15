# -*- coding: utf-8 -*-
import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def calculate_inferno_dry_days(ls_rain, con_rain, threshold, timestep=1):
    """INFERNO dry-day period calculation across timesteps for comparison.

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
    assert len(ls_rain.shape) == 2, "Expect shape (timesteps, land_pts)."
    assert ls_rain.shape == con_rain.shape, "Expect equal shapes."

    day_s = 24 * 60 * 60

    if (day_s % timestep) != 0:
        raise ValueError("A day is not divisible by the timestep.")

    # Number of timesteps per day.
    day_timesteps = round(day_s / timestep)

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

    # Iterate over locations.
    for l in range(ls_rain.shape[1]):
        # Iterate over timesteps.
        timestep_counter = 0
        daily_rain = 0.0

        for i in range(ls_rain.shape[0]):
            if timestep_counter % day_timesteps == 0:
                # Reset the accumulated rain every day.
                daily_rain = 0.0

            # Add precipitation to the daily counter.
            daily_rain += inferno_rain[i, l] * timestep

            if daily_rain > threshold:
                # If the daily precipitation has exceeded the threshold, reset the dry
                # days.
                dry_days[i, l] = 0
            else:
                # Accumulate dry days otherwise.
                dry_days[i, l] = dry_days[i - 1, l] + timestep
            timestep_counter += 1

    # Return dry days in units of days.
    return dry_days / day_s
