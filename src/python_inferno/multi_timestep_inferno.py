# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .calc_c_comps_triffid import calc_c_comps_triffid
from .configuration import avg_ba, land_pts, npft
from .inferno import calc_burnt_area, calc_flam, calc_ignitions
from .qsat_wat import qsat_wat
from .utils import get_pft_group_index, transform_dtype

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _get_diagnostics(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    ls_rain,
    con_rain,
):
    Nt = pstar.shape[0]

    # Output arrays.
    inferno_rain_out = np.zeros((Nt, land_pts), dtype=np.float32)
    qsat_out = np.zeros((Nt, npft, land_pts), dtype=np.float32)
    inferno_rhum_out = np.zeros((Nt, npft, land_pts), dtype=np.float32)

    # Tolerance number to filter non-physical rain values
    rain_tolerance = 1.0e-18  # kg/m2/s

    for l in prange(0, land_pts):
        for ti in range(Nt):
            ls_rain_filtered = ls_rain[ti, l]
            con_rain_filtered = con_rain[ti, l]

            if ls_rain_filtered < rain_tolerance:
                ls_rain_filtered = 0.0
            if con_rain_filtered < rain_tolerance:
                con_rain_filtered = 0.0

            inferno_rain = ls_rain_filtered + con_rain_filtered

            inferno_rain_out[ti, l] = inferno_rain

            for i in range(npft):
                # Get the tile relative humidity using saturation routine
                qsat = qsat_wat(t1p5m_tile[ti, i, l], pstar[ti, l])
                qsat_out[ti, i, l] = qsat

                inferno_rhum = (q1p5m_tile[ti, i, l] / qsat) * 100.0
                inferno_rhum_out[ti, i, l] = inferno_rhum

    return inferno_rain_out, qsat_out, inferno_rhum_out


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _get_checks_failed_mask(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    sthu_soilt_single,
    ls_rain,
    con_rain,
):
    # Ensure consistency of the time dimension.
    if not (
        t1p5m_tile.shape[0]
        == q1p5m_tile.shape[0]
        == pstar.shape[0]
        == sthu_soilt_single.shape[0]
        == ls_rain.shape[0]
        == con_rain.shape[0]
    ):
        raise ValueError("All arrays need to have the same time dimension.")

    Nt = pstar.shape[0]

    # Ignore mask stored for diagnostic purposes.
    checks_failed_mask = np.ones((Nt, npft, land_pts), dtype=np.bool_)

    # Tolerance number to filter non-physical rain values
    rain_tolerance = 1.0e-18  # kg/m2/s

    # Soil Humidity (inferno_sm)
    inferno_sm = sthu_soilt_single

    for l in prange(land_pts):
        for ti in range(Nt):
            # Rainfall (inferno_rain)

            # Rain fall values have a significant impact in the calculation of flammability.
            # In some cases we may be presented with values that have no significant meaning -
            # e.g in the UM context negative values or very small values can often be found/

            ls_rain_filtered = ls_rain[ti, l]
            con_rain_filtered = con_rain[ti, l]

            if ls_rain_filtered < rain_tolerance:
                ls_rain_filtered = 0.0
            if con_rain_filtered < rain_tolerance:
                con_rain_filtered = 0.0

            inferno_rain = ls_rain_filtered + con_rain_filtered

            # The maximum rain rate ever observed is 38mm in one minute,
            # here we assume 0.5mm/s stops fires altogether
            if (inferno_rain > 0.5) or (inferno_rain < 0.0):
                continue

            # Soil moisture is a fraction of saturation
            if (inferno_sm[ti, l] > 1.0) or (inferno_sm[ti, l] < 0.0):
                continue

            for i in range(npft):
                # Conditional statements to make sure we are dealing with
                # reasonable weather. Note initialisation to 0 already done.
                # If the driving variables are singularities, we assume
                # no burnt area.

                # Temperatures constrained akin to qsat (from the WMO)
                if (t1p5m_tile[ti, i, l] > 338.15) or (t1p5m_tile[ti, i, l] < 183.15):
                    continue

                # Get the tile relative humidity using saturation routine
                qsat = qsat_wat(t1p5m_tile[ti, i, l], pstar[ti, l])

                inferno_rhum = (q1p5m_tile[ti, i, l] / qsat) * 100.0

                # Relative Humidity should be constrained to 0-100
                if (inferno_rhum > 100.0) or (inferno_rhum < 0.0):
                    continue

                # Record the fact that the checks have passed.
                checks_failed_mask[ti, i, l] = False

    return checks_failed_mask


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _multi_timestep_inferno(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    sthu_soilt_single,
    frac,
    c_soil_dpm_gb,
    c_soil_rpm_gb,
    canht,
    ls_rain,
    con_rain,
    pop_den,
    flash_rate,
    ignition_method,
    fuel_build_up,
    fapar_diag_pft,
    grouped_dry_bal,
    dry_days,
    flammability_method,
    dryness_method,
    litter_pool,
    fuel_build_up_method,
    include_temperature,
    fapar_factor,
    fapar_centre,
    fapar_shape,
    fuel_build_up_factor,
    fuel_build_up_centre,
    fuel_build_up_shape,
    temperature_factor,
    temperature_centre,
    temperature_shape,
    litter_pool_factor,
    litter_pool_centre,
    litter_pool_shape,
    dry_day_factor,
    dry_day_centre,
    dry_day_shape,
    dry_bal_factor,
    dry_bal_centre,
    dry_bal_shape,
    fapar_weight,
    dryness_weight,
    temperature_weight,
    fuel_weight,
    land_point,
    checks_failed,
):
    # Ensure consistency of the time dimension.
    if not (
        t1p5m_tile.shape[0]
        == q1p5m_tile.shape[0]
        == pstar.shape[0]
        == sthu_soilt_single.shape[0]
        == frac.shape[0]
        == c_soil_dpm_gb.shape[0]
        == c_soil_rpm_gb.shape[0]
        == canht.shape[0]
        == ls_rain.shape[0]
        == con_rain.shape[0]
        == fuel_build_up.shape[0]
        == fapar_diag_pft.shape[0]
    ):
        raise ValueError("All arrays need to have the same time dimension.")

    Nt = pstar.shape[0]

    # Store the output BA (averaged over PFTs).
    burnt_area = np.zeros((Nt, land_pts))

    # Plant Material that is available as fuel (on the surface)
    pmtofuel = 0.7

    # Fuel availability high/low threshold
    fuel_low = 0.02
    fuel_high = 0.2
    fuel_diff = fuel_high - fuel_low

    # Tolerance number to filter non-physical rain values
    rain_tolerance = 1.0e-18  # kg/m2/s

    # Get the available DPM and RPM using a scaling parameter
    dpm_fuel = pmtofuel * c_soil_dpm_gb

    # Soil Humidity (inferno_sm)
    inferno_sm = sthu_soilt_single

    if land_point == -1:
        loop_start = 0
        loop_end = land_pts
    else:
        loop_start = land_point
        loop_end = land_point + 1
    for l in prange(loop_start, loop_end):
        for ti in range(Nt):
            # Rainfall (inferno_rain)

            # Rain fall values have a significant impact in the calculation of flammability.
            # In some cases we may be presented with values that have no significant meaning -
            # e.g in the UM context negative values or very small values can often be found/

            ls_rain_filtered = ls_rain[ti, l]
            con_rain_filtered = con_rain[ti, l]

            if ls_rain_filtered < rain_tolerance:
                ls_rain_filtered = 0.0
            if con_rain_filtered < rain_tolerance:
                con_rain_filtered = 0.0

            inferno_rain = ls_rain_filtered + con_rain_filtered

            for i in range(npft):
                if checks_failed[ti, i, l]:
                    continue

                pft_group_i = get_pft_group_index(i)

                # Diagnose the balanced-growth leaf area index and the carbon
                # contents of leaves and wood.
                leaf_inf = calc_c_comps_triffid(i, canht[ti, i, l])[1]

                # Calculate the fuel density
                # We use normalised Leaf Carbon + the available DPM
                inferno_fuel = (leaf_inf + dpm_fuel[ti, l] - fuel_low) / (fuel_diff)

                if inferno_fuel < 0.0:
                    inferno_fuel = 0.0
                elif inferno_fuel > 1.0:
                    inferno_fuel = 1.0

                # Get the tile relative humidity using saturation routine
                qsat = qsat_wat(t1p5m_tile[ti, i, l], pstar[ti, l])

                inferno_rhum = (q1p5m_tile[ti, i, l] / qsat) * 100.0

                ignitions = calc_ignitions(
                    pop_den[l],
                    flash_rate[l],
                    ignition_method,
                )

                flammability_ft = calc_flam(
                    t1p5m_tile[ti, i, l],
                    inferno_rhum,
                    inferno_fuel,
                    inferno_sm[ti, l],
                    inferno_rain,
                    fuel_build_up[ti, i, l],
                    fapar_diag_pft[ti, i, l],
                    dry_days[ti, l],
                    flammability_method,
                    dryness_method,
                    fuel_build_up_method,
                    fapar_factor[pft_group_i],
                    fapar_centre[pft_group_i],
                    fapar_shape[pft_group_i],
                    fuel_build_up_factor[pft_group_i],
                    fuel_build_up_centre[pft_group_i],
                    fuel_build_up_shape[pft_group_i],
                    temperature_factor[pft_group_i],
                    temperature_centre[pft_group_i],
                    temperature_shape[pft_group_i],
                    dry_day_factor[pft_group_i],
                    dry_day_centre[pft_group_i],
                    dry_day_shape[pft_group_i],
                    grouped_dry_bal[ti, pft_group_i, l],
                    dry_bal_factor[pft_group_i],
                    dry_bal_centre[pft_group_i],
                    dry_bal_shape[pft_group_i],
                    litter_pool[ti, i, l],
                    litter_pool_factor[pft_group_i],
                    litter_pool_centre[pft_group_i],
                    litter_pool_shape[pft_group_i],
                    include_temperature,
                    fapar_weight[pft_group_i],
                    dryness_weight[pft_group_i],
                    temperature_weight[pft_group_i],
                    fuel_weight[pft_group_i],
                )

                burnt_area_ft = calc_burnt_area(flammability_ft, ignitions, avg_ba[i])

                # We add pft-specific variables to the gridbox totals
                burnt_area[ti, l] += frac[ti, i, l] * burnt_area_ft

    return burnt_area


def multi_timestep_inferno(
    *,
    overall_scale,
    land_point=-1,
    **kwargs,
):
    raw_ba = transform_dtype(_multi_timestep_inferno)(
        land_point=land_point,
        **kwargs,
    )
    return overall_scale * raw_ba
