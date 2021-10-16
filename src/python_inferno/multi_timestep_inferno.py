# -*- coding: utf-8 -*-
import numpy as np
from loguru import logger
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .calc_c_comps_triffid import calc_c_comps_triffid
from .configuration import N_pft_groups, avg_ba, land_pts, npft
from .inferno import calc_burnt_area, calc_flam, calc_ignitions
from .precip_dry_day import precip_moving_sum
from .qsat_wat import qsat_wat
from .utils import get_pft_group_index

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


def multi_timestep_inferno(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    sthu_soilt,
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
    litter_pool,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    litter_pool_factor,
    litter_pool_centre,
    flammability_method,
    dryness_method,
    fuel_build_up_method,
    dry_day_factor,
    dry_day_centre,
    dry_bal_factor,
    dry_bal_centre,
    include_temperature,
    timestep,
):
    param_vars = dict(
        fapar_factor=fapar_factor,
        fapar_centre=fapar_centre,
        fuel_build_up_factor=fuel_build_up_factor,
        fuel_build_up_centre=fuel_build_up_centre,
        temperature_factor=temperature_factor,
        temperature_centre=temperature_centre,
        dry_day_factor=dry_day_factor,
        dry_day_centre=dry_day_centre,
        dry_bal_factor=dry_bal_factor,
        dry_bal_centre=dry_bal_centre,
        litter_pool_factor=litter_pool_factor,
        litter_pool_centre=litter_pool_centre,
    )

    # Ensure the parameters are given as arrays with `N_pft_groups` elements.
    transformed_param_vars = dict()
    for name, val in param_vars.items():
        if not hasattr(val, "__iter__"):
            logger.debug(f"Duplicating: {name}")
            val = [val] * N_pft_groups
        transformed_param_vars[name] = np.asarray(val, dtype=np.float64)
        assert transformed_param_vars[name].shape == (N_pft_groups,)

    # Call the below using normal, non-numba Python to enable features like
    # keyword-only arguments with default arguments as above.
    ba = _multi_timestep_inferno(
        t1p5m_tile=t1p5m_tile,
        q1p5m_tile=q1p5m_tile,
        pstar=pstar,
        sthu_soilt=sthu_soilt,
        frac=frac,
        c_soil_dpm_gb=c_soil_dpm_gb,
        c_soil_rpm_gb=c_soil_rpm_gb,
        canht=canht,
        ls_rain=ls_rain,
        con_rain=con_rain,
        pop_den=pop_den,
        flash_rate=flash_rate,
        ignition_method=ignition_method,
        fuel_build_up=fuel_build_up,
        fapar_diag_pft=fapar_diag_pft,
        grouped_dry_bal=grouped_dry_bal,
        dry_days=dry_days,
        flammability_method=flammability_method,
        dryness_method=dryness_method,
        cum_rain=precip_moving_sum(
            ls_rain=ls_rain, con_rain=con_rain, timestep=timestep
        ),
        litter_pool=litter_pool,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        **transformed_param_vars,
    )
    return ba


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def _multi_timestep_inferno(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    sthu_soilt,
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
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    litter_pool_factor,
    litter_pool_centre,
    flammability_method,
    dryness_method,
    fuel_build_up_method,
    dry_day_factor,
    dry_day_centre,
    dry_bal_factor,
    dry_bal_centre,
    cum_rain,
    litter_pool,
    include_temperature,
):
    # Ensure consistency of the time dimension.
    if not (
        t1p5m_tile.shape[0]
        == q1p5m_tile.shape[0]
        == pstar.shape[0]
        == sthu_soilt.shape[0]
        == frac.shape[0]
        == c_soil_dpm_gb.shape[0]
        == c_soil_rpm_gb.shape[0]
        == canht.shape[0]
        == ls_rain.shape[0]
        == con_rain.shape[0]
        == fuel_build_up.shape[0]
        == fapar_diag_pft.shape[0]
        == cum_rain.shape[0]
    ):
        raise ValueError("All arrays need to have the same time dimension.")

    Nt = pstar.shape[0]

    # Store the output BA (averaged over PFTs).
    burnt_area = np.zeros((Nt, land_pts))
    burnt_area_ft = np.zeros((Nt, npft, land_pts))

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
    # XXX What does selecting one of the 4 layers change here?
    inferno_sm = sthu_soilt[:, 0, 0]

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
                pft_group_i = get_pft_group_index(i)

                # Conditional statements to make sure we are dealing with
                # reasonable weather. Note initialisation to 0 already done.
                # If the driving variables are singularities, we assume
                # no burnt area.

                # Temperatures constrained akin to qsat (from the WMO)
                if (t1p5m_tile[ti, i, l] > 338.15) or (t1p5m_tile[ti, i, l] < 183.15):
                    continue

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

                # Relative Humidity should be constrained to 0-100
                if (inferno_rhum > 100.0) or (inferno_rhum < 0.0):
                    continue

                # If all these checks are passes, start fire calculations

                ignitions = calc_ignitions(
                    pop_den[l],
                    flash_rate[l],
                    ignition_method,
                )

                flammability_ft = calc_flam(
                    temp_l=t1p5m_tile[ti, i, l],
                    rhum_l=inferno_rhum,
                    fuel_l=inferno_fuel,
                    sm_l=inferno_sm[ti, l],
                    rain_l=inferno_rain,
                    cum_rain_l=cum_rain[ti, l],
                    fuel_build_up=fuel_build_up[ti, i, l],
                    fapar=fapar_diag_pft[ti, i, l],
                    dry_days=dry_days[ti, l],
                    flammability_method=flammability_method,
                    dryness_method=dryness_method,
                    fuel_build_up_method=fuel_build_up_method,
                    fapar_factor=fapar_factor[pft_group_i],
                    fapar_centre=fapar_centre[pft_group_i],
                    fuel_build_up_factor=fuel_build_up_factor[pft_group_i],
                    fuel_build_up_centre=fuel_build_up_centre[pft_group_i],
                    temperature_factor=temperature_factor[pft_group_i],
                    temperature_centre=temperature_centre[pft_group_i],
                    dry_day_factor=dry_day_factor[pft_group_i],
                    dry_day_centre=dry_day_centre[pft_group_i],
                    dry_bal=grouped_dry_bal[ti, pft_group_i, l],
                    dry_bal_factor=dry_bal_factor[pft_group_i],
                    dry_bal_centre=dry_bal_centre[pft_group_i],
                    litter_pool=litter_pool[ti, i, l],
                    litter_pool_factor=litter_pool_factor[pft_group_i],
                    litter_pool_centre=litter_pool_centre[pft_group_i],
                    include_temperature=include_temperature,
                )

                burnt_area_ft[ti, i, l] = calc_burnt_area(
                    flammability_ft, ignitions, avg_ba[i]
                )

                # We add pft-specific variables to the gridbox totals
                burnt_area[ti, l] += frac[ti, i, l] * burnt_area_ft[ti, i, l]

    return burnt_area
