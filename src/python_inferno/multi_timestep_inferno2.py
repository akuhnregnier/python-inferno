# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .calc_c_comps_triffid import calc_c_comps_triffid
from .configuration import avg_ba, land_pts, m2_in_km2, npft, s_in_day, s_in_month
from .precip_dry_day import precip_moving_sum
from .qsat_wat import qsat_wat

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


def multi_timestep_inferno2(
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
    dry_bal=None,
    dry_days,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    flammability_method,
    dryness_method,
    dry_day_factor,
    dry_day_centre,
    rain_f,
    vpd_f,
    dry_bal_factor,
    dry_bal_centre,
    timestep,
    return_dry_bal=False,
):
    if dry_bal is None:
        dry_bal = np.zeros_like(fapar_diag_pft)

    param_vars = dict(
        fapar_factor=fapar_factor,
        fapar_centre=fapar_centre,
        fuel_build_up_factor=fuel_build_up_factor,
        fuel_build_up_centre=fuel_build_up_centre,
        temperature_factor=temperature_factor,
        temperature_centre=temperature_centre,
        dry_day_factor=dry_day_factor,
        dry_day_centre=dry_day_centre,
        rain_f=rain_f,
        vpd_f=vpd_f,
        dry_bal_factor=dry_bal_factor,
        dry_bal_centre=dry_bal_centre,
    )

    # Ensure the parameters are given as arrays with 'npft' elements.
    transformed_param_vars = dict()
    for name, val in param_vars.items():
        if not hasattr(val, "__iter__"):
            print(f"Duplicating: {name}")
            val = [val] * npft
        transformed_param_vars[name] = np.asarray(val, dtype=np.float64)
        assert transformed_param_vars[name].shape == (npft,)

    # Call the below using normal, non-numba Python to enable features like
    # keyword-only arguments with default arguments as above.
    ba, dry_bal = _multi_timestep_inferno(
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
        dry_bal=dry_bal,
        dry_days=dry_days,
        flammability_method=flammability_method,
        dryness_method=dryness_method,
        cum_rain=precip_moving_sum(
            ls_rain=ls_rain, con_rain=con_rain, timestep=timestep
        ),
        **transformed_param_vars,
    )
    if return_dry_bal:
        return ba, dry_bal
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
    dry_bal,
    dry_days,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    flammability_method,
    dryness_method,
    dry_day_factor,
    dry_day_centre,
    rain_f,
    vpd_f,
    dry_bal_factor,
    dry_bal_centre,
    cum_rain,
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

                flammability_ft, dry_bal[ti, i, l] = calc_flam(
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
                    fapar_factor=fapar_factor[i],
                    fapar_centre=fapar_centre[i],
                    fuel_build_up_factor=fuel_build_up_factor[i],
                    fuel_build_up_centre=fuel_build_up_centre[i],
                    temperature_factor=temperature_factor[i],
                    temperature_centre=temperature_centre[i],
                    dry_day_factor=dry_day_factor[i],
                    dry_day_centre=dry_day_centre[i],
                    dry_bal=dry_bal[max(ti - 1, 0), i, l],
                    rain_f=rain_f[i],
                    vpd_f=vpd_f[i],
                    dry_bal_factor=dry_bal_factor[i],
                    dry_bal_centre=dry_bal_centre[i],
                )

                burnt_area_ft[ti, i, l] = calc_burnt_area(
                    flammability_ft, ignitions, avg_ba[i]
                )

                # We add pft-specific variables to the gridbox totals
                burnt_area[ti, l] += frac[ti, i, l] * burnt_area_ft[ti, i, l]

    return burnt_area, dry_bal


@njit(nogil=True, cache=True, fastmath=True)
def calc_ignitions(pop_den_l, flash_rate_l, ignition_method):
    # Description:
    #     Calculate the number of ignitions/m2/s at each gridpoint
    #
    # Method:
    #     See original paper by Pechony and Shindell (2009),
    #     originally proposed for monthly totals, here per timestep.

    # ignition_method
    # The integer defining the method used for ignitions:
    # 1 = constant,
    # 2 = constant (Anthropogenic) + Varying (lightning),
    # 3 = Varying  (Anthropogenic and lightning)

    # flash_rate_l,
    # The Cloud to Ground lightning flash rate (flashes/km2)
    # pop_den_l
    # The population density (ppl/km2)

    # ignitions_l
    # The number of ignitions/m2/s

    # man_ign_l,
    # Human-induced fire ignition rate (ignitions/km2/s)
    # nat_ign_l,
    # Lightning natural ignition rate (number/km2/sec)
    # non_sup_frac_l
    # Fraction of fire ignition non suppressed by humans

    tune_MODIS = 7.7
    # Parameter originally used by P&S (2009) to match MODIS

    if ignition_method == 1:
        # Assume a multi-year annual mean of 2.7/km2/yr
        # (Huntrieser et al. 2007) 75% are Cloud to Ground flashes
        # (Prentice and Mackerras 1977)
        nat_ign_l = 2.7 / s_in_month / m2_in_km2 / 12.0 * 0.75

        # We parameterised 1.5 ignitions/km2/month globally from GFED
        man_ign_l = 1.5 / s_in_month / m2_in_km2

        return man_ign_l + nat_ign_l

    elif ignition_method == 2:
        # Flash Rate (Cloud to Ground) always lead to one fire
        nat_ign_l = min(max(flash_rate_l / m2_in_km2 / s_in_day, 0.0), 1.0)

        # We parameterised 1.5 ignitions/km2/month globally from GFED
        man_ign_l = 1.5 / s_in_month / m2_in_km2

        return man_ign_l + nat_ign_l

    elif ignition_method == 3:
        # Flash Rate (Cloud to Ground) always lead to one fire
        nat_ign_l = flash_rate_l / m2_in_km2 / s_in_day

        man_ign_l = 0.2 * pop_den_l ** (0.4) / m2_in_km2 / s_in_month

        non_sup_frac_l = 0.05 + 0.9 * np.exp(-0.05 * pop_den_l)

        ignitions_l = (nat_ign_l + man_ign_l) * non_sup_frac_l

        # Tune ignitions to MODIS data (see Pechony and Shindell, 2009)
        return ignitions_l * tune_MODIS


@njit(nogil=True, cache=True, fastmath=True)
def fuel_param(x, factor, centre):
    # Description:
    # Takes the value to be transformed, `x`, and applies a simple linear
    # transformation about `centre` with a slope determined by `factor`
    # (+ve or -ve). The result is in [0, 1].
    return 1.0 / (1.0 + np.exp(-factor * (x - centre)))


@njit(nogil=True, cache=True, fastmath=True)
def calc_flam(
    temp_l,
    rhum_l,
    fuel_l,
    sm_l,
    rain_l,
    cum_rain_l,
    fuel_build_up,
    fapar,
    dry_days,
    flammability_method,
    dryness_method,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    dry_day_factor,
    dry_day_centre,
    dry_bal,
    rain_f,
    vpd_f,
    dry_bal_factor,
    dry_bal_centre,
):
    # Description:
    #   Performs the calculation of the flammibility
    #
    # Method:
    #   In essence, utilizes weather and vegetation variables to
    #   estimate how flammable a m2 is every second.

    # temp_l,
    #   # Surface Air Temperature (K)
    # rhum_l,
    #   # Relative Humidity (%)
    # sm_l,
    #   # The INFERNO soil moisture fraction (sthu's 1st level)
    # rain_l,
    #   # The precipitation rate (kg.m-2.s-1)
    # fuel_l
    #   # The Fuel Density (0-1)

    # flam_l
    #   # The flammability of the cell

    # These are variables to the Goff-Gratch equation
    a = -7.90298
    d = 11.344
    c = -1.3816e-07
    b = 5.02808
    f = 8.1328e-03
    h = -3.49149

    # Water saturation temperature
    Ts = 373.16
    # Precipitation factor (-2(day/mm)*(kg/m2/s))
    cr = -2.0 * s_in_day
    # Upper boundary to the relative humidity
    rhum_up = 90.0
    # Lower boundary to the relative humidity
    rhum_low = 10.0

    # Z_l,
    #   # Component of the Goff-Gratch saturation vapor pressure
    # TsbyT_l,
    #   # Reciprocal of the temperature times ts
    # f_rhum_l,
    #   # The factor dependence on relative humidity
    # f_sm_l,
    #   # The factor dependence on soil moisture
    # rain_rate

    TsbyT_l = Ts / temp_l

    Z_l = (
        a * (TsbyT_l - 1.0)
        + b * np.log10(TsbyT_l)
        + c * (10.0 ** (d * (1.0 - TsbyT_l)) - 1.0)
        + f * (10.0 ** (h * (TsbyT_l - 1.0)) - 1.0)
    )

    f_rhum_l = (rhum_up - rhum_l) / (rhum_up - rhum_low)

    # Create boundary limits
    # First for relative humidity
    if rhum_l < rhum_low:
        # Always fires for RH < 10%
        f_rhum_l = 1.0
    if rhum_l > rhum_up:
        # No fires for RH > 90%
        f_rhum_l = 0.0

    # The flammability goes down linearly with soil moisture
    f_sm_l = 1 - sm_l

    # convert rain rate from kg/m2/s to mm/day
    rain_rate = rain_l * s_in_day

    if flammability_method == 1:
        # Old flammability calculation.
        flammability = max(
            min(10.0 ** Z_l * f_rhum_l * fuel_l * f_sm_l * np.exp(cr * rain_rate), 1.0),
            0.0,
        )
    elif flammability_method == 2:
        # New calculation, based on FAPAR (and derived fuel_build_up).

        if dryness_method == 1:
            dry_factor = fuel_param(dry_days, dry_day_factor, dry_day_centre)
        elif dryness_method == 2:
            # Evolve the `dry_bal` variable.
            # Clamp to [-1, 1].
            # TODO Scale depending on timestep.
            vpd = (10.0 ** Z_l) * f_rhum_l
            dry_bal += max(
                min(rain_f * cum_rain_l - (1 - np.exp(-vpd_f * vpd)), 1.0), -1.0
            )
            dry_factor = fuel_param(dry_bal, dry_bal_factor, dry_bal_centre)
        else:
            raise ValueError("Unknown 'dryness_method'.")

        # Convert fuel build-up index to flammability factor.
        flammability = (
            dry_factor
            * fuel_param(temp_l, temperature_factor, temperature_centre)
            * fuel_param(fuel_build_up, fuel_build_up_factor, fuel_build_up_centre)
            * fuel_param(fapar, fapar_factor, fapar_centre)
        )
    else:
        raise ValueError("Unknown 'flammability_method'.")

    return flammability, dry_bal


@njit(nogil=True, cache=True, fastmath=True)
def calc_burnt_area(flam_l, ignitions_l, avg_ba_i):
    # Description:
    #    Calculate the burnt area
    #
    # Method:
    #    Multiply ignitions by flammability by average PFT burnt area

    # flam_l,
    #   # Flammability (depends on weather and vegetation)
    # ignitions_l,
    #   # Fire ignitions (ignitions/m2/s)
    # avg_ba_i
    #   # The average burned area (m2) for this PFT

    # Returns:
    # burnt_area_i_l
    #   # The burnt area (fraction of PFT per s)

    return flam_l * ignitions_l * avg_ba_i
