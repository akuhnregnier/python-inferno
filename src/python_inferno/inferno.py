# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from .calc_c_comps_triffid import calc_c_comps_triffid
from .configuration import avg_ba, land_pts, m2_in_km2, npft, s_in_day, s_in_month
from .qsat_wat import qsat_wat

# Indexing convention is time, pft, land


@njit(cache=True)
def inferno_io(
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
    flammability_method,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
):
    # Description:
    #   Called every model timestep, this subroutine updates INFERNO's
    #   driving variables and calls the scientific routines.
    #
    # Note that this code is currently incompatible with soil tiling.
    # The calculation of inferno_sm needs work.
    # Therefore all soil tile dimensions are hard coded to 1

    # c_root,
    #   # Carbon in leaves (kg m-2).
    # c_veg
    #   # Carbon in vegetation (kg m-2).

    # # Local temporary variables used in the interactive fire code
    # inferno_temp(land_pts),
    #   # The temperature (K)
    # inferno_rhum(land_pts),
    #   # The Relative Humidity (%)
    # inferno_sm(land_pts),
    #   # The Soil Moisture (Fraction of saturation)
    # inferno_rain(land_pts),
    #   # The total rainfall (kg/m2/s)
    # inferno_fuel(land_pts),
    #   # The fuel density (fine litter and leaves - kg/m3)
    # qsat(land_pts),
    #   # Saturation humidity
    # ignitions(land_pts),
    #   # The number of ignitions (#/m2/s)
    # lai_bal_inf(land_pts,npft),
    #   # The balanced lai used to compute carbon pools
    # leaf_inf(land_pts,npft),
    #   # The leaf carbon
    # wood_inf(land_pts,npft),
    #   # The wood carbon
    # dpm_fuel(land_pts),
    #   # The amount of DPM that is available to burn (kgC.m-2)
    # rpm_fuel(land_pts),
    #   # The amount of RPM that is available to burn (kgC.m-2)
    # ls_rain_filtered(land_pts),
    #   # Large scale rain from input after filtering negative values
    # con_rain_filtered(land_pts)
    #   # Convective rain from input after filtering negative values

    # # HARDCODED Emission factors for DPM in g kg-1
    # fef_co2_dpm = 1637.0
    # fef_co_dpm = 89.0
    # fef_ch4_dpm = 3.92
    # fef_nox_dpm = 2.51
    # fef_so2_dpm = 0.40
    # fef_oc_dpm = 8.2
    # fef_bc_dpm = 0.56

    # # HARDCODED Emission factors for RPM in g kg-1
    # fef_co2_rpm = 1489.0
    # fef_co_rpm = 127.0
    # fef_ch4_rpm = 5.96
    # fef_nox_rpm = 0.90
    # fef_so2_rpm = 0.40
    # fef_oc_rpm = 8.2
    # fef_bc_rpm = 0.56

    # Plant Material that is available as fuel (on the surface)
    pmtofuel = 0.7

    # Fuel availability high/low threshold
    fuel_low = 0.02
    fuel_high = 0.2

    # Tolerance number to filter non-physical rain values
    rain_tolerance = 1.0e-18  # kg/m2/s

    # Driving variables
    # - inferno_temp(:)
    # - inferno_rhum(:)
    # - inferno_sm(:)
    # - inferno_rain(:)
    # - inferno_fuel(:)

    inferno_temp = np.zeros((land_pts,))
    inferno_rhum = np.zeros((land_pts,))
    inferno_sm = np.zeros((land_pts,))
    inferno_rain = np.zeros((land_pts,))
    inferno_fuel = np.zeros((land_pts,))

    # Work variables
    # - qsat(:)
    # - lai_bal_inf(:,:)
    # - leaf_inf(:,:)
    # - wood_inf(:,:)
    # - ignitions(:)

    qsat = np.zeros((land_pts,))
    lai_bal_inf = np.zeros((npft, land_pts))
    leaf_inf = np.zeros((npft, land_pts))
    wood_inf = np.zeros((npft, land_pts))
    ignitions = np.zeros((land_pts,))

    # INFERNO diagnostic variables
    # - flammability_ft(:,:)
    # - burnt_area(:)
    # - burnt_area_ft(:,:)

    flammability_ft = np.zeros((npft, land_pts))
    burnt_area = np.zeros((land_pts,))
    burnt_area_ft = np.zeros((npft, land_pts))

    # Update antecedent fuel load.
    # TODO
    # fuel_build_up = fuel_build_up + fuel_build_up_alpha * (
    #     fapar_diag_pft - fuel_build_up
    # )

    # Get the available DPM and RPM using a scaling parameter

    dpm_fuel = pmtofuel * c_soil_dpm_gb
    # rpm_fuel = pmtofuel * c_soil_rpm_gb

    # Get the inferno meteorological variables for the whole gridbox

    # Soil Humidity (inferno_sm)
    # XXX What does selecting one of the 4 layers change here?
    inferno_sm = sthu_soilt[0, 0, :]

    # Rainfall (inferno_rain)

    # Rain fall values have a significant impact in the calculation of flammability.
    # In some cases we may be presented with values that have no significant meaning -
    # e.g in the UM context negative values or very small values can often be found/

    ls_rain_filtered = ls_rain.copy()
    con_rain_filtered = con_rain.copy()

    ls_rain_filtered[ls_rain < rain_tolerance] = 0.0
    con_rain_filtered[con_rain < rain_tolerance] = 0.0

    inferno_rain = ls_rain_filtered + con_rain_filtered

    # Diagnose the balanced-growth leaf area index and the carbon
    # contents of leaves and wood.
    for i in range(npft):
        for l in range(land_pts):
            (
                lai_bal_inf[i, l],
                leaf_inf[i, l],
                c_root,
                wood_inf[i, l],
                c_veg,
            ) = calc_c_comps_triffid(i, canht[i, l])

    # Fire calculations - per PFT
    for i in range(npft):
        # Calculate the fuel density
        # We use normalised Leaf Carbon + the available DPM
        inferno_fuel = (leaf_inf[i, :] + dpm_fuel - fuel_low) / (fuel_high - fuel_low)

        inferno_fuel[inferno_fuel < 0.0] = 0.0
        inferno_fuel[inferno_fuel > 1.0] = 1.0

        inferno_temp = t1p5m_tile[i, :]

        for l in range(land_pts):
            # Conditional statements to make sure we are dealing with
            # reasonable weather. Note initialisation to 0 already done.
            # If the driving variables are singularities, we assume
            # no burnt area.

            # Temperatures constrained akin to qsat (from the WMO)
            if (inferno_temp[l] > 338.15) or (inferno_temp[l] < 183.15):
                continue

            # The maximum rain rate ever observed is 38mm in one minute,
            # here we assume 0.5mm/s stops fires altogether
            if (inferno_rain[l] > 0.5) or (inferno_rain[l] < 0.0):
                continue

            # Fuel Density is an index constrained to 0-1
            if (inferno_fuel[l] > 1.0) or (inferno_fuel[l] < 0.0):
                continue

            # Soil moisture is a fraction of saturation
            if (inferno_sm[l] > 1.0) or (inferno_sm[l] < 0.0):
                continue

            # Get the tile relative humidity using saturation routine
            qsat[l] = qsat_wat(inferno_temp[l], pstar[l])

            # XXX: Is this bracket pair correct?
            inferno_rhum[l] = (q1p5m_tile[i, l] / qsat[l]) * 100.0

            # Relative Humidity should be constrained to 0-100
            if (inferno_rhum[l] > 100.0) or (inferno_rhum[l] < 0.0):
                continue

            # If all these checks are passes, start fire calculations

            ignitions[l] = calc_ignitions(
                pop_den[l],
                flash_rate[l],
                ignition_method,
            )

            flammability_ft[i, l] = calc_flam(
                inferno_temp[l],
                inferno_rhum[l],
                inferno_fuel[l],
                inferno_sm[l],
                inferno_rain[l],
                fuel_build_up[i, l],
                fapar_diag_pft[i, l],
                flammability_method,
                fapar_factor,
                fapar_centre,
                fuel_build_up_factor,
                fuel_build_up_centre,
            )

            burnt_area_ft[i, l] = calc_burnt_area(
                flammability_ft[i, l], ignitions[l], avg_ba[i]
            )

        # We add pft-specific variables to the gridbox totals
        burnt_area = burnt_area + frac[i, :] * burnt_area_ft[i, :]

    return burnt_area, burnt_area_ft


# Note: calc_ignitions, calc_flam and calc_burnt_area) are
# computed for each landpoint to ascertain no points with
# unrealistic weather contain fires.
# These are then aggregated in inferno_io_mod into pft arrays.


@njit(cache=True)
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


@njit(cache=True)
def fuel_param(x, factor, centre):
    # Description:
    # Takes the value to be transformed, `x`, and applies a simple linear
    # transformation about `centre` with a slope determined by `factor`
    # (+ve or -ve). The result is in [0, 1].
    return max(min(factor * (x - centre), 0.5), -0.5) + 0.5


@njit(cache=True)
def calc_flam(
    temp_l,
    rhum_l,
    fuel_l,
    sm_l,
    rain_l,
    fuel_build_up,
    fapar,
    flammability_method,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
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
        return max(
            min(10.0 ** Z_l * f_rhum_l * fuel_l * f_sm_l * np.exp(cr * rain_rate), 1.0),
            0.0,
        )

    elif flammability_method == 2:
        # New calculation, based solely on FAPAR (and derived fuel_build_up).
        # Convert fuel build-up index to flammability factor.
        return fuel_param(
            fuel_build_up, fuel_build_up_factor, fuel_build_up_centre
        ) * fuel_param(fapar, fapar_factor, fapar_centre)
    else:
        return -1.0


@njit(cache=True)
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
