# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from .configuration import m2_in_km2, s_in_day, s_in_month

# Indexing convention is time, pft, land


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
def sigmoid(x, factor, centre, shape):
    """Apply generalised sigmoid with slope determine by `factor`, position by
    `centre`, and shape by `shape`, with the result being in [0, 1]."""
    return (1.0 + np.exp(factor * shape * (centre - x))) ** (-1.0 / shape)


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
    fuel_build_up_method,
    fapar_factor,
    fapar_centre,
    fapar_shape,
    fuel_build_up_factor,
    fuel_build_up_centre,
    fuel_build_up_shape,
    temperature_factor,
    temperature_centre,
    temperature_shape,
    dry_day_factor,
    dry_day_centre,
    dry_day_shape,
    dry_bal,
    dry_bal_factor,
    dry_bal_centre,
    dry_bal_shape,
    litter_pool,
    litter_pool_factor,
    litter_pool_centre,
    litter_pool_shape,
    include_temperature,
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

    if flammability_method == 1:
        # Old flammability calculation.

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

        flammability = max(
            min(10.0 ** Z_l * f_rhum_l * fuel_l * f_sm_l * np.exp(cr * rain_rate), 1.0),
            0.0,
        )
    elif flammability_method == 2:
        # New calculation, based on FAPAR (and derived fuel_build_up).

        if dryness_method == 1:
            dry_factor = sigmoid(
                dry_days, dry_day_factor, dry_day_centre, dry_day_shape
            )
        elif dryness_method == 2:
            dry_factor = sigmoid(dry_bal, dry_bal_factor, dry_bal_centre, dry_bal_shape)
        else:
            raise ValueError("Unknown 'dryness_method'.")

        if fuel_build_up_method == 1:
            fuel_factor = sigmoid(
                fuel_build_up,
                fuel_build_up_factor,
                fuel_build_up_centre,
                fuel_build_up_shape,
            )
        elif fuel_build_up_method == 2:
            fuel_factor = sigmoid(
                litter_pool, litter_pool_factor, litter_pool_centre, litter_pool_shape
            )
        else:
            raise ValueError("Unknown 'fuel_build_up_method'.")

        if include_temperature == 1:
            temperature_factor = sigmoid(
                temp_l, temperature_factor, temperature_centre, temperature_shape
            )
        elif include_temperature == 0:
            temperature_factor = 1.0
        else:
            raise ValueError("Unknown 'include_temperature'.")

        # Convert fuel build-up index to flammability factor.
        flammability = (
            dry_factor
            * temperature_factor
            * fuel_factor
            * sigmoid(fapar, fapar_factor, fapar_centre, fapar_shape)
        )
    else:
        raise ValueError("Unknown 'flammability_method'.")

    return flammability


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
