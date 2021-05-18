# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from .qsat_data import es_wat as es


@njit(cache=True)
def qsat_wat(t, p):
    """
    Calculates saturation vapour pressure

    Parameters
    ----------
    t : float
        Temperature
    p : float
        surface pressure

    Returns
    -------
    float
        Saturation vapour pressure

    """
    repsilon = 0.62198
    one_minus_epsilon = 1.0 - repsilon
    zerodegc = 273.15
    delta_t = 0.1
    t_low = 183.15
    t_high = 338.15

    one = 1.0
    pconv = 1.0e-8
    term1 = 4.5
    term2 = 6.0e-4

    # Compute the factor that converts from sat vapour pressure in a
    # pure water system to sat vapour pressure in air, fsubw.
    # This formula is taken from equation A4.7 of Adrian Gill's book:
    # atmosphere-ocean dynamics. Note that his formula works in terms
    # of pressure in mb and temperature in celsius, so conversion of
    # units leads to the slightly different equation used here.
    fsubw = one + pconv * p * (term1 + term2 * (t - zerodegc) * (t - zerodegc))

    # Use the lookup table to find saturated vapour pressure. Store it in qs.
    tt = np.maximum(t_low, t)
    tt = np.minimum(t_high, tt)
    atable = (tt - t_low + delta_t) / delta_t
    itable = int(atable)
    atable = int(atable - itable)
    qs = (one - atable) * es[itable] + atable * es[itable + 1]

    # Multiply by fsubw to convert to saturated vapour pressure in air
    # (equation A4.6 OF Adrian Gill's book).
    qs *= fsubw

    # Now form the accurate expression for qs, which is a rearranged
    # version of equation A4.3 of Gill's book.
    # Note that at very low pressures we apply a fix, to prevent a
    # singularity (qsat tends to 1. kg/kg).
    return (repsilon * qs) / (np.maximum(p, qs) - one_minus_epsilon * qs)
