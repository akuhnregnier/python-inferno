# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .cache import mark_dependency
from .configuration import land_pts, npft

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
@mark_dependency
def calculate_litter(
    *,
    leaf_litC,
    T,
    sm,
    dt,
    litter_tc,
    leaf_f,
    # NOTE This is where the initial state is placed and should be an
    # (npft, land_pts) np.float64 array. This is used e.g. for spinup.
    init,
    # NOTE This is where the output is placed and should be an (Nt, npft,
    # land_pts) np.float64 array.
    out,
):
    """Calculate litter pool over time.

    Args:
        leaf_litC ((Nt, npft, land_pts) array): Litter carbon due to leaf turnover.
        T ((Nt, npft, land_pts) array): Temperature (K).
        sm ((Nt, land_pts) array): Upper soil moisture fraction.
        dt (float): Timestep.
        litter_tc ((npft,) array): Litter decomposition factor (time constant).
        leaf_f ((npft,) array): Factor modifying leaf litter contribution to litter pool.
        out ((Nt, npft, land_pts) array): Array to which the calculated litter pool
            data will be written.

    """
    Nt = leaf_litC.shape[0]

    assert T.shape == (Nt, npft, land_pts)
    assert sm.shape == (Nt, land_pts)
    assert leaf_litC.shape == (Nt, npft, land_pts)
    assert out.shape == (Nt, npft, land_pts)

    for l in prange(land_pts):
        for ti in range(Nt):
            for i in range(npft):
                if ti == 0:
                    prev_litter = init[i, l]
                else:
                    prev_litter = out[ti - 1, i, l]

                # Eqn. 45 in Sitch et al. 2003.
                kdt = litter_tc[i] * decomposition_factor(T[ti, i, l], sm[ti, l]) * dt

                # Cap k * dt.
                if kdt > 1.0:
                    kdt = 1.0

                litter_diff = (
                    # Decomposition of existing litter.
                    -kdt * prev_litter
                    # Addition of new litter.
                    + leaf_f[i] * leaf_litC[ti, i, l] * dt
                )

                out[ti, i, l] = prev_litter + litter_diff


@njit(cache=True, nogil=True, fastmath=True)
def decomposition_factor(T, sm):
    """Litter pool decomposition factor.

    This will be multiplied by another factor subject to optimisation.
    Together, this represents the timescale of decomposition.

    Args:
        T (float): Temperature (K).
        sm (float): Upper soil moisture fraction.

    """
    return g_arrhenius(T) * f_moisture(sm)


@njit(cache=True, nogil=True, fastmath=True)
def g_arrhenius(T):
    """Modified Arrhenius relationships (Lloyd & Taylor, 1994).

    Args:
        T (float): Temperature (K).

    """
    return np.exp(
        308.56
        * (
            (1 / 56.02)
            # Eqn. 23 from Sitch et al. 2003 is (T + 46.02), but is written in °C.
            - (1 / (T + 227.13))
        )
    )


@njit(cache=True, nogil=True, fastmath=True)
def f_moisture(sm):
    """Effect of soil moisture on decomposition (Foley (1995)).

    Following Sitch et al. 2003, W1 is the saturated fraction of available water
    holding capacity (volumetric water content, upper soil layer, 'sthu' in JULES(?)).

    Args:
        sm (float): Upper soil moisture fraction.

    """
    return 0.25 + 0.75 * sm
