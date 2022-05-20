# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .cache import mark_dependency
from .configuration import land_pts, npft

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


@njit(nogil=True, fastmath=True, cache=True)
@mark_dependency
def _spinup(
    Nt,
    pft_i,
    l,
    leaf_litC,
    T,
    sm,
    dt,
    litter_tc,
    leaf_f,
    spinup_relative_delta,
    max_spinup_cycles,
    # NOTE This is where the output is placed and should be an (Nt, npft,
    # land_pts) np.float32 array.
    out,
):
    spinup_comp_litter = 0.0

    for s in range(max_spinup_cycles):
        for ti in range(Nt):
            if ti == 0 and s > 0:
                # Use end of last spinup iterations's run.
                prev_litter = out[Nt - 1, pft_i, l]
            elif ti > 0:
                prev_litter = out[ti - 1, pft_i, l]
            else:
                # Only at the beginning.
                prev_litter = 0.0

            # Eqn. 45 in Sitch et al. 2003.
            kdt = (
                litter_tc[pft_i] * decomposition_factor(T[ti, pft_i, l], sm[ti, l]) * dt
            )

            # Cap k * dt.
            if kdt > 1.0:
                kdt = 1.0

            litter_diff = (
                # Decomposition of existing litter.
                -kdt * prev_litter
                # Addition of new litter.
                + leaf_f[pft_i] * leaf_litC[ti, pft_i, l] * dt
            )

            out[ti, pft_i, l] = prev_litter + litter_diff

        current_litter = out[0, pft_i, l]

        if s > 0:
            if current_litter > 1e-15:
                # If we have done at least 1 spinup cycle, check for
                # convergence at this (pft, land) point.
                delta = abs(current_litter - spinup_comp_litter) / current_litter

                if delta < spinup_relative_delta:
                    return True
            elif current_litter < 0:
                return False
            elif current_litter <= 1e-15:
                # Abort, since there does not appear to be any litter buildup
                # here.
                # Record this as a success, however!
                return True

        spinup_comp_litter = current_litter

    return False


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
@mark_dependency
def litter_spinup(
    *,
    leaf_litC,
    T,
    sm,
    dt,
    litter_tc,
    leaf_f,
    spinup_relative_delta,
    max_spinup_cycles,
    # NOTE This is where the output is placed and should be an
    # (Nt, npft, land_pts) np.float32 array.
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
        spinup_relative_delta (float): Target relative difference.
        max_spinup_cycles (int): Maximum number of spinup cycles to perform.
        out ((Nt, npft, land_pts) array): Array to which the calculated litter pool
            data will be written.

    """
    Nt = leaf_litC.shape[0]

    assert T.shape == (Nt, npft, land_pts)
    assert sm.shape == (Nt, land_pts)
    assert leaf_litC.shape == (Nt, npft, land_pts)
    assert out.shape == (Nt, npft, land_pts)

    assert max_spinup_cycles > 1

    # Each land point and PFT are independent.
    converged = np.zeros((npft, land_pts), dtype=np.bool_)
    for l_pft_i in prange(land_pts * npft):
        pft_i = l_pft_i // land_pts
        l = l_pft_i - (pft_i * land_pts)

        converged[pft_i, l] = _spinup(
            Nt=Nt,
            pft_i=pft_i,
            l=l,
            leaf_litC=leaf_litC,
            T=T,
            sm=sm,
            dt=dt,
            litter_tc=litter_tc,
            leaf_f=leaf_f,
            spinup_relative_delta=spinup_relative_delta,
            max_spinup_cycles=max_spinup_cycles,
            out=out,
        )
    return np.all(converged)


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
@mark_dependency
def calculate_litter_old(
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
        for pft_i in range(npft):
            for ti in range(Nt):
                if ti == 0:
                    prev_litter = init[pft_i, l]
                else:
                    prev_litter = out[ti - 1, pft_i, l]

                # Eqn. 45 in Sitch et al. 2003.
                kdt = (
                    litter_tc[pft_i]
                    * decomposition_factor(T[ti, pft_i, l], sm[ti, l])
                    * dt
                )

                # Cap k * dt.
                if kdt > 1.0:
                    kdt = 1.0

                litter_diff = (
                    # Decomposition of existing litter.
                    -kdt * prev_litter
                    # Addition of new litter.
                    + leaf_f[pft_i] * leaf_litC[ti, pft_i, l] * dt
                )

                out[ti, pft_i, l] = prev_litter + litter_diff


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
