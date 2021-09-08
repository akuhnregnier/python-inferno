# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .cache import mark_dependency
from .configuration import N_pft_groups, land_pts, npft, pft_groups_lengths
from .qsat_wat import qsat_wat
from .utils import get_pft_group_index

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
@mark_dependency
def calculate_grouped_vpd(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
):
    # These are variables to the Goff-Gratch equation
    a = -7.90298
    d = 11.344
    c = -1.3816e-07
    b = 5.02808
    f = 8.1328e-03
    h = -3.49149

    # Water saturation temperature
    Ts = 373.16
    # Upper boundary to the relative humidity
    rhum_up = 90.0
    # Lower boundary to the relative humidity
    rhum_low = 10.0

    Nt = pstar.shape[0]
    grouped_vpd = np.zeros((Nt, N_pft_groups, land_pts))

    for l in prange(land_pts):
        for ti in range(Nt):
            # TODO - warning if this occurs?
            # The maximum rain rate ever observed is 38mm in one minute,
            # here we assume 0.5mm/s stops fires altogether
            # if (inferno_rain > 0.5) or (inferno_rain < 0.0):
            #     continue

            # TODO - warning if this occurs?
            # Soil moisture is a fraction of saturation
            # if (inferno_sm[ti, l] > 1.0) or (inferno_sm[ti, l] < 0.0):
            #     continue

            for i in range(npft):
                # Conditional statements to make sure we are dealing with
                # reasonable weather. Note initialisation to 0 already done.
                # If the driving variables are singularities, we assume
                # no burnt area.

                # TODO - warning if this occurs?
                # Temperatures constrained akin to qsat (from the WMO)
                # if (t1p5m_tile[ti, i, l] > 338.15) or (t1p5m_tile[ti, i, l] < 183.15):
                #     continue

                # Get the tile relative humidity using saturation routine
                qsat = qsat_wat(t1p5m_tile[ti, i, l], pstar[ti, l])

                inferno_rhum = (q1p5m_tile[ti, i, l] / qsat) * 100.0

                # TODO - warning if this occurs?
                # # Relative Humidity should be constrained to 0-100
                # if (inferno_rhum > 100.0) or (inferno_rhum < 0.0):
                #     continue

                TsbyT_l = Ts / t1p5m_tile[ti, i, l]

                Z_l = (
                    a * (TsbyT_l - 1.0)
                    + b * np.log10(TsbyT_l)
                    + c * (10.0 ** (d * (1.0 - TsbyT_l)) - 1.0)
                    + f * (10.0 ** (h * (TsbyT_l - 1.0)) - 1.0)
                )

                f_rhum_l = (rhum_up - inferno_rhum) / (rhum_up - rhum_low)

                # Create boundary limits
                # First for relative humidity
                if inferno_rhum < rhum_low:
                    # Always fires for RH < 10%
                    f_rhum_l = 1.0
                if inferno_rhum > rhum_up:
                    # No fires for RH > 90%
                    f_rhum_l = 0.0

                grouped_vpd[ti, get_pft_group_index(i), l] += (10.0 ** Z_l) * f_rhum_l

    # Normalise by dividing by the number of PFTs in each group.
    for pft_group_index in range(N_pft_groups):
        grouped_vpd[:, pft_group_index] /= pft_groups_lengths[pft_group_index]

    return grouped_vpd
