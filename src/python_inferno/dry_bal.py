# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, set_num_threads
from wildfires.qstat import get_ncpus

from .cache import mark_dependency
from .configuration import N_pft_groups, land_pts

# Indexing convention is time, pft, land


set_num_threads(get_ncpus())


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
@mark_dependency
def calculate_grouped_dry_bal(
    *,
    grouped_vpd,
    cum_rain,
    rain_f,
    vpd_f,
):
    Nt = grouped_vpd.shape[0]
    grouped_dry_bal = np.zeros((Nt, N_pft_groups, land_pts))

    assert rain_f.shape[0] == N_pft_groups
    assert vpd_f.shape[0] == N_pft_groups

    for l in prange(land_pts):
        for ti in range(Nt):
            for i in range(N_pft_groups):
                prev_dry_bal = grouped_dry_bal[max(ti - 1, 0), i, l]
                vpd_val = grouped_vpd[ti, i, l]

                grouped_dry_bal[ti, i, l] = max(
                    min(
                        prev_dry_bal
                        + rain_f[i] * cum_rain[ti, l]
                        - (1 - np.exp(-vpd_f[i] * vpd_val)),
                        1.0,
                    ),
                    -1.0,
                )
    return grouped_dry_bal
