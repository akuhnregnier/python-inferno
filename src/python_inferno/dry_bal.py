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
    init,  # (N_pft_groups, land_pts) array
    # NOTE This is where the output is placed and should be an (Nt, N_pft_groups,
    # land_pts) np.float64 array.
    out,
):
    Nt = grouped_vpd.shape[0]

    assert rain_f.shape[0] == N_pft_groups
    assert vpd_f.shape[0] == N_pft_groups
    assert out.shape == (Nt, N_pft_groups, land_pts)

    for l in prange(land_pts):
        for ti in range(Nt):
            for i in range(N_pft_groups):
                if ti == 0:
                    prev_dry_bal = init[i, l]
                else:
                    prev_dry_bal = out[ti - 1, i, l]

                vpd_val = grouped_vpd[ti, i, l]

                new_dry_bal = (
                    prev_dry_bal
                    + rain_f[i] * cum_rain[ti, l]
                    - (1 - np.exp(-vpd_f[i] * vpd_val))
                )

                if new_dry_bal < -1.0:
                    out[ti, i, l] = -1.0
                elif new_dry_bal > 1.0:
                    out[ti, i, l] = 1.0
                else:
                    out[ti, i, l] = new_dry_bal

    return out
