# -*- coding: utf-8 -*-
"""Model configuration."""
import numpy as np

# Seconds in a month.
# Note that this is approx. 365 days/12months but is slightly larger.
# This should be changed in a future update.
s_in_month = 2.6280288e6
m2_in_km2 = 1.0e6
rsec_per_day = s_in_day = 86400.0

npft = 13
land_pts = 7771

# Fraction of leaf dry matter in the form of C
# Used to convert LMA (kgleaf/m2) to sigl (kgC/m2)
cmass = 0.40

l_trait_phys = True
a_ws = (12, 13, 12, 10, 10, *(6 * [1]), 13, 13)
eta_sl = 13 * (0.01,)
a_wl = (0.78, 0.845, 0.78, 0.8, 0.65, *(6 * [0.005]), 0.13, 0.13)
b_wl = 13 * (1.667,)
lma = (
    0.0823,
    0.1036,
    0.1403,
    0.1006,
    0.2263,
    0.0495,
    0.0495,
    0.0495,
    0.137,
    0.137,
    0.137,
    0.0709,
    0.1515,
)
sigl = (0.0375, 0.0375, 0.0375, 0.1, 0.1, 0.025, 0.025, 0.025, *(5 * [0.05]))

n_day_fuel_build_up = 13 * (90,)
avg_ba = (*(5 * [1.7e6]), 3.2e6, 0.4e6, 3.2e6, 3.2e6, 0.4e6, 3.2e6, 2.7e6, 2.7e6)

# Code in JULES currently uses exp( -1 * timestep / ...), but timestep should be 1
# when this is calculate (during initialisation)?
fuel_build_up_alpha = tuple(
    1.0 - np.exp(-1.0 / (np.array(n_day_fuel_build_up) * rsec_per_day))
)

# Checks.

# Check for the proper number of PFTs
for var in (
    a_ws,
    eta_sl,
    a_wl,
    b_wl,
    lma,
    sigl,
    n_day_fuel_build_up,
    avg_ba,
    fuel_build_up_alpha,
):
    assert len(var) == npft

pft_group_names = ("Trees", "Grass", "Shrubs")
pft_groups = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10), (11, 12))
N_pft_groups = len(pft_groups)

pft_groups_array = np.zeros((N_pft_groups, npft), dtype=np.int64)
pft_groups_lengths = np.zeros((N_pft_groups,), dtype=np.int64)

for i, indices in enumerate(pft_groups):
    pft_groups_array[i][: len(indices)] = indices
    pft_groups_lengths[i] = len(indices)
