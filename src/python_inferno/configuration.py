# -*- coding: utf-8 -*-
"""Model configuration."""
import os
from enum import Enum
from itertools import product
from pathlib import Path

import numpy as np

# Seconds in a month.
# Note that this is approx. 365 days/12months but is slightly larger.
# This should be changed in a future update.
s_in_month = 2.6280288e6
m2_in_km2 = 1.0e6
rsec_per_day = s_in_day = 86400.0

n_total_pft = 17
npft = 13
land_pts = 7771

n_cell_tot_pft = lambda Nt: Nt * n_total_pft * land_pts
n_cell_nat_pft = lambda Nt: Nt * npft * land_pts
n_cell_grp_pft = lambda Nt: Nt * N_pft_groups * land_pts
n_cell_no_pft = lambda Nt: Nt * land_pts

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
assert len(pft_group_names) == N_pft_groups

pft_groups_array = np.zeros((N_pft_groups, npft), dtype=np.int64)
pft_groups_lengths = np.zeros((N_pft_groups,), dtype=np.int64)

for i, indices in enumerate(pft_groups):
    pft_groups_array[i][: len(indices)] = indices
    pft_groups_lengths[i] = len(indices)

pft_group_map = np.zeros(13, dtype=np.int64)
for pft_group_i, indices in enumerate(pft_groups):
    for i in indices:
        pft_group_map[i] = pft_group_i

dryness_descr = {1: "Dry Day", 2: "VPD & Precip"}
dryness_schemes = {1: "A", 2: "B"}
fuel_descr = {1: "Antec NPP", 2: "Leaf Litter Pool"}
fuel_schemes = {1: "1", 2: "2"}

dryness_keys = {1: "Dry_Day", 2: "VPD_Precip"}
fuel_keys = {1: "Antec_NPP", 2: "Leaf_Litter_Pool"}


def get_exp_name(*, dryness_method, fuel_build_up_method):
    return (
        f"Dry:{dryness_descr[dryness_method]}, Fuel:{fuel_descr[fuel_build_up_method]}"
    )


def get_exp_key(*, dryness_method, fuel_build_up_method):
    return f"dry_{dryness_keys[dryness_method]}__fuel_{fuel_keys[fuel_build_up_method]}"


scheme_name_map = {}
for dryness_method, fuel_build_up_method in product([1, 2], [1, 2]):
    scheme_name = dryness_schemes[dryness_method] + fuel_schemes[fuel_build_up_method]

    scheme_name_map[
        get_exp_key(
            dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
        )
    ] = scheme_name

    scheme_name_map[
        get_exp_name(
            dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
        )
    ] = scheme_name

    scheme_name_map[(dryness_method, fuel_build_up_method)] = scheme_name


Dims = Enum("Dims", ["TIME", "PFT", "LAND", "SAMPLE"])

default_opt_record_dir = Path(os.environ["EPHEMERAL"]) / "new_run_opt_record"


def get_name_key_map(*, dryness_method, fuel_build_up_method):
    name_map = {
        "t1p5m_tile": "temperature",
        "fapar_diag_pft": "fapar",
    }

    if dryness_method == 1:
        name_map["dry_days"] = "dry_day"
    elif dryness_method == 2:
        name_map["grouped_dry_bal"] = "dry_bal"

    if fuel_build_up_method == 1:
        name_map["fuel_build_up"] = "fuel_build_up"
    elif fuel_build_up_method == 2:
        name_map["litter_pool"] = "litter_pool"

    return name_map


def get_weight_key_map(*, dryness_method, fuel_build_up_method):
    weight_key_map = {
        "t1p5m_tile": "temperature_weight",
        "fapar_diag_pft": "fapar_weight",
    }

    if dryness_method == 1:
        weight_key_map["dry_days"] = "dryness_weight"
    elif dryness_method == 2:
        weight_key_map["grouped_dry_bal"] = "dryness_weight"

    if fuel_build_up_method == 1:
        weight_key_map["fuel_build_up"] = "fuel_weight"
    elif fuel_build_up_method == 2:
        weight_key_map["litter_pool"] = "fuel_weight"

    return weight_key_map
