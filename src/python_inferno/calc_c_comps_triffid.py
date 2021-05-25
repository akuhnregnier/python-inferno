# -*- coding: utf-8 -*-
from numba import njit

from .configuration import a_wl, a_ws, b_wl, cmass, eta_sl, l_trait_phys, lma, sigl


@njit(nogil=True, cache=True)
def calc_c_comps_triffid(n, ht):
    # Description:
    #   Calculates carbon contents from vegetation height

    # n (int): PFT number.
    # ht (float): Vegetation height (m).

    # Returns:
    # lai_bal_pft
    #                       # OUT Balanced leaf area index
    # ,leaf
    #                       # OUT Leaf biomass for balanced LAI (kg C/m2).
    # ,root
    #                       # OUT Root biomass (kg C/m2).
    # ,wood
    #                       # OUT Woody biomass (kg C/m2).
    # ,c_veg
    #                       # OUT Total carbon content of vegetation (kg C/m2).

    lai_bal_pft = (a_ws[n] * eta_sl[n] * ht / a_wl[n]) ** (1.0 / (b_wl[n] - 1.0))
    if l_trait_phys:
        leaf = cmass * lma[n] * lai_bal_pft
    else:
        leaf = sigl[n] * lai_bal_pft

    root = leaf
    wood = a_wl[n] * (lai_bal_pft ** b_wl[n])
    c_veg = leaf + root + wood

    return lai_bal_pft, leaf, root, wood, c_veg
