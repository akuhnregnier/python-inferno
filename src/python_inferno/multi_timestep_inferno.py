# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from .configuration import land_pts
from .inferno import inferno_io


@njit(parallel=True, nogil=True, cache=True)
def multi_timestep_inferno(
    *,
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    sthu_soilt,
    frac,
    c_soil_dpm_gb,
    c_soil_rpm_gb,
    canht,
    ls_rain,
    con_rain,
    pop_den,
    flash_rate,
    ignition_method,
    fuel_build_up,
    fapar_diag_pft,
    dry_days,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    flammability_method,
    dryness_method,
    dry_day_factor,
    dry_day_centre,
):
    # Ensure consistency of the time dimension.
    if not (
        t1p5m_tile.shape[0]
        == q1p5m_tile.shape[0]
        == pstar.shape[0]
        == sthu_soilt.shape[0]
        == frac.shape[0]
        == c_soil_dpm_gb.shape[0]
        == c_soil_rpm_gb.shape[0]
        == canht.shape[0]
        == ls_rain.shape[0]
        == con_rain.shape[0]
        == fuel_build_up.shape[0]
        == fapar_diag_pft.shape[0]
    ):
        raise ValueError("All arrays need to have the same time dimension.")

    # Store the output BA (averaged over PFTs).
    ba = np.zeros_like(pstar)

    land_pts_dummy = np.zeros((land_pts,)) - 1

    for ti in range(fapar_diag_pft.shape[0]):
        # Retrieve the individual time slices.
        ba[ti] = inferno_io(
            t1p5m_tile=t1p5m_tile[ti],
            q1p5m_tile=q1p5m_tile[ti],
            pstar=pstar[ti],
            sthu_soilt=sthu_soilt[ti],
            frac=frac[ti],
            c_soil_dpm_gb=c_soil_dpm_gb[ti],
            c_soil_rpm_gb=c_soil_rpm_gb[ti],
            canht=canht[ti],
            ls_rain=ls_rain[ti],
            con_rain=con_rain[ti],
            # Not used for ignition mode 1.
            pop_den=land_pts_dummy,
            flash_rate=land_pts_dummy,
            ignition_method=ignition_method,
            fuel_build_up=fuel_build_up[ti],
            fapar_diag_pft=fapar_diag_pft[ti],
            dry_days=dry_days[ti],
            fapar_factor=fapar_factor,
            fapar_centre=fapar_centre,
            fuel_build_up_factor=fuel_build_up_factor,
            fuel_build_up_centre=fuel_build_up_centre,
            temperature_factor=temperature_factor,
            temperature_centre=temperature_centre,
            flammability_method=flammability_method,
            dryness_method=dryness_method,
            dry_day_factor=dry_day_factor,
            dry_day_centre=dry_day_centre,
        )[0]
    return ba
