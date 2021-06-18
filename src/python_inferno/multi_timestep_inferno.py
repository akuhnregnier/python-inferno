# -*- coding: utf-8 -*-
import numpy as np
from wildfires.utils import parallel_njit

from .configuration import land_pts
from .inferno import inferno_io
from .precip_dry_day import precip_moving_sum


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
    dry_bal=None,
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
    rain_f,
    vpd_f,
    dry_bal_factor,
    dry_bal_centre,
    timestep,
    return_dry_bal=False,
):
    if dry_bal is None:
        dry_bal = np.zeros_like(fapar_diag_pft)

    # Call the below using normal, non-numba Python to enable features like
    # keyword-only arguments with default arguments as above.
    ba, dry_bal = _multi_timestep_inferno(
        t1p5m_tile=t1p5m_tile,
        q1p5m_tile=q1p5m_tile,
        pstar=pstar,
        sthu_soilt=sthu_soilt,
        frac=frac,
        c_soil_dpm_gb=c_soil_dpm_gb,
        c_soil_rpm_gb=c_soil_rpm_gb,
        canht=canht,
        ls_rain=ls_rain,
        con_rain=con_rain,
        pop_den=pop_den,
        flash_rate=flash_rate,
        ignition_method=ignition_method,
        fuel_build_up=fuel_build_up,
        fapar_diag_pft=fapar_diag_pft,
        dry_bal=dry_bal,
        dry_days=dry_days,
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
        rain_f=rain_f,
        vpd_f=vpd_f,
        dry_bal_factor=dry_bal_factor,
        dry_bal_centre=dry_bal_centre,
        cum_rain=precip_moving_sum(
            ls_rain=ls_rain, con_rain=con_rain, timestep=timestep
        ),
    )
    if return_dry_bal:
        return ba, dry_bal
    return ba


@parallel_njit(cache=True)
def _multi_timestep_inferno(
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
    dry_bal,
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
    rain_f,
    vpd_f,
    dry_bal_factor,
    dry_bal_centre,
    cum_rain,
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
        == cum_rain.shape[0]
    ):
        raise ValueError("All arrays need to have the same time dimension.")

    # Store the output BA (averaged over PFTs).
    ba = np.zeros_like(pstar)

    land_pts_dummy = np.zeros((land_pts,)) - 1

    for ti in range(fapar_diag_pft.shape[0]):
        # Retrieve the individual time slices.
        ba[ti], dry_bal[ti] = inferno_io(
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
            dry_bal=dry_bal[max(ti - 1, 0)],
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
            rain_f=rain_f,
            vpd_f=vpd_f,
            dry_bal_factor=dry_bal_factor,
            dry_bal_centre=dry_bal_centre,
            cum_rain=cum_rain[ti],
        )
    return ba, dry_bal
