# -*- coding: utf-8 -*-
from time import time

import numpy as np

from ..configuration import land_pts, n_total_pft, npft
from .py_gpu_inferno import GPUCompute as _GPUCompute


class GPUInferno:
    def __init__(
        self,
        *,
        ignition_method,
        flammability_method,
        dryness_method,
        fuel_build_up_method,
        include_temperature,
        Nt,
        t1p5m_tile,
        q1p5m_tile,
        pstar,
        sthu_soilt_single,
        frac,
        c_soil_dpm_gb,
        c_soil_rpm_gb,
        canht,
        ls_rain,
        con_rain,
        pop_den,
        flash_rate,
        fuel_build_up,
        fapar_diag_pft,
        grouped_dry_bal,
        litter_pool,
        dry_days,
    ):
        self.gpu_compute = _GPUCompute()

        self.gpu_compute.set_data(
            _ignitionMethod=ignition_method,
            _flammabilityMethod=flammability_method,
            _drynessMethod=dryness_method,
            _fuelBuildUpMethod=fuel_build_up_method,
            _includeTemperature=include_temperature,
            _Nt=Nt,
            t1p5m_tile=t1p5m_tile,
            q1p5m_tile=q1p5m_tile,
            pstar=pstar,
            sthu_soilt_single=sthu_soilt_single,
            frac=frac,
            c_soil_dpm_gb=c_soil_dpm_gb,
            c_soil_rpm_gb=c_soil_rpm_gb,
            canht=canht,
            ls_rain=ls_rain,
            con_rain=con_rain,
            pop_den=pop_den,
            flash_rate=flash_rate,
            fuel_build_up=fuel_build_up,
            fapar_diag_pft=fapar_diag_pft,
            grouped_dry_bal=grouped_dry_bal,
            litter_pool=litter_pool,
            dry_days=dry_days,
        )

    def run(
        self,
        *,
        fapar_factor,
        fapar_centre,
        fapar_shape,
        fuel_build_up_factor,
        fuel_build_up_centre,
        fuel_build_up_shape,
        temperature_factor,
        temperature_centre,
        temperature_shape,
        dry_day_factor,
        dry_day_centre,
        dry_day_shape,
        dry_bal_factor,
        dry_bal_centre,
        dry_bal_shape,
        litter_pool_factor,
        litter_pool_centre,
        litter_pool_shape,
    ):
        self.gpu_compute.set_params(
            fapar_factor=fapar_factor,
            fapar_centre=fapar_centre,
            fapar_shape=fapar_shape,
            fuel_build_up_factor=fuel_build_up_factor,
            fuel_build_up_centre=fuel_build_up_centre,
            fuel_build_up_shape=fuel_build_up_shape,
            temperature_factor=temperature_factor,
            temperature_centre=temperature_centre,
            temperature_shape=temperature_shape,
            dry_day_factor=dry_day_factor,
            dry_day_centre=dry_day_centre,
            dry_day_shape=dry_day_shape,
            dry_bal_factor=dry_bal_factor,
            dry_bal_centre=dry_bal_centre,
            dry_bal_shape=dry_bal_shape,
            litter_pool_factor=litter_pool_factor,
            litter_pool_centre=litter_pool_centre,
            litter_pool_shape=litter_pool_shape,
        )

        return self.gpu_compute.run()

    def release(self):
        self.gpu_compute.release()


def frac_weighted_mean(*, data, frac):
    assert len(data.shape) == 3, "Need time, PFT, and space coords."
    assert data.shape[1] in (npft, n_total_pft)
    assert frac.shape[1] == n_total_pft

    # NOTE - The below would be more correct, but does not currently correspond to the
    # NUMBA Python implementation.
    # return np.sum(data * frac[:, : data.shape[1]], axis=1) / np.sum(
    #     frac[:, : data.shape[1]], axis=1
    # )
    return np.sum(data * frac[:, : data.shape[1]], axis=1)


def run_single_shot(
    *,
    # Init params.
    ignition_method,
    flammability_method,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
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
    fuel_build_up,
    fapar_diag_pft,
    grouped_dry_bal,
    litter_pool,
    dry_days,
    # Per-run params.
    fapar_factor,
    fapar_centre,
    fapar_shape,
    fuel_build_up_factor,
    fuel_build_up_centre,
    fuel_build_up_shape,
    temperature_factor,
    temperature_centre,
    temperature_shape,
    dry_day_factor,
    dry_day_centre,
    dry_day_shape,
    dry_bal_factor,
    dry_bal_centre,
    dry_bal_shape,
    litter_pool_factor,
    litter_pool_centre,
    litter_pool_shape,
):
    Nt = pstar.shape[0]

    compute = GPUInferno(
        ignition_method=ignition_method,
        flammability_method=flammability_method,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        Nt=Nt,
        t1p5m_tile=t1p5m_tile.ravel(),
        q1p5m_tile=q1p5m_tile.ravel(),
        pstar=pstar.ravel(),
        sthu_soilt_single=sthu_soilt[:, 0, 0].ravel(),
        frac=frac.ravel(),
        c_soil_dpm_gb=c_soil_dpm_gb.ravel(),
        c_soil_rpm_gb=c_soil_rpm_gb.ravel(),
        canht=canht.ravel(),
        ls_rain=ls_rain.ravel(),
        con_rain=con_rain.ravel(),
        pop_den=pop_den.ravel(),
        flash_rate=flash_rate.ravel(),
        fuel_build_up=fuel_build_up.ravel(),
        fapar_diag_pft=fapar_diag_pft.ravel(),
        grouped_dry_bal=grouped_dry_bal.ravel(),
        litter_pool=litter_pool.ravel(),
        dry_days=dry_days.ravel(),
    )

    out = compute.run(
        fapar_factor=fapar_factor,
        fapar_centre=fapar_centre,
        fapar_shape=fapar_shape,
        fuel_build_up_factor=fuel_build_up_factor,
        fuel_build_up_centre=fuel_build_up_centre,
        fuel_build_up_shape=fuel_build_up_shape,
        temperature_factor=temperature_factor,
        temperature_centre=temperature_centre,
        temperature_shape=temperature_shape,
        dry_day_factor=dry_day_factor,
        dry_day_centre=dry_day_centre,
        dry_day_shape=dry_day_shape,
        dry_bal_factor=dry_bal_factor,
        dry_bal_centre=dry_bal_centre,
        dry_bal_shape=dry_bal_shape,
        litter_pool_factor=litter_pool_factor,
        litter_pool_centre=litter_pool_centre,
        litter_pool_shape=litter_pool_shape,
    )

    compute.release()

    weighted = frac_weighted_mean(data=out.reshape((Nt, npft, land_pts)), frac=frac)
    return weighted
