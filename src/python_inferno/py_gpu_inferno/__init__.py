# -*- coding: utf-8 -*-
from time import time

import numpy as np

from ..configuration import land_pts, n_total_pft, npft
from .py_gpu_inferno import GPUCalculateMPD as _GPUCalculateMPD
from .py_gpu_inferno import GPUCalculatePhase
from .py_gpu_inferno import GPUCompute as _GPUCompute
from .py_gpu_inferno import GPUConsAvg as _GPUConsAvg
from .py_gpu_inferno import calculate_phase
from .py_gpu_inferno import nme as _nme


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
        self.Nt = Nt
        self.frac = frac
        assert frac.shape == (self.Nt * n_total_pft * land_pts,)

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
        self.out = np.empty((self.Nt, land_pts), dtype=np.float32)

    def run(
        self,
        *,
        overall_scale,
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
        fapar_weight,
        dryness_weight,
        temperature_weight,
        fuel_weight,
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
            fapar_weight=fapar_weight,
            dryness_weight=dryness_weight,
            temperature_weight=temperature_weight,
            fuel_weight=fuel_weight,
        )
        self.gpu_compute.run(self.out)
        return overall_scale * self.out

    def release(self):
        self.gpu_compute.release()


class GPUCalculateMPD:
    def __init__(self, N):
        self.N = N
        self._GPUCalculateMPD = _GPUCalculateMPD(N)

    def run(self, *, obs, pred, return_ignored=False):
        out = self._GPUCalculateMPD.run(
            obs=np.asarray(obs, dtype=np.float32),
            pred=np.asarray(pred, dtype=np.float32),
        )
        if return_ignored:
            return out
        return out[0]


class GPUConsAvg:
    def __init__(self, L, weights):
        self.L = L
        self.M = weights.shape[0]
        self.N = weights.shape[1]
        self._GPUConsAvg = _GPUConsAvg(L, np.asarray(weights, dtype=np.float32))

    def run(self, data, mask):
        out_data, out_mask = self._GPUConsAvg.run(
            np.asarray(data, dtype=np.float32), mask
        )
        return out_data.reshape((self.N, self.L)), out_mask.reshape((self.N, self.L))


def cpp_nme(*, obs, pred):
    return _nme(
        obs=np.asarray(obs, dtype=np.float32), pred=np.asarray(pred, dtype=np.float32)
    )
