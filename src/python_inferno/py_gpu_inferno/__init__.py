# -*- coding: utf-8 -*-
from operator import itemgetter
from time import time

import numpy as np

from ..configuration import Dims, N_pft_groups, land_pts, n_total_pft, npft
from .py_gpu_inferno import GPUCalculateMPD as _GPUCalculateMPD
from .py_gpu_inferno import GPUCalculatePhase
from .py_gpu_inferno import GPUCompute as _GPUCompute
from .py_gpu_inferno import GPUConsAvg as _GPUConsAvg
from .py_gpu_inferno import GPUConsAvgNoMask as _GPUConsAvgNoMask
from .py_gpu_inferno import GPUFlam2
from .py_gpu_inferno import GPUInfernoAvg as _GPUInfernoAvg
from .py_gpu_inferno import GPUInfernoAvgScore as _GPUInfernoAvgScore
from .py_gpu_inferno import GPUSACompute as _GPUSACompute
from .py_gpu_inferno import calculate_phase
from .py_gpu_inferno import cons_avg_no_mask as _cons_avg_no_mask
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
        checks_failed,
    ):
        self.gpu_compute = self._get_compute()
        self.Nt = Nt
        self.frac = frac
        assert frac.shape == (self.Nt * n_total_pft * land_pts,)

        self.gpu_compute.set_data(
            ignitionMethod=ignition_method,
            flammabilityMethod=flammability_method,
            drynessMethod=dryness_method,
            fuelBuildUpMethod=fuel_build_up_method,
            includeTemperature=include_temperature,
            Nt=Nt,
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
            checks_failed=checks_failed,
        )
        self.out = self._get_out_arr()

    def _get_compute(self):
        return _GPUCompute()

    def get_checks_failed_mask(self):
        return self.gpu_compute.get_checks_failed_mask().reshape(
            self.Nt, npft, land_pts
        )

    def get_diagnostics(self):
        return self.gpu_compute.get_diagnostics()

    def _get_out_arr(self):
        return np.empty((self.Nt, land_pts), dtype=np.float32)

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


class GPUInfernoAvg(GPUInferno):
    def __init__(self, *args, weights, **kwargs):
        weights[weights < 1e-9] = 0

        self.weights = weights
        super().__init__(*args, **kwargs)

    def _get_compute(self):
        return _GPUInfernoAvg(land_pts, self.weights.astype(np.float32))

    def _get_out_arr(self):
        return np.empty((12, land_pts), dtype=np.float32)


class GPUInfernoAvgScore(GPUInferno):
    def __init__(self, *args, weights, obs_data, obs_pftcrop, **kwargs):
        weights[weights < 1e-9] = 0

        self.weights = weights
        self.obs_data = obs_data
        self.obs_pftcrop = obs_pftcrop
        super().__init__(*args, **kwargs)

    def _get_compute(self):
        return _GPUInfernoAvgScore(
            land_pts,
            self.weights.astype(np.float32),
            self.obs_data.astype(np.float32),
            self.obs_pftcrop.astype(np.float32),
        )

    def _get_out_arr(self):
        return None

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def score(
        self,
        *,
        crop_f,
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
        score_info = self.gpu_compute.run(overall_scale, crop_f)
        return dict(
            arcsinh_nme=score_info[0],
            mpd=score_info[1],
            mpd_ignored=score_info[2],
        )


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
        weights[weights < 1e-9] = 0

        self.L = L
        self.M = weights.shape[0]
        self.N = weights.shape[1]
        self._GPUConsAvg = _GPUConsAvg(L, np.asarray(weights, dtype=np.float32))

    def run(self, data, mask):
        out_data, out_mask = self._GPUConsAvg.run(
            np.asarray(data, dtype=np.float32), mask
        )
        return out_data.reshape((self.N, self.L)), out_mask.reshape((self.N, self.L))


class GPUConsAvgNoMask:
    def __init__(self, L, weights):
        weights[weights < 1e-9] = 0

        self.L = L
        self.M = weights.shape[0]
        self.N = weights.shape[1]
        self._GPUConsAvgNoMask = _GPUConsAvgNoMask(
            L, np.asarray(weights, dtype=np.float32)
        )

    def run(self, data):
        return self._GPUConsAvgNoMask.run(np.asarray(data, dtype=np.float32)).reshape(
            (self.N, self.L)
        )


def cpp_nme(*, obs, pred):
    return _nme(
        obs=np.asarray(obs, dtype=np.float32), pred=np.asarray(pred, dtype=np.float32)
    )


def cpp_cons_avg_no_mask_inplace(*, weights, data, out):
    _cons_avg_no_mask(
        weights=np.asarray(weights, dtype=np.float32),
        data=np.asarray(data, dtype=np.float32),
        out=out,
    )


class GPUSA:

    no_pft_data = ("dry_days", "obs_pftcrop_1d")

    def __init__(
        self,
        *,
        Nt,
        drynessMethod,
        fuelBuildUpMethod,
        includeTemperature,
        overallScale,
        crop_f,
    ):
        self.Nt = Nt
        self.drynessMethod = drynessMethod
        self.fuelBuildUpMethod = fuelBuildUpMethod
        self.includeTemperature = includeTemperature
        self.overallScale = overallScale
        self.crop_f = crop_f

        self.gpu_sa = _GPUSACompute(Nt)
        self.max_n_samples = self.gpu_sa.getMaxNSamples()

        self.out_arr = np.empty(
            (self.max_n_samples, Nt),
            dtype=np.float32,
        )

        self.sample_data = dict(
            # Data.
            t1p5m_tile=np.empty((self.max_n_samples, Nt, npft), dtype=np.float32),
            frac=np.empty((self.max_n_samples, Nt, npft), dtype=np.float32),
            fuel_build_up=np.empty((self.max_n_samples, Nt, npft), dtype=np.float32),
            fapar_diag_pft=np.empty((self.max_n_samples, Nt, npft), dtype=np.float32),
            grouped_dry_bal=np.empty(
                (self.max_n_samples, Nt, N_pft_groups), dtype=np.float32
            ),
            litter_pool=np.empty((self.max_n_samples, Nt, npft), dtype=np.float32),
            dry_days=np.empty((self.max_n_samples, Nt), dtype=np.float32),
            obs_pftcrop_1d=np.empty((self.max_n_samples, Nt), dtype=np.float32),
            # Params.
            fapar_factor=np.empty((self.max_n_samples, N_pft_groups), dtype=np.float32),
            fapar_centre=np.empty((self.max_n_samples, N_pft_groups), dtype=np.float32),
            fapar_shape=np.empty((self.max_n_samples, N_pft_groups), dtype=np.float32),
            fuel_build_up_factor=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            fuel_build_up_centre=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            fuel_build_up_shape=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            temperature_factor=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            temperature_centre=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            temperature_shape=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            dry_day_factor=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            dry_day_centre=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            dry_day_shape=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            dry_bal_factor=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            dry_bal_centre=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            dry_bal_shape=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            litter_pool_factor=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            litter_pool_centre=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            litter_pool_shape=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            fapar_weight=np.empty((self.max_n_samples, N_pft_groups), dtype=np.float32),
            dryness_weight=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            temperature_weight=np.empty(
                (self.max_n_samples, N_pft_groups), dtype=np.float32
            ),
            fuel_weight=np.empty((self.max_n_samples, N_pft_groups), dtype=np.float32),
        )

        self._check_n_samples = {key: 0 for key in self.sample_data}

        # Ensure all arrays have the PFT dimension last (with certain exceptions).
        for name, arr in self.sample_data.items():
            if name in self.no_pft_data:
                continue
            assert arr.shape[-1] in (npft, N_pft_groups)

    def set_sample_data(self, index_data, values, n_samples, offset_data=None):
        if offset_data is not None:
            assert values.ndim == offset_data.ndim
            values = values + offset_data

        name, s, dims = itemgetter("name", "slice", "dims")(index_data)

        try:
            pft_index = dims.index(Dims.PFT)
            pft_s = s[pft_index]
        except ValueError:
            if name in self.no_pft_data:
                # Expect no PFT index.
                pft_index = None
                pft_s = None
            else:
                raise

        if Dims.TIME in dims:
            # Ensure all temporal values are targeted.
            assert s[dims.index(Dims.TIME)] == slice(None)

        if Dims.LAND in dims:
            # Ensure only a single land point is targeted.
            assert isinstance(s[dims.index(Dims.LAND)], (int, np.integer))

        if dims == (Dims.TIME, Dims.PFT, Dims.LAND):
            self.sample_data[name][:n_samples, :, pft_s] = values
        elif dims == (Dims.TIME, Dims.LAND):
            self.sample_data[name][:n_samples, :] = values
        elif dims == (Dims.PFT,):
            self.sample_data[name][:n_samples, pft_s] = values
        else:
            raise ValueError(f"Unhandled dims '{dims}'.")

        self._check_n_samples[name] = n_samples

    def run(
        self,
        *,
        nSamples,
    ):
        assert len(set(self._check_n_samples.values())) == 1
        assert next(iter(self._check_n_samples.values())) == nSamples
        assert nSamples <= self.max_n_samples
        assert nSamples >= 1

        self.gpu_sa.setData(
            drynessMethod=self.drynessMethod,
            fuelBuildUpMethod=self.fuelBuildUpMethod,
            includeTemperature=self.includeTemperature,
            nSamples=nSamples,
            overallScale=self.overallScale,
            crop_f=self.crop_f,
            **self.sample_data,
        )
        self.gpu_sa.run(out=self.out_arr)
        return self.out_arr[:nSamples]

    def release(self):
        self.gpu_sa.release()
