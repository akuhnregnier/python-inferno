# -*- coding: utf-8 -*-
import numpy as np
import pytest

from python_inferno import py_gpu_inferno
from python_inferno.configuration import (
    N_pft_groups,
    n_cell_grp_pft,
    n_cell_nat_pft,
    n_cell_no_pft,
    n_cell_tot_pft,
)


@pytest.fixture
def compute():
    compute = py_gpu_inferno._GPUCompute()
    yield compute
    compute.release()


@pytest.mark.parametrize("includeTemperature", [0, 1])
@pytest.mark.parametrize("fuelBuildUpMethod", [1, 2])
@pytest.mark.parametrize("drynessMethod", [1, 2])
@pytest.mark.parametrize("flammabilityMethod", [1, 2])
@pytest.mark.parametrize("Nt", [100, 10])
def test_GPUCompute(
    compute,
    includeTemperature,
    fuelBuildUpMethod,
    drynessMethod,
    flammabilityMethod,
    Nt,
):

    compute.set_data(
        _ignitionMethod=1,
        _flammabilityMethod=flammabilityMethod,
        _drynessMethod=drynessMethod,
        _fuelBuildUpMethod=fuelBuildUpMethod,
        _includeTemperature=includeTemperature,
        _Nt=Nt,
        t1p5m_tile=np.zeros(n_cell_tot_pft(Nt), dtype=np.float32) + 300,
        q1p5m_tile=np.zeros(n_cell_tot_pft(Nt), dtype=np.float32),
        pstar=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        sthu_soilt_single=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        frac=np.zeros(n_cell_tot_pft(Nt), dtype=np.float32),
        c_soil_dpm_gb=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        c_soil_rpm_gb=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        canht=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
        ls_rain=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        con_rain=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        pop_den=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        flash_rate=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
        fuel_build_up=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
        fapar_diag_pft=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
        grouped_dry_bal=np.zeros(n_cell_grp_pft(Nt), dtype=np.float32),
        litter_pool=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
        dry_days=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
    )

    compute.set_params(
        fapar_factor=np.zeros(N_pft_groups, dtype=np.float32),
        fapar_centre=np.zeros(N_pft_groups, dtype=np.float32),
        fapar_shape=np.zeros(N_pft_groups, dtype=np.float32),
        fuel_build_up_factor=np.zeros(N_pft_groups, dtype=np.float32),
        fuel_build_up_centre=np.zeros(N_pft_groups, dtype=np.float32),
        fuel_build_up_shape=np.zeros(N_pft_groups, dtype=np.float32),
        temperature_factor=np.zeros(N_pft_groups, dtype=np.float32),
        temperature_centre=np.zeros(N_pft_groups, dtype=np.float32),
        temperature_shape=np.zeros(N_pft_groups, dtype=np.float32),
        dry_day_factor=np.zeros(N_pft_groups, dtype=np.float32),
        dry_day_centre=np.zeros(N_pft_groups, dtype=np.float32),
        dry_day_shape=np.zeros(N_pft_groups, dtype=np.float32),
        dry_bal_factor=np.zeros(N_pft_groups, dtype=np.float32),
        dry_bal_centre=np.zeros(N_pft_groups, dtype=np.float32),
        dry_bal_shape=np.zeros(N_pft_groups, dtype=np.float32),
        litter_pool_factor=np.zeros(N_pft_groups, dtype=np.float32),
        litter_pool_centre=np.zeros(N_pft_groups, dtype=np.float32),
        litter_pool_shape=np.zeros(N_pft_groups, dtype=np.float32),
    )

    compute.run()
