# -*- coding: utf-8 -*-

import numpy as np
import pytest

from python_inferno.ba_model import GPUBAModel
from python_inferno.configuration import (
    N_pft_groups,
    n_cell_grp_pft,
    n_cell_nat_pft,
    n_cell_no_pft,
    n_cell_tot_pft,
)
from python_inferno.py_gpu_inferno import _GPUCompute


@pytest.fixture
def compute():
    compute = _GPUCompute()
    yield compute
    compute.release()


@pytest.fixture
def get_compute_data():
    def _get_compute_data(
        includeTemperature=1,
        fuelBuildUpMethod=1,
        drynessMethod=1,
        flammabilityMethod=1,
        Nt=10,
    ):
        return dict(
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

    return _get_compute_data


@pytest.fixture
def compute_params():
    return dict(
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
        fapar_weight=np.ones(N_pft_groups, dtype=np.float32),
        dryness_weight=np.ones(N_pft_groups, dtype=np.float32),
        temperature_weight=np.ones(N_pft_groups, dtype=np.float32),
        fuel_weight=np.ones(N_pft_groups, dtype=np.float32),
    )


@pytest.mark.parametrize("includeTemperature", [0, 1])
@pytest.mark.parametrize("fuelBuildUpMethod", [1, 2])
@pytest.mark.parametrize("drynessMethod", [1, 2])
@pytest.mark.parametrize("flammabilityMethod", [1, 2])
@pytest.mark.parametrize("Nt", [100, 10])
def test_GPUCompute_synthetic(
    get_compute_data,
    compute_params,
    compute,
    includeTemperature,
    fuelBuildUpMethod,
    drynessMethod,
    flammabilityMethod,
    Nt,
):
    compute.set_data(
        **get_compute_data(
            includeTemperature=includeTemperature,
            fuelBuildUpMethod=fuelBuildUpMethod,
            drynessMethod=drynessMethod,
            flammabilityMethod=flammabilityMethod,
            Nt=Nt,
        )
    )

    compute.set_params(**compute_params)

    compute.run()


@pytest.mark.timeout(45)
def test_many_run(compute, get_compute_data, compute_params):
    compute.set_data(**get_compute_data())
    for i in range(10000):
        compute.set_params(**compute_params)
        compute.run()


def test_GPUBAModel(params_model_ba):
    for params, expected_model_ba in params_model_ba:
        # Set up the model.
        model = GPUBAModel(**params)
        # Initialise the parameters using random values to ensure that modifying the
        # parameters later on works as expected.
        rng = np.random.default_rng(0)
        random_model_ba = model.run(
            **{
                key: rng.random(1)
                for key in (
                    *params.keys(),
                    "fapar_weight",
                    "dryness_weight",
                    "temperature_weight",
                    "fuel_weight",
                )
            }
        )["model_ba"]
        assert not np.allclose(
            random_model_ba, expected_model_ba["metal"], atol=1e-12, rtol=1e-7
        )
        # Set the proper parameters and run.
        model_ba = model.run(
            **{
                **dict(
                    fapar_weight=1,
                    dryness_weight=1,
                    temperature_weight=1,
                    fuel_weight=1,
                ),
                **params,
            }
        )["model_ba"]
        assert np.allclose(model_ba, expected_model_ba["metal"], atol=1e-12, rtol=1e-7)

        model.release()
